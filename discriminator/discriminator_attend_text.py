import tensorflow as tf
import tensorflow.contrib.layers as layers
import layer_utils
import math

from discriminator.discriminator import BaseDiscriminator


def select_from_sequence(sequence, sequence_length, index_pos):
    selector = tf.one_hot(tf.cast(index_pos, tf.int32), depth=sequence_length)
    selector = tf.expand_dims(selector, axis=2)
    final_lstm_output = tf.reduce_sum(selector * sequence, axis=1)
    return final_lstm_output


def conv_image_context(image_input, output_size, k_size=2, dilation=2):
    return layers.conv2d(image_input, num_outputs=output_size, kernel_size=k_size, activation_fn=tf.nn.relu,
                         rate=dilation)

def restricted_softmax_on_sequence(logits, sentence_length, not_null_count):
    large_neg = (logits * 0) - 100000
    sequence_mask = tf.sequence_mask(tf.cast(not_null_count, dtype=tf.int32), sentence_length)
    print("seq mask: ", sequence_mask)
    binary_mask = tf.cast(sequence_mask, dtype=tf.float32)

    large_neg_on_empty = large_neg * (1 - binary_mask)

    print("bin mask: ", binary_mask)
    return tf.nn.softmax(logits + large_neg_on_empty)


class VanillaDotProduct(object):
    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim

        self.lstm_outputs = None
        self.image_proj = None

        self.output = None
        self.logits = None
        self.rewards = None

    def build(self, caption_input, not_null_count, image_input, scope):
        # input here can consider excluding start/end tokens

        init_state = layer_utils.affine_transform(tf.reduce_max(caption_input, axis=1), self.hidden_dim, scope="init")
        initial_lstm_state = tf.nn.rnn_cell.LSTMStateTuple(init_state * 0, init_state * 0)

        with tf.variable_scope(scope):
            cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dim)
            lstm_output, _ = tf.nn.dynamic_rnn(cell, caption_input,
                                               sequence_length=not_null_count, time_major=False,
                                               dtype=tf.float32,
                                               initial_state=initial_lstm_state)

        self.lstm_outputs = lstm_output

        # image
        with tf.variable_scope("img_proj"):
            self.image_proj = layer_utils.affine_transform(image_input, self.hidden_dim, scope)

        outputs = self.lstm_outputs

        sentence_length = tf.shape(caption_input)[1]
        final_lstm_output = select_from_sequence(outputs, sentence_length, not_null_count - 1)

        single_dim_logit = tf.reduce_sum(final_lstm_output * self.image_proj, axis=1)
        single_dim_logit = tf.reshape(single_dim_logit, [-1, 1])

        self.logits = tf.concat([-1 * single_dim_logit, single_dim_logit], axis=1)  # pos 1 is demo

        word_relevancy = tf.reduce_sum(tf.expand_dims(self.image_proj, axis=1) * self.lstm_outputs, axis=2)
        self.rewards = word_relevancy

    def get_logits(self):
        return self.logits

    def get_rewards(self):
        return self.rewards


class DiscriminatorClassification(BaseDiscriminator):
    def _compute_reward_and_loss(self, reward_config):
        print("Building classification")
        cls = self.build_classification_model()
        self._logits = cls.get_logits()
        return self._compute_loss(self._logits, cls.get_rewards())

    def build_classification_model(self):

        if self.is_attention:
            scope = "cls_attention"
        else:
            scope = "cls"
        self.learner_model.build(self.caption_input.get_embedding(),
                                 self.caption_input.get_not_null_count(),
                                 self.image_input.get_image_features(),
                                 scope=scope)
        return self.learner_model

    def _secondary_loss(self):
        if self.is_attention:
            # masked_alpha = self.learner_model.get_alphas() * self.caption_input.get_not_null_numeric_mask()
            # loss = tf.reduce_mean(tf.square(1 - tf.reduce_sum(masked_alpha, axis=1)))
            # return 0.05 * loss
            return 0
        else:
            return 0

    def _compute_loss(self, logits, rewards):

        labels = self.metadata_input.get_labels()
        labels = tf.one_hot(labels, depth=2)

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

        batch_loss = tf.reduce_mean(loss)
        total_loss = batch_loss + self._secondary_loss()

        masked_reward = rewards * self.caption_input.get_not_null_numeric_mask()
        mean_reward_for_each_sentence = tf.reduce_sum(masked_reward, axis=1) / self.caption_input.get_not_null_count()
        return total_loss, rewards, mean_reward_for_each_sentence

    def _pred(self):
        argmax = tf.argmax(self._logits, axis=1)
        return argmax


class TextualAttention(object):
    def __init__(self,
                 max_sentence_length,
                 image_part_num,
                 image_feature_dim,
                 hidden_dim,
                 attention_dim):

        """
        :param max_sentence_length:
        :param image_part_num: image part num here is assumed to perfect square
        :param image_feature_dim:
        :param attention_dim:
        """
        self.attention_dim = attention_dim
        self.hidden_dim = hidden_dim
        self.image_part_num = image_part_num
        self.image_feature_dim = image_feature_dim
        self.pooling_stride = 2

        # image param
        self.conv_output_dim = 512
        self.img_mlp_hidden_size = 1024
        self.img_proj_dim = 2048

        self.alphas = None
        self.logits = None
        self.rewards = None

        self.attention_cname = "attention_results"

    def _caption_lstm_output(self, caption_input, not_null_count, scope):

        init_state = layer_utils.affine_transform(tf.reduce_max(caption_input, axis=1), self.hidden_dim, scope="init")
        initial_lstm_state = tf.nn.rnn_cell.LSTMStateTuple(init_state * 0, init_state * 0)

        with tf.variable_scope(scope):
            cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dim)
            lstm_output, _ = tf.nn.dynamic_rnn(cell, caption_input,
                                               sequence_length=not_null_count, time_major=False,
                                               dtype=tf.float32,
                                               initial_state=initial_lstm_state)
        return lstm_output

    def _image_layers(self, image_input):

        def multi_scale_layers(img_arg, stride, output_num, scope):
            with tf.variable_scope(scope):
                out = layers.convolution2d(img_arg, num_outputs=output_num, kernel_size=3, stride=stride,
                                           activation_fn=tf.nn.relu)
                out = layers.flatten(out)
                return out

        strides = [1, 2, 4]
        img_diff_res = []
        for s in strides:
            img_diff_res.append(multi_scale_layers(image_input, s, self.conv_output_dim, "conv_s{}".format(s)))

        img_projs = []
        for i, img in enumerate(img_diff_res):
            img_projs.append(layer_utils.build_mlp(img, self.hidden_dim,
                                                   n_layers=3 - i,
                                                   size=self.hidden_dim,
                                                   activation=tf.nn.relu,
                                                   scope="img_proj{}".format(i)))

        return img_projs

    @staticmethod
    def restricted_softmax_on_map(logits, sentence_length, not_null_count):

        large_neg = (logits * 0) - 100000
        sequence_mask = tf.sequence_mask(tf.cast(not_null_count, dtype=tf.int32), sentence_length)
        print("seq mask: ", sequence_mask)
        binary_mask = tf.cast(sequence_mask, dtype=tf.float32)
        binary_mask = tf.expand_dims(binary_mask, axis=1)
        binary_mask = tf.expand_dims(binary_mask, axis=2)
        large_neg_on_empty = large_neg * (1 - binary_mask)

        print("bin mask: ", binary_mask)
        return tf.nn.softmax(logits + large_neg_on_empty)



    def build(self, caption_input, not_null_count, image_input, scope):

        self.build_on_dilated_img_ctx(caption_input, image_input, not_null_count, scope)

    def get_alphas(self, graph=None):
        if graph:
            return graph.get_collection(self.attention_cname)[0]
        else:
            return self.alphas

    def enriched_image_to_prediction(self, flat_img, pooled_part_num, scope, num_class=2, reuse=False):

        un_flattened = tf.reshape(flat_img, [-1, pooled_part_num, pooled_part_num, self.hidden_dim])
        with tf.variable_scope(scope, reuse=reuse):
            with tf.variable_scope("max_pooling"):
                out = layers.max_pool2d(un_flattened, kernel_size=(2, 2))
            out = layers.flatten(out)
            with tf.variable_scope("label_logits"):
                logits = layer_utils.build_mlp(out, output_size=num_class, size=4096, scope=scope,
                                               activation=tf.nn.relu)
            return logits

    def get_rewards_over_words(self):

        prob_demo = tf.nn.softmax(self.logits)[:, 1]
        prob_demo = tf.expand_dims(prob_demo, dim=1)
        return self.alphas * prob_demo

    def get_logits(self):
        return self.logits

    def get_rewards(self):
        return self.rewards

    def _conv_layer(self, img, pool_stride, output_num):

        with tf.variable_scope("convnet"):
            out = layers.convolution2d(img, num_outputs=output_num, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
            out = layers.max_pool2d(out, kernel_size=(2, 2), stride=pool_stride, padding='SAME')
            return out

    def build_on_dilated_img_ctx(self, caption_input, image_input, not_null_count, scope):
        conv_output_size = 1024
        image_input = self._conv_layer(image_input, self.pooling_stride, conv_output_size)
        print("image_input: ", image_input)
        lstm_ouput = self._caption_lstm_output(caption_input, not_null_count, scope="lstm_output")
        sentence_len = tf.shape(caption_input)[1]
        img_width = tf.shape(image_input)[1]
        print("lstm output: ", lstm_ouput)
        ctx_dim = self.hidden_dim
        lstm_word_ctx = layer_utils.affine_transform(lstm_ouput, ctx_dim, scope="lstm_ctx")
        img_ctx = conv_image_context(image_input, ctx_dim)
        print("lstm word ctx: ", lstm_word_ctx)
        print("img_ctx: ", img_ctx)
        expanded_img_ctx = tf.expand_dims(img_ctx, axis=3)
        expanded_lstm_ctx = tf.expand_dims(tf.expand_dims(lstm_word_ctx, axis=1), axis=2)
        element_wise_mul = expanded_img_ctx * expanded_lstm_ctx
        # dot-product map (?, img_width, img_width, sen_len)
        ctx_map = tf.abs(tf.reduce_sum(element_wise_mul, axis=4))
        # alphas (?, img_width, img_width, sen_len)
        alphas = self.restricted_softmax_on_map(ctx_map, sentence_len, not_null_count)
        print("ctx map: ", ctx_map)
        # tiled lstm (?, img_width, img_width, sen_len, hidden_dim)
        tiled_lstm = tf.tile(expanded_lstm_ctx, [1, img_width, img_width, 1, 1])
        expanded_alpha = tf.expand_dims(alphas, axis=4)
        weighted_ctx = tf.reduce_sum(tiled_lstm * expanded_alpha, axis=3)  # (?, img_width, img_width, hidden_dim)
        print("weighed contxt: ", weighted_ctx)
        print("image input: ", image_input)
        # sum_alpha = tf.reduce_sum(tf.reduce_sum(alphas, axis=1),axis=1)
        # sum_alpha = tf.reduce_sum(sum_alpha, axis=1)
        max_alphas = tf.reduce_max(tf.reduce_max(alphas, axis=1), axis=1)
        overall_alpha = restricted_softmax_on_sequence(max_alphas, sentence_len, not_null_count)
        print("overall alpha:", overall_alpha)
        overall_output = layer_utils.affine_transform(lstm_ouput, ctx_dim, scope="lstm_overall")
        overall_output = tf.reduce_sum(tf.expand_dims(overall_alpha, axis=-1) * overall_output, axis=1)
        overall_output = tf.expand_dims(tf.expand_dims(overall_output, axis=1), axis=2)
        print("overall output: ", overall_output)
        self.alphas = overall_alpha
        with tf.variable_scope("img_proj1"):
            image_proj1 = layer_utils.affine_transform(image_input, self.hidden_dim, scope)
        relevancy_map = tf.reduce_sum(weighted_ctx * image_proj1, axis=3)
        with tf.variable_scope("img_proj"):
            image_proj2 = layer_utils.affine_transform(image_input, self.hidden_dim, scope)
        overall_rel = tf.reduce_sum(overall_output * image_proj2, axis=3)
        print("relevancy map: ", relevancy_map)
        conv_rel = layers.conv2d(relevancy_map, num_outputs=4, kernel_size=3, activation_fn=tf.nn.relu)
        flat_rel = layers.flatten(conv_rel)
        overall_conv_rel = layers.conv2d(overall_rel, num_outputs=4, kernel_size=3, activation_fn=tf.nn.relu)
        overall_flat_rel = layers.flatten(overall_conv_rel)
        print("flat rel: ", flat_rel)
        logits = layer_utils.affine_transform(tf.concat([flat_rel, overall_flat_rel], axis=1), 2, scope="rel_to_logits")
        print("final alpahas: ", self.alphas)
        # print("sum alphs: ", sum_alpha)
        tf.add_to_collection(self.attention_cname, self.alphas)
        self.logits = logits
        print("logits: ", self.logits)
        self.rewards = self.get_rewards_over_words()
