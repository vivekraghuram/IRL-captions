import tensorflow as tf
import tensorflow.contrib.layers as layers
import layer_utils

from discriminator.discriminator import BaseDiscriminator


def select_from_sequence(sequence, sequence_length, index_pos):
    selector = tf.one_hot(tf.cast(index_pos, tf.int32), depth=sequence_length)
    selector = tf.expand_dims(selector, axis=2)
    final_lstm_output = tf.reduce_sum(selector * sequence, axis=1)
    return final_lstm_output


def conv_image_context(image_input, output_size, k_size=2, dilation=1):
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


def get_rewards_over_words(logits, alphas):
    prob_demo = tf.nn.softmax(logits)[:, 1]
    prob_demo = tf.expand_dims(prob_demo, dim=1)
    return alphas * prob_demo


class Learner(object):
    def get_other_info(self):
        return dict()


class VanillaDotProduct(Learner):
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

        self.rewards = get_rewards_over_words(self.logits, alphas=(word_relevancy * 0) + 1)

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
            loss = 1 - self.learner_model.get_alphas()
            masked_loss = loss * self.caption_input.get_not_null_numeric_mask()

            mean_loss = tf.reduce_sum(masked_loss, axis=-1) / self.caption_input.get_not_null_count()
            batch_loss = tf.reduce_mean(mean_loss)
            batch_loss = batch_loss * 0.1
            self.secondary_loss = batch_loss
            return self.secondary_loss
        else:
            self.secondary_loss = tf.constant(0, dtype=tf.float32)
            return self.secondary_loss

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

    def _get_other_info_map(self):
        map = dict()
        map["secondary_loss"] = self.secondary_loss
        map.update(self.learner_model.get_other_info())
        return map


class TextualAttention(Learner):
    model_dilated = "dilated_ctx"
    model_relevancy_map = "rel_map"

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
        self.alpha_map = None
        self.logits = None
        self.rewards = None
        self.relevancy_map = None

        self.attention_cname = "attention_results"

    def _caption_lstm_output(self, caption_input, not_null_count, hidden_dim, scope, is_bidirectional=False):

        with tf.variable_scope("bi_{}".format(scope)):
            init_state_template = tf.identity(caption_input[:, 0])
            zero_init_state = layer_utils.affine_transform(init_state_template, hidden_dim, scope="init") * 0
            seq_len = tf.cast(not_null_count, tf.int32)

            cell_fw = tf.nn.rnn_cell.LSTMCell(hidden_dim)
            init_fw = tf.nn.rnn_cell.LSTMStateTuple(zero_init_state, zero_init_state)

            if is_bidirectional:
                cell_bw = tf.nn.rnn_cell.LSTMCell(hidden_dim)
                init_bw = tf.nn.rnn_cell.LSTMStateTuple(zero_init_state, zero_init_state)

                bi_lstm_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, caption_input,
                                                                    sequence_length=seq_len,
                                                                    initial_state_fw=init_fw, initial_state_bw=init_bw,
                                                                    dtype=tf.float32, time_major=False)
                concat_bi_ouput = tf.concat([bi_lstm_output[0], bi_lstm_output[1]], axis=2)
                print("bi-lstm: ", concat_bi_ouput)
                return concat_bi_ouput
            else:
                output, _ = tf.nn.dynamic_rnn(cell_fw, caption_input, sequence_length=seq_len, initial_state=init_fw,
                                              dtype=tf.float32, time_major=False)
                return output

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

        which_model = TextualAttention.model_dilated

        print("Building {}..".format(which_model))
        if which_model == TextualAttention.model_dilated:
            self.build_on_dilated_img_ctx(caption_input, image_input, not_null_count, scope)
        elif which_model == TextualAttention.model_relevancy_map:
            self.build_relevancy_map(caption_input, image_input, not_null_count, scope)

    def get_alphas(self, graph=None):
        if graph:
            return graph.get_collection(self.attention_cname)[0]
        else:
            return self.alphas

    def get_other_info(self):
        return {
            "rel_map": self.relevancy_map,
            "alpha_map": self.alpha_map
        }

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

    def get_logits(self):
        return self.logits

    def get_rewards(self):
        return self.rewards

    def _conv_layer(self, img, pool_stride, output_num):

        with tf.variable_scope("convnet"):
            out = layers.convolution2d(img, num_outputs=output_num, kernel_size=2, stride=1, activation_fn=tf.nn.relu)
            out = layers.max_pool2d(out, kernel_size=(2, 2), stride=pool_stride, padding='SAME')
            return out

    def build_on_dilated_img_ctx(self, caption_input, image_input, not_null_count, scope):

        not_null_count = not_null_count - 1
        rel_score = "multiplicative"
        apply_alpha = "sum_with_weight"

        # reduce the size of image
        # consider just simply max-pooling to reduce image size without further convolving
        conv_output_size = 1024
        image_input = self._conv_layer(image_input, self.pooling_stride, conv_output_size)
        print("image_input: ", image_input)

        ctx_dim = self.hidden_dim
        sentence_len = tf.shape(caption_input)[1]
        img_width = tf.shape(image_input)[1]

        lstm_ouput = self._caption_lstm_output(caption_input, not_null_count, hidden_dim=self.hidden_dim/2, is_bidirectional=True,scope="lstm_output")
        lstm_ouput = layer_utils.affine_transform(lstm_ouput, ctx_dim, scope="lstm_fw_bw")
        print("lstm output: ", lstm_ouput)

        img_ctx = conv_image_context(image_input, ctx_dim)
        print("img_ctx: ", img_ctx)
        expanded_img_ctx = tf.expand_dims(img_ctx, axis=3)

        if rel_score == "additive":
            lstm_word_ctx = layer_utils.affine_transform(lstm_ouput, ctx_dim, scope="lstm_ctx")
            print("lstm word ctx: ", lstm_word_ctx)
            expanded_lstm_ctx = tf.expand_dims(tf.expand_dims(lstm_word_ctx, axis=1), axis=2)
            print("tiled img: ", expanded_img_ctx)
            print("tiled lstm: ", expanded_lstm_ctx)
            ctx_map = layer_utils.affine_transform(tf.nn.relu(expanded_img_ctx + expanded_lstm_ctx), 1, scope="att")
            ctx_map = tf.squeeze(ctx_map, axis=-1)
        elif rel_score == "multiplicative":
            print("Multiplicative style: ")
            expanded_lstm_ctx = tf.expand_dims(tf.expand_dims(lstm_ouput, axis=1), axis=2)
            element_wise_mul = tf.nn.relu(expanded_img_ctx + expanded_lstm_ctx)
            ctx_map = tf.reduce_sum(element_wise_mul, axis=4)
        print("ctx map: ", ctx_map)

        alphas = self.restricted_softmax_on_map(ctx_map, sentence_len, not_null_count)

        expanded_alpha = tf.expand_dims(alphas, axis=4)
        weighted_captions = self.get_weighted_caption(apply_alpha, caption_input, ctx_dim, expanded_alpha, img_width,
                                                      lstm_ouput, None, not_null_count)
        print("weighed contxt: ", weighted_captions)

        repre_lstm = select_from_sequence(lstm_ouput, sentence_len, not_null_count * 0)
        repre_lstm = (repre_lstm + select_from_sequence(lstm_ouput, sentence_len, not_null_count)) / 2
        repre_lstm_expanded = tf.expand_dims(tf.expand_dims(repre_lstm, axis=1), axis=2)
        repre_lstm_expanded = tf.tile(repre_lstm_expanded, [1, img_width, img_width, 1])
        print("last expanded: ", repre_lstm_expanded)

        to_dot = weighted_captions + repre_lstm_expanded

        print("to dot: ", to_dot)

        with tf.variable_scope("img_proj1"):
            image_proj1 = layer_utils.affine_transform(image_input, self.hidden_dim, scope)

        relevancy_map = tf.reduce_sum(to_dot * image_proj1, axis=3)

        print("relevancy map: ", relevancy_map)
        flat_rel = layers.flatten(relevancy_map)
        print("flat rel: ", flat_rel)
        logits = layer_utils.affine_transform(flat_rel, 2, scope="rel_to_logits")
        self.logits = logits
        print("logits: ", self.logits)

        # reward assignment
        summed_alpha = tf.reduce_sum(tf.reduce_sum(alphas, axis=1), axis=1)
        overall_alpha = summed_alpha / tf.cast(img_width * img_width, tf.float32)
        self.alpha_map = alphas
        self.alphas = overall_alpha
        print("overall alpha:", overall_alpha)
        tf.add_to_collection(self.attention_cname, self.alphas)

        self.rewards = get_rewards_over_words(self.logits, self.alphas)
        self.relevancy_map = relevancy_map

    def get_weighted_caption(self, apply_alpha, caption_input, ctx_dim, expanded_alpha, img_width, lstm_ouput,
                             lstm_word_ctx, not_null_count):

        if apply_alpha == "sum_with_weight":
            print("Expected lstm output")
            expanded_lstm_caption = tf.expand_dims(tf.expand_dims(lstm_ouput, axis=1), axis=2)
            tiled_lstm = tf.tile(expanded_lstm_caption, [1, img_width, img_width, 1, 1])
            weighted_captions = tf.reduce_sum(tiled_lstm * expanded_alpha, axis=3)

        elif apply_alpha == "embedding":
            print("Weighting raw embedding")
            transform_embedding = layer_utils.affine_transform(caption_input, self.hidden_dim,
                                                               scope="embedding_to_hidden")
            expanded_caption_input = tf.expand_dims(tf.expand_dims(transform_embedding, axis=1), axis=2)
            tiled_input = tf.tile(expanded_caption_input, [1, img_width, img_width, 1, 1])
            weighted_captions = tf.reduce_sum(tiled_input * expanded_alpha, axis=3)
        elif apply_alpha == "different_lstm":
            print("Different lstm")

            different_lstm = self._caption_lstm_output(caption_input, not_null_count, hidden_dim=self.hidden_dim,
                                                       scope="different_lstm")
            expanded_lstm_caption = tf.expand_dims(tf.expand_dims(different_lstm, axis=1), axis=2)
            tiled_lstm = tf.tile(expanded_lstm_caption, [1, img_width, img_width, 1, 1])
            weighted_captions = tf.reduce_sum(tiled_lstm * expanded_alpha, axis=3)
        else:
            print("Same lstm context is now weighted...")
            expanded_lstm_caption = tf.expand_dims(tf.expand_dims(lstm_word_ctx, axis=1), axis=2)
            tiled_lstm = tf.tile(expanded_lstm_caption, [1, img_width, img_width, 1, 1])
            weighted_captions = tf.reduce_sum(tiled_lstm * expanded_alpha, axis=3)

        return weighted_captions

    def build_relevancy_map(self, caption_input, image_input, not_null_count, scope):

        lstm_ouput = self._caption_lstm_output(caption_input, not_null_count, self.hidden_dim,
                                               scope="lstm_output", is_bidirectional=True)

        sentence_length = tf.shape(caption_input)[1]
        mid_lstm = select_from_sequence(lstm_ouput, sentence_length, (not_null_count / 2))

        with tf.variable_scope("img_proj"):
            image_proj = layer_utils.affine_transform(image_input, self.hidden_dim, scope)
        image_proj = tf.concat([image_proj, image_proj], axis=-1)

        final_output_expanded = tf.expand_dims(tf.expand_dims(mid_lstm, axis=1), axis=2)
        word_relevancy = tf.reduce_sum(final_output_expanded * image_proj, axis=3)

        print("final lstm expanded: ", final_output_expanded)
        print("word rel: ", word_relevancy)
        relevancy_map = tf.expand_dims(word_relevancy, axis=3)
        print("relevancy map: ", relevancy_map)
        conv_rel = layers.conv2d(relevancy_map, num_outputs=4, kernel_size=3, activation_fn=tf.nn.relu)
        flat_rel = layers.flatten(conv_rel)
        print("flat rel: ", flat_rel)
        logits = layer_utils.affine_transform(flat_rel, 2, scope="rel_to_logits")

        mean_img = tf.reduce_mean(tf.reduce_mean(image_proj, axis=2), axis=1)
        self.alphas = tf.reduce_sum(tf.expand_dims(mean_img, axis=1) * lstm_ouput, axis=2)
        tf.add_to_collection(self.attention_cname, self.alphas)
        self.logits = logits
        print("logits: ", self.logits)

        self.rewards = self.get_rewards_over_words()