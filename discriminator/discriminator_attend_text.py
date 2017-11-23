import tensorflow as tf
import tensorflow.contrib.layers as layers
import layer_utils
import math

from discriminator.discriminator import BaseDiscriminator


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

        sentence_length = tf.shape(caption_input)[1]
        selector = tf.one_hot(tf.cast(not_null_count - 1, tf.int32), depth=sentence_length)
        selector = tf.expand_dims(selector, axis=2)

        final_lstm_output = tf.reduce_sum(selector * self.lstm_outputs, axis=1)
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
            masked_alpha = self.learner_model.get_alphas() * self.caption_input.get_not_null_numeric_mask()
            loss = tf.reduce_mean(tf.square(1 - tf.reduce_sum(masked_alpha, axis=1)))
            return 0.05 * loss
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

    def _caption_lstm_output(self, caption_input, scope):

        init_state = layer_utils.affine_transform(tf.reduce_max(caption_input, axis=1), self.hidden_dim, scope="init")

        initial_lstm_state = tf.nn.rnn_cell.LSTMStateTuple(init_state * 0, init_state*0)

        with tf.variable_scope(scope):
            cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dim)
            lstm_output, _ = tf.nn.dynamic_rnn(cell, caption_input, time_major=False, dtype=tf.float32,
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

    def build(self, caption_input, not_null_count, image_input, scope):

        image_parts = self._image_layers(image_input) # dim 2048
        total_regions = len(image_parts)

        word_projection = self._caption_lstm_output(caption_input, "lstm_words")

        with tf.variable_scope(scope):
            prev_img = image_parts[0]

            sentence_length = tf.shape(caption_input)[1]

            with tf.variable_scope("attentive_lstm") as lstm_scope:
                output_seq = []
                alpha_seq = []
                for idx in range(1, total_regions):
                    # previous output context
                    prev_ctx = tf.tile(tf.expand_dims(prev_img, 1), [1, sentence_length, 1])

                    # ctx = tf.nn.relu(word_projection + prev_ctx)
                    # ctx = tf.squeeze(layer_utils.affine_transform(ctx, 1, scope="context"), axis=2)

                    ctx = tf.reduce_sum(word_projection * prev_ctx, axis=2)

                    alpha = tf.nn.softmax(ctx)

                    weighted_ctx = tf.reduce_mean(word_projection, axis=1)

                    image_part = image_parts[idx]
                    print("img part: ", image_part)
                    print("weighted contx: ", weighted_ctx)
                    output = tf.concat([image_part, weighted_ctx], axis=1)

                    output_seq.append(output)
                    alpha_seq.append(alpha)
                    lstm_scope.reuse_variables()

                    prev_img = image_part

                final_output = tf.concat(output_seq, axis=1)
                print("final output: ", final_output)
                all_alphas = tf.stack(alpha_seq, axis=1)

        self.alphas = tf.reduce_max(all_alphas, axis=1)

        tf.add_to_collection(self.attention_cname, self.alphas)
        self.logits = layer_utils.build_mlp(final_output, output_size=2, size=512, scope=scope,activation=tf.nn.relu)
        print("logits: ", self.logits)
        self.rewards = self.get_rewards_over_words()

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
