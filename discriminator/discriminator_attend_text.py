import tensorflow as tf
import layer_utils
import tensorflow.contrib.layers as layers
import math

from discriminator.discriminator import BaseDiscriminator


class DiscriminatorClassification(BaseDiscriminator):

    def _compute_reward_and_loss(self, reward_config):
        cls = self.build_classification_model()
        return self._compute_loss(cls.get_logits(), cls.get_rewards())

    def build_classification_model(self):
        self.attention_model.build(self.caption_input.get_embedding(),
                                   self.caption_input.get_not_null_count(),
                                   self.image_input.get_image_features(),
                                   scope='attention')
        return self.attention_model

    def _compute_loss(self, logits, rewards):

        labels = self.metadata_input.get_labels()
        labels = tf.one_hot(labels, depth=2)

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

        batch_loss = tf.reduce_mean(loss)

        masked_reward = rewards * self.caption_input.get_not_null_numeric_mask()
        mean_reward_for_each_sentence = tf.reduce_sum(masked_reward, axis=1) / self.caption_input.get_not_null_count()
        return batch_loss, self.pred(logits, rewards), mean_reward_for_each_sentence

    def pred(self, logits, rewards):
        return tf.expand_dims(tf.argmax(logits, axis=1), 1) * tf.ones(tf.shape(rewards), dtype=tf.int64)


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

        self.output = None
        self.alphas = None
        self.logits = None
        self.rewards = None

        self.attention_cname = "attention_results"

    def build(self, caption_input, not_null_count, image_input, scope):

        def get_mean_caption(caption_input):
            return tf.reduce_sum(caption_input, axis=1) / tf.expand_dims(not_null_count, axis=1)

        conv_output_size = 1024
        image_pooled = self._conv_layer(image_input, self.pooling_stride, conv_output_size)
        pooled_part_num = math.ceil(math.sqrt(self.image_part_num) / self.pooling_stride)
        total_regions = pooled_part_num * pooled_part_num
        image_parts = tf.reshape(image_pooled, [-1, total_regions, conv_output_size])

        with tf.variable_scope(scope):

            mean_caption = get_mean_caption(caption_input)
            init_hidden_state = layer_utils.affine_transform(mean_caption, self.hidden_dim, "init_h")
            init_cell_state = layer_utils.affine_transform(mean_caption, self.hidden_dim, "init_c")
            state = tf.nn.rnn_cell.LSTMStateTuple(init_cell_state, init_hidden_state)
            output = init_hidden_state

            sentence_length = tf.shape(caption_input)[1]

            with tf.variable_scope("attentive_lstm") as lstm_scope:
                output_seq = []
                alpha_seq = []
                for idx in range(total_regions):
                    # previous output context
                    prev_ctx = layer_utils.affine_transform(output, self.attention_dim, scope="prev_to_context")
                    prev_ctx = tf.tile(tf.expand_dims(prev_ctx, 1), [1, sentence_length, 1])

                    word_projection = layer_utils.affine_transform(caption_input, self.attention_dim,
                                                                   scope="words_to_ann")
                    ctx = tf.nn.relu(word_projection + prev_ctx)
                    ctx = tf.squeeze(layer_utils.affine_transform(ctx, 1, scope="context"), axis=2)
                    alpha = tf.nn.softmax(ctx)

                    weighted_ctx = tf.reduce_mean(word_projection * tf.expand_dims(alpha, 2), axis=1)

                    lstm = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, initializer=tf.random_normal_initializer(stddev=0.03))

                    image_part = image_parts[:, idx]
                    lstm_input = tf.concat([image_part, weighted_ctx], axis=1)
                    output, state = lstm(lstm_input, state)

                    output_seq.append(output)
                    alpha_seq.append(alpha)
                    lstm_scope.reuse_variables()

                self.output = tf.stack(output_seq, axis=1)
                all_alphas = tf.stack(alpha_seq, axis=1)

        self.alphas = tf.reduce_max(all_alphas, axis=1)

        tf.add_to_collection(self.attention_cname, self.alphas)
        self.logits = self.enriched_image_to_prediction(self.output, pooled_part_num, "2d_pooling")
        self.rewards = self.get_rewards_over_words()

    def get_output(self):
        return self.output

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
