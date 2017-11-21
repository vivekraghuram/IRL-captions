import tensorflow as tf
import layer_utils
import tensorflow.contrib.layers as layers
import math

class AttentionOverText(object):
    def __init__(self,
                 max_sentence_length,
                 image_part_num,
                 image_feature_dim,
                 attention_dim):

        """
        :param max_sentence_length:
        :param image_part_num: image part num here is assumed to perfect square
        :param image_feature_dim:
        :param attention_dim:
        """
        self.max_sentence_length = max_sentence_length
        self.attention_dim = attention_dim
        self.image_part_num = image_part_num
        self.image_feature_dim = image_feature_dim
        self.pooling_stride = 2

        self.output = None
        self.alphas = None
        self.logits = None
        self.rewards = None

        self.attention_cname = "attention_results"

    def build(self, caption_input, image_input, hidden_dim, scope):

        image_pooled = self.conv_layer(image_input, self.pooling_stride)
        pooled_part_num = math.ceil(math.sqrt(self.image_part_num)/self.pooling_stride)
        total_regions = pooled_part_num * pooled_part_num
        print(total_regions)
        image_parts = tf.reshape(image_pooled, [-1, total_regions, self.image_feature_dim])

        with tf.variable_scope(scope):
            # mean embedding as initial state
            init_hidden_state = layer_utils.affine_transform(tf.reduce_mean(caption_input, axis=1), hidden_dim,
                                                             "init_h")
            init_cell_state = layer_utils.affine_transform(tf.reduce_mean(caption_input, axis=1), hidden_dim,
                                                           "init_c")
            state = tf.nn.rnn_cell.LSTMStateTuple(init_cell_state, init_hidden_state)

            output = init_hidden_state
            with tf.variable_scope("attentive_lstm") as lstm_scope:
                output_seq = []
                alpha_seq = []
                for idx in range(total_regions):
                    # previous output context
                    prev_ctx = layer_utils.affine_transform(output, self.attention_dim, scope="prev_to_context")
                    prev_ctx = tf.tile(tf.expand_dims(prev_ctx, 1), [1, self.max_sentence_length, 1])

                    word_projection = layer_utils.affine_transform(caption_input, self.attention_dim,
                                                                   scope="words_to_ann")
                    ctx = tf.nn.relu(word_projection + prev_ctx)
                    ctx = tf.squeeze(layer_utils.affine_transform(ctx, 1, scope="context"), axis=2)
                    alpha = tf.nn.softmax(ctx)

                    weighted_ctx = tf.reduce_mean(caption_input * tf.expand_dims(alpha, 2), axis=1)

                    lstm = tf.nn.rnn_cell.LSTMCell(hidden_dim, initializer=tf.random_normal_initializer(stddev=0.03))

                    image_part = image_parts[:, idx]
                    output, state = lstm(tf.concat([image_part, weighted_ctx], axis=1), state)

                    output_seq.append(output)
                    alpha_seq.append(alpha)
                    lstm_scope.reuse_variables()

                self.output = tf.stack(output_seq, axis=1)
                self.alphas = tf.stack(alpha_seq, axis=1)

        tf.add_to_collection(self.attention_cname, self.alphas)

        self.logits = self.enriched_image_to_prediction(self.output, "2d_pooling")
        self.rewards = self.get_rewards_over_words()

    def get_output(self):
        return self.output

    def get_alphas(self, graph=None):
        if graph:
            return graph.get_collection(self.attention_cname)[0]
        else:
            return self.alphas

    def enriched_image_to_prediction(self, flat_img, scope, num_class=2, reuse=False):

        un_flattened = tf.reshape(flat_img, [-1, self.image_part_num, self.image_part_num, self.image_feature_dim])
        with tf.variable_scope(scope, reuse=reuse):
            with tf.variable_scope("max_pooling"):
                out = layers.max_pool2d(un_flattened, kernel_size=(2, 2))
            out = layers.flatten(out)
            with tf.variable_scope("label_logits"):
                logits = layer_utils.build_mlp(out, output_size=num_class, size=4096, scope=scope,
                                               activation=tf.nn.relu)
            return logits

    def get_rewards_over_words(self):
        max_alpha_over_word = tf.reduce_max(self.get_alphas(), axis=1)
        prob_demo = tf.nn.softmax(self.logits)[:, 1]
        return max_alpha_over_word * prob_demo

    def get_logits(self):
        return self.logits

    def get_rewards(self):
        return self.rewards

    def conv_layer(self, img, pool_stride):

        with tf.variable_scope("convnet"):
            out = layers.convolution2d(img, num_outputs=1024, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
            out = layers.max_pool2d(out, kernel_size=(2, 2), stride=pool_stride, padding='SAME')
            return out
