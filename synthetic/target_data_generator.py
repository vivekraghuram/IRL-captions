import tensorflow as tf
import numpy as np

class TargetDataGenerator(object):
    def __init__(self, vocab_size, batch_size, embedding_dim, hidden_dim, seq_len, start_token=0):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.start_token = np.array([start_token] * self.batch_size, dtype=np.int32)

        tf.set_random_seed(294)
        self._build_model()

    def _build_model(self):
        with tf.variable_scope("target_data_generator"):
            rnn_cell = tf.contrib.rnn.LSTMCell(self.hidden_dim, forget_bias=1.0, initializer=tf.random_normal_initializer())

            self.ph_input = tf.placeholder(tf.int32, (self.batch_size), "ph_input")
            embedding = tf.Variable(np.random.normal(size=[self.vocab_size, self.embedding_dim]), dtype=tf.float32)
            embedded_input = tf.nn.embedding_lookup(embedding, self.ph_input)

            self.ph_hidden_state = tf.placeholder(tf.float32, (self.batch_size, self.hidden_dim), "ph_hidden_state")
            self.ph_cell_state = tf.placeholder(tf.float32, (self.batch_size, self.hidden_dim), "ph_cell_state")
            state = tf.nn.rnn_cell.LSTMStateTuple(self.ph_cell_state, self.ph_hidden_state)

            rnn_output, self.rnn_state = rnn_cell(embedded_input, state)
            self.logits = tf.identity(tf.layers.dense(rnn_output, self.vocab_size,
                                                      kernel_initializer=tf.random_normal_initializer(),
                                                      bias_initializer=tf.random_normal_initializer()),
                                      name="rnn_logits")

            log_prob = tf.reshape(tf.log(tf.nn.softmax(self.logits)), (self.batch_size, self.vocab_size))
            self.next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)

            self.ph_target = tf.placeholder(tf.int32, (self.batch_size), "ph_target")
            target_one_hot = tf.one_hot(self.ph_target, self.vocab_size, dtype=tf.float32)
            raw_cross_entropy = target_one_hot * log_prob
            self.cross_entropy = -tf.reduce_mean(tf.reduce_sum(raw_cross_entropy, axis=1), name="cross_entropy")

    def generate_data(self, sess):
        sequences = np.zeros((self.batch_size, self.seq_len))
        sequences[:, 0] = self.start_token
        feed_dict = {
            self.ph_input: self.start_token,
            self.ph_hidden_state: np.zeros((self.batch_size, self.hidden_dim)),
            self.ph_cell_state: np.zeros((self.batch_size, self.hidden_dim))
        }
        for i in range(self.seq_len - 1):
            tokens, state = sess.run([self.next_token, self.rnn_state], feed_dict=feed_dict)

            sequences[:, i + 1] = tokens
            feed_dict[self.ph_input] = tokens
            feed_dict[self.ph_cell_state] = state[0]
            feed_dict[self.ph_hidden_state] = state[1]
        return sequences

    def evaluate(self, sess, data):
        data.set_mode('MLE').set_batch_size(self.batch_size)
        cross_entropy = 0.0
        count = 0

        for mini_batch in data.training_batches:
            image_features, caption_input, caption_targets, target_masks = mini_batch
            feed_dict = {
                self.ph_hidden_state : np.zeros((self.batch_size, self.hidden_dim)),
                self.ph_cell_state   : np.zeros((self.batch_size, self.hidden_dim))
            }

            for i in range(self.seq_len - 1):
                feed_dict[self.ph_input] = caption_input[:, i]
                feed_dict[self.ph_target] = caption_targets[:, i]

                ce, state = sess.run([self.cross_entropy, self.rnn_state], feed_dict=feed_dict)
                cross_entropy += ce
                count += 1
                feed_dict[self.ph_hidden_state] = state[1]
                feed_dict[self.ph_cell_state] = state[0]

        print("GT Loss: %.5f" % (cross_entropy / count))
        return cross_entropy / count
