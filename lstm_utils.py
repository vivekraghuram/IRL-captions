import numpy as np
import tensorflow as tf
import layer_utils

class LSTM(object):

  def __init__(self,
               NULL_ID,
               START_ID,
               END_ID,
               hidden_dim=512,
               vocab_dim=1004,
               image_feature_dim=4096,
               word_embedding_dim=256,
               learning_rate=5e-5,
               batch_size=50,
               num_layers=1):

    self.NULL_ID = NULL_ID
    self.START_ID = START_ID
    self.END_ID = END_ID
    self.hidden_dim = hidden_dim
    self.vocab_dim = vocab_dim
    self.image_feature_dim = image_feature_dim
    self.word_embedding_dim = word_embedding_dim
    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.num_layers = num_layers

    self.sy_keep_prob = tf.placeholder(tf.float32, shape=[], name="is_initial_step")
    self.sy_caption_input = tf.placeholder(shape=[self.batch_size, None], name="caption_input", dtype=tf.int32)
    self.sy_image_input = tf.placeholder(shape=[self.batch_size, self.image_feature_dim], name="image_feat_input", dtype=tf.float32)
    self.sy_caption_target = tf.placeholder(shape=[self.batch_size, None], name="caption_target", dtype=tf.int64)
    self.sy_target_mask = tf.placeholder(shape=[self.batch_size, None], name="mask_not_null", dtype=tf.bool)
    self.sy_is_initial_step = tf.placeholder(shape=[], name="is_initial_step", dtype=tf.bool)

    self.sy_hidden_states = [None] * self.num_layers
    self.sy_cell_states = [None] * self.num_layers
    for i in range(self.num_layers)
      self.sy_hidden_states[i] = tf.placeholder(shape=[self.batch_size, self.hidden_dim], name="hidden_lstm_state-%i"%i, dtype=tf.float32)
      self.sy_cell_states[i] = tf.placeholder(shape=[self.batch_size, self.hidden_dim], name="cell_lstm_state-%i"%i, dtype=tf.float32)


  def build_model(self):
    # Set up word embedding inputs
    embedding = tf.get_variable("embedding", [self.vocab_dim, self.word_embedding_dim], dtype=tf.float32)
    word_inputs = tf.nn.dropout(tf.nn.embedding_lookup(embedding, self.sy_caption_input), self.sy_keep_prob)

    # Set up image projection as LSTM initial cell state
    image_projection = layer_utils.affine_transform(self.sy_image_input, self.hidden_dim, 'image_proj')

    # Set up training target captions
    target_one_hot = tf.one_hot(self.sy_caption_target, self.vocab_dim, dtype=tf.int32)

    cell = tf.contrib.rnn.LSTMBlockCell(self.hidden_dim, forget_bias=0.0)
    cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(self.num_layers)], state_is_tuple=True)
    initial_state = tf.cond(self.sy_is_initial_step > 0,
                            lambda: tuple([tf.nn.rnn_cell.LSTMStateTuple(image_projection * 0, image_projection)] * self.num_layers),
                            lambda: tuple([tf.nn.rnn_cell.LSTMStateTuple(self.sy_cell_states[i], self.sy_hidden_states[i] for i in range(self.num_layers)]))
    self.output_hidden_state, self.output_cell_state = tf.contrib.rnn.dynamic_rnn(cell, word_inputs, initial_state=initial_state)

    hidden_to_word = layer_utils.affine_transform(output_hidden_state, vocab_dim, 'hidden_to_word')
    raw_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=target_one_hot, logits=hidden_to_word)

    # Apply not-null mask to ignore cross-entropy on trailing padding
    mask_not_null = tf.cast(sy_target_mask, dtype=tf.float32)
    masked_cross_entropy = raw_cross_entropy * mask_not_null

    # Get average cross entropy over caption
    self.cross_entropy = tf.reduce_mean(tf.reduce_sum(masked_cross_entropy, axis=1))
    self.update_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Prediction accuracy
    num_predictions = tf.cast(tf.reduce_sum(mask_not_null), tf.float32)
    prediction = tf.argmax(hidden_to_word, axis = 2)
    self.accuracy = tf.reduce_sum(tf.cast(tf.equal(prediction, sy_caption_target), tf.float32)) / num_predictions

  def train(self, data, steps):
    for i in range(step):
      mini_batch, image_features, url = sample_coco_minibatch(data, batch_size=self.batch_size, split='train')
      train_captions, target_captions, target_mask = get_train_target_caption(mini_batch, self.NULL_ID)
      _, c, a = self.train_step()

  def train_step(keep_prob=1.0):
    return sess.run([update_op, cross_entropy, accuracy], feed_dict=
                     {sy_image_input: image_features,
                      sy_caption_input: train_captions,
                      sy_caption_target: target_captions,
                      sy_target_mask: target_mask,
                      sy_is_initial_step: True,
                      sy_hidden_state: dummy_hidden_state,
                      sy_cell_state: dummy_cell_state
                     })

  def test_step():
    pass
