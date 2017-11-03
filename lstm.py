import numpy as np
import tensorflow as tf

class LSTM(object):

  def __init__(self,
               hidden_dim=512,
               output_dim=1004,
              #  input_dim=4096,
               learning_rate=5e-5,
               batch_size=50,
               num_layers=1):

    self.hidden_dim = hidden_dim
    self.output_dim = output_dim # vocab_dim
    # self.input_dim = input_dim
    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.num_layers = num_layers

    # self.sy_keep_prob = tf.placeholder(tf.float32, shape=[], name="keep_prob")
    # self.sy_input = tf.placeholder(shape=[self.batch_size, None, self.input_dim], name="input", dtype=tf.float32)
    self.sy_target = tf.placeholder(shape=[self.batch_size, None], name="target", dtype=tf.int64)
    self.sy_target_mask = tf.placeholder(shape=[self.batch_size, None], name="mask_not_null", dtype=tf.float32)
    self.sy_initial_step = tf.placeholder(shape=[], name="is_initial_step", dtype=tf.bool)

    self.sy_hidden_states = [None] * self.num_layers
    self.sy_cell_states = [None] * self.num_layers
    for i in range(self.num_layers):
      self.sy_hidden_states[i] = tf.placeholder(shape=[self.batch_size, self.hidden_dim], name="hidden_state-%i"%i, dtype=tf.float32)
      self.sy_cell_states[i] = tf.placeholder(shape=[self.batch_size, self.hidden_dim], name="cell_state-%i"%i, dtype=tf.float32)

  def build_model(self, input_layer, initial_hidden_state=None, initial_cell_state=None, activation=None):
    if self.num_layers > 1:
      if initial_cell_state != None and initial_hidden_state != None:
        initial_state = tf.cond(self.sy_initial_step,
                                lambda: tuple([tf.nn.rnn_cell.LSTMStateTuple(
                                                                         initial_hidden_state,
                                                                         initial_cell_state
                                                                       ) for i in range(self.num_layers)]),
                                lambda: tuple([tf.nn.rnn_cell.LSTMStateTuple(
                                                                         self.sy_hidden_states[i],
                                                                         self.sy_cell_states[i]
                                                                       ) for i in range(self.num_layers)]))
      else:
        initial_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(
                                                self.sy_hidden_states[i],
                                                self.sy_cell_states[i]
                                              ) for i in range(self.num_layers)])
    elif initial_cell_state != None and initial_hidden_state != None:
      initial_state = tf.cond(self.sy_initial_step,
                              lambda: tf.nn.rnn_cell.LSTMStateTuple(initial_hidden_state, initial_cell_state),
                              lambda: tf.nn.rnn_cell.LSTMStateTuple(self.sy_hidden_states[0], self.sy_cell_states[0]))
    else:
      initial_state = tf.nn.rnn_cell.LSTMStateTuple(self.sy_hidden_states[0], self.sy_cell_states[0])


    cell = tf.contrib.rnn.LSTMBlockCell(self.hidden_dim, forget_bias=0.0)
    if self.num_layers > 1:
      cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(self.num_layers)], state_is_tuple=True)


    rnn_output, self.rnn_state = tf.nn.dynamic_rnn(cell, input_layer, initial_state=initial_state)

    logits = tf.layers.dense(inputs=rnn_output, units=self.output_dim, activation=activation)

    # Apply not-null mask to ignore cross-entropy on trailing padding
    target_one_hot = tf.one_hot(self.sy_target, self.output_dim, dtype=tf.int64)
    raw_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=target_one_hot, logits=logits)
    masked_cross_entropy = raw_cross_entropy * self.sy_target_mask

    # # Get average cross entropy over caption
    self.cross_entropy = tf.reduce_mean(tf.reduce_sum(raw_cross_entropy, axis=1))
    self.update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy)

    self.logits = logits
    self.predictions = tf.argmax(logits, axis = 2)
    correct_pred = tf.equal(self.predictions, self.sy_target)
    self.accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.float32) * self.sy_target_mask) / tf.reduce_sum(self.sy_target_mask)


  def train(self, sess, targets, target_masks, feed_dict):
    """
    sess: Initialized tensorflow session
    targets: numpy array consisting of batch_size targets
    target_mask: numpy array consisting batch_size masks
    feed_dict: dictionary with key and values entered for inputs and initial states already
    """
    assert(self.batch_size == targets.shape[0])
    assert(self.batch_size == target_masks.shape[0])

    # merge the dictionaries
    feed_dict = {**{
      self.sy_target       : targets,
      self.sy_target_mask  : target_masks,
      self.sy_initial_step : True
    }, **feed_dict}

    for i in range(self.num_layers):
      feed_dict[self.sy_hidden_states[i]] = np.zeros((self.batch_size, self.hidden_dim))
      feed_dict[self.sy_cell_states[i]] = np.zeros((self.batch_size, self.hidden_dim))

    _, c, a = sess.run([self.update_op, self.cross_entropy, self.accuracy], feed_dict=feed_dict)

    return c, a

  def pseudo_test(self, sess, feed_dict):
    assert(self.batch_size == targets.shape[0])
    feed_dict[self.sy_initial_step] = True

    for i in range(self.num_layers):
      feed_dict[self.sy_hidden_states[i]] = np.zeros((self.batch_size, self.hidden_dim))
      feed_dict[self.sy_cell_states[i]] = np.zeros((self.batch_size, self.hidden_dim))

    p = sess.run(self.predictions, feed_dict=feed_dict)

    return p


  def test(self, sess, input_placeholder, feed_dict, max_steps=16):
    """
    sess: Initialized tensorflow session
    feed_dict: dictionary with key and values entered for inputs and initial states already
    """
    for i in range(self.num_layers):
      feed_dict[self.sy_hidden_states[i]] = np.zeros((self.batch_size, self.hidden_dim))
      feed_dict[self.sy_cell_states[i]] = np.zeros((self.batch_size, self.hidden_dim))

    feed_dict[self.sy_initial_step] = True
    output = np.zeros((self.batch_size, max_steps), dtype=int)

    for i in range(max_steps):
      p, rs, l = sess.run([self.predictions, self.rnn_state, self.logits],
                            feed_dict=feed_dict)

      feed_dict[self.sy_initial_step] = False
      if self.num_layers > 1:
        for i in range(self.num_layers):
          feed_dict[self.sy_hidden_states[i]] = rs[i][0]
          feed_dict[self.sy_cell_states[i]] = rs[i][1]
      else:
        feed_dict[self.sy_hidden_states[0]] = rs[0]
        feed_dict[self.sy_cell_states[0]] = rs[1]

      assert(p.shape[1] == 1 and p.shape[0] == self.batch_size)
      feed_dict[input_placeholder] = p
      output[:, i] = p[:, 0]

    return output, l
