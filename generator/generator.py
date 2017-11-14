import numpy as np
import tensorflow as tf
from layer_utils import build_mlp

class Generator(object):

  def __init__(self, generator_spec, scope, load_model=False, seed=294):
    self.gen_spec = generator_spec
    self.seed = seed

    if load_model:
      self.load_model(scope)
    else:
      self.build_model(scope)

  def build_model(self, scope):
    with tf.variable_scope(scope):
      rnn_output, self.rnn_state = tf.nn.dynamic_rnn(self.build_cell(), self.build_input(), initial_state=self.build_state())
      self.logits = tf.identity(tf.layers.dense(rnn_output, self.gen_spec.output_dim, self.gen_spec.rnn_activation), name="rnn_logits")
      self.rnn_state = tf.identity(self.rnn_state, name="rnn_state")

      self.build_loss()

  def build_state(self):
    self.ph_initial_step = tf.placeholder(tf.bool, (), "ph_initial_step")

    self.ph_hidden_state = tf.placeholder(tf.float32, (self.gen_spec.batch_size, self.gen_spec.hidden_dim), "ph_hidden_state")
    self.ph_cell_state = tf.placeholder(tf.float32, (self.gen_spec.batch_size, self.gen_spec.hidden_dim), "ph_cell_state")

    self.ph_image_feat_input = tf.placeholder(tf.float32, (self.gen_spec.batch_size, self.gen_spec.image_feature_dim), "ph_image_feat_input")
    self.initial_hidden_state = tf.identity(tf.layers.dense(self.ph_image_feat_input, self.gen_spec.hidden_dim), name='initial_hidden_state')
    self.initial_cell_state = tf.identity(self.initial_hidden_state * 0, name='initial_cell_state')

    return tf.cond(self.ph_initial_step,
                   lambda: tf.nn.rnn_cell.LSTMStateTuple(self.initial_hidden_state, self.initial_cell_state),
                   lambda: tf.nn.rnn_cell.LSTMStateTuple(self.ph_hidden_state, self.ph_cell_state))

  def build_cell(self):
    return tf.contrib.rnn.LSTMCell(self.gen_spec.hidden_dim, forget_bias=0.0)

  def build_input(self):
    self.ph_input = tf.placeholder(tf.int32, (self.gen_spec.batch_size, None), "ph_input")

    if self.gen_spec.embedding_init != None:
      embedding = tf.get_variable("embedding", initializer=self.gen_spec.embedding_init)
    else:
      embedding_init = tf.random_normal_initializer()
      embedding = tf.get_variable("embedding", (self.gen_spec.output_dim, self.gen_spec.input_dim), tf.float32, embedding_init)

    return tf.nn.embedding_lookup(embedding, self.ph_input)

  def build_loss(self):
    self.ph_target = tf.placeholder(tf.int64, (self.gen_spec.batch_size, None), "ph_target")
    self.ph_target_mask = tf.placeholder(tf.float32, (self.gen_spec.batch_size, None), "ph_target_mask")

    self.build_mle_loss()
    self.build_pg_loss()
    self.build_baseline()

  def build_mle_loss(self):
    with tf.variable_scope("MLE"):
      target_one_hot = tf.one_hot(self.ph_target, self.gen_spec.output_dim, dtype=tf.int64)
      raw_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=target_one_hot, logits=self.logits)
      # masked_cross_entropy = raw_cross_entropy * self.ph_target_mask

      self.cross_entropy = tf.reduce_mean(tf.reduce_sum(raw_cross_entropy, axis=1), name="cross_entropy")
      self.mle_update_op = tf.train.AdamOptimizer(self.gen_spec.mle_learning_rate).minimize(self.cross_entropy,
                                                                                            name="update_op")
      tf.add_to_collection("mle_update_op", self.mle_update_op)

      self.predictions = tf.argmax(self.logits, axis = 2, name="predictions")
      correct_pred = tf.equal(self.predictions, self.ph_target)
      num_correct_pred = tf.reduce_sum(tf.cast(correct_pred, tf.float32) * self.ph_target_mask)
      self.accuracy = tf.divide(num_correct_pred, tf.reduce_sum(self.ph_target_mask), name="accuracy")

  def build_pg_loss(self):
    with tf.variable_scope("PG"):
      multinomial_logits = tf.reshape(self.logits, (tf.shape(self.logits)[0], self.gen_spec.output_dim))
      self.sampled_ac = tf.reshape(tf.multinomial(multinomial_logits, 1, seed=self.seed),
                                   (tf.shape(self.logits)[0], 1), name="sampled_ac")
      self.neglogp = tf.identity(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.ph_target,
                                                                                logits=self.logits), name="neglogp")

      # Get average cross entropy over caption
      self.ph_adv = tf.placeholder(tf.float32, (self.gen_spec.batch_size, None), "ph_adv")
      self.loss = tf.reduce_mean(tf.multiply(self.neglogp, self.ph_adv), name="loss")
      self.pg_update_op = tf.train.AdamOptimizer(self.gen_spec.pg_learning_rate).minimize(self.loss, name="update_op")
      tf.add_to_collection("pg_update_op", self.pg_update_op)

  def build_baseline(self):
    with tf.variable_scope("baseline"):
      baseline_input = tf.cond(self.ph_initial_step,
                               lambda: self.initial_hidden_state,
                               lambda: self.ph_hidden_state)
      self.baseline_prediction = tf.identity(tf.squeeze(build_mlp(baseline_input, 1, "baseline",
                                                                  self.gen_spec.n_baseline_layers,
                                                                  self.gen_spec.baseline_hidden_dim)),
                                             name="baseline_prediction")
      self.ph_baseline_targets = tf.placeholder(tf.float32, (self.gen_spec.batch_size), "ph_baseline_targets")

      baseline_loss = tf.nn.l2_loss(self.ph_baseline_targets - self.baseline_prediction)
      self.baseline_update_op = tf.train.AdamOptimizer(self.gen_spec.baseline_learning_rate).minimize(baseline_loss, name="update_op")
      tf.add_to_collection("baseline_update_op", self.baseline_update_op)

  def load_model(self, scope):
    graph = tf.get_default_graph()

    # Load main model
    self.logits = graph.get_tensor_by_name("%s/rnn_logits:0" % scope)
    self.rnn_state = graph.get_tensor_by_name("%s/rnn_state:0" % scope)

    # Load state
    self.ph_initial_step = graph.get_tensor_by_name("%s/ph_initial_step:0" % scope)
    self.ph_hidden_state = graph.get_tensor_by_name("%s/ph_hidden_state:0" % scope)
    self.ph_cell_state = graph.get_tensor_by_name("%s/ph_cell_state:0" % scope)
    self.ph_image_feat_input = graph.get_tensor_by_name("%s/ph_image_feat_input:0" % scope)
    self.initial_hidden_state = graph.get_tensor_by_name("%s/initial_hidden_state:0" % scope)
    self.initial_cell_state = graph.get_tensor_by_name("%s/initial_cell_state:0" % scope)

    # Load input
    self.ph_input = graph.get_tensor_by_name("%s/ph_input:0" % scope)

    # Load loss
    self.ph_target = graph.get_tensor_by_name("%s/ph_target:0" % scope)
    self.ph_target_mask = graph.get_tensor_by_name("%s/ph_target_mask:0" % scope)

    # Load MLE loss
    self.cross_entropy = graph.get_tensor_by_name("%s/MLE/cross_entropy:0" % scope)
    self.mle_update_op = tf.get_collection("mle_update_op")[0]
    self.predictions = graph.get_tensor_by_name("%s/MLE/predictions:0" % scope)
    self.accuracy = graph.get_tensor_by_name("%s/MLE/accuracy:0" % scope)

    # Load PG loss
    self.sampled_ac = graph.get_tensor_by_name("%s/PG/sampled_ac:0" % scope)
    self.neglogp = graph.get_tensor_by_name("%s/PG/neglogp:0" % scope)
    self.ph_adv = graph.get_tensor_by_name("%s/PG/ph_adv:0" % scope)
    self.loss = graph.get_tensor_by_name("%s/PG/loss:0" % scope)
    self.pg_update_op = tf.get_collection("pg_update_op")[0]

    # Load baseline
    self.baseline_prediction = graph.get_tensor_by_name("%s/baseline/baseline_prediction:0" % scope)
    self.ph_baseline_targets = graph.get_tensor_by_name("%s/baseline/ph_baseline_targets:0" % scope)
    self.baseline_update_op = tf.get_collection("baseline_update_op")[0]
