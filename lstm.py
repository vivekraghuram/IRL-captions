import numpy as np
import tensorflow as tf
from layer_utils import build_mlp
from pyciderevalcap.eval import CIDErEvalCap as ciderEval

class GenericLSTM(object):

  def __init__(self,
               hidden_dim=512,
               output_dim=1004,
               input_dim=512,
               learning_rate=5e-5,
               batch_size=50):

    self.hidden_dim = hidden_dim
    self.output_dim = output_dim # vocab_dim
    self.input_dim = input_dim
    self.learning_rate = learning_rate
    self.batch_size = batch_size

  def build_model(self, activation=None):
    initial_state = self.build_state()
    cell = self.build_cell()
    input_layer = self.build_input()

    rnn_output, self.rnn_state = tf.nn.dynamic_rnn(cell, input_layer, initial_state=initial_state)
    self.logits = tf.layers.dense(inputs=rnn_output, units=self.output_dim, activation=activation, name="rnn_logits")
    self.rnn_state = tf.identity(self.rnn_state, name="rnn_state")
    tf.add_to_collection("rnn_logits", self.logits)

    self.build_loss()

  def build_state(self):
    self.sy_initial_step = tf.placeholder(shape=[], name="sy_initial_step", dtype=tf.bool)
    self.sy_hidden_state = tf.placeholder(shape=[self.batch_size, self.hidden_dim], name="sy_hidden_state", dtype=tf.float32)
    self.sy_cell_state = tf.placeholder(shape=[self.batch_size, self.hidden_dim], name="sy_cell_state", dtype=tf.float32)

    return tf.nn.rnn_cell.LSTMStateTuple(self.sy_hidden_state, self.sy_cell_state)

  def build_input(self):
    self.sy_input = f.placeholder(shape=[self.batch_size, None, self.input_dim], name="sy_input", dtype=tf.float32)
    return self.sy_input

  def build_cell(self):
    return tf.contrib.rnn.LSTMBlockCell(self.hidden_dim, forget_bias=0.0)

  def build_loss(self):
    raise NotImplementedError

  def save_model(self, sess, modelname):
    saver = tf.train.Saver()
    saver.save(sess, 'models/%s'%(modelname))

  def load_model(self, sess, modelname):
    loader = tf.train.import_meta_graph('%s.meta'%modelname)
    loader.restore(sess, modelname)

    graph = tf.get_default_graph()
    self.sy_input = graph.get_tensor_by_name("sy_input:0")

    self.sy_hidden_state = graph.get_tensor_by_name("sy_hidden_state:0")
    self.sy_cell_state = graph.get_tensor_by_name("sy_cell_state:0")
    self.sy_initial_step = graph.get_tensor_by_name("sy_initial_step:0")

    self.rnn_state = graph.get_tensor_by_name("rnn_state:0")
    self.logits = tf.get_collection("rnn_logits")[0]


  def train(self, sess, data):
    raise NotImplementedError

  def test(self, sess, data):
    raise NotImplementedError

class MaxLikelihoodLSTM(GenericLSTM):

  def __init__(self,
               embedding_init=None,
               hidden_dim=512,
               output_dim=1004,
               input_dim=512,
               learning_rate=5e-5,
               batch_size=50,
               image_feature_dim=4096):

    super().__init__(hidden_dim, output_dim, input_dim, learning_rate, batch_size)
    self.embedding_init = embedding_init
    self.image_feature_dim = image_feature_dim

  def build_state(self):
    recurrent_state = GenericLSTM.build_state(self)

    self.sy_image_feat_input = tf.placeholder(shape=[self.batch_size, self.image_feature_dim], name="sy_image_feat_input", dtype=tf.float32)
    self.initial_hidden_state = tf.identity(tf.layers.dense(inputs=self.sy_image_feat_input, units=self.hidden_dim), name='initial_hidden_state')
    self.initial_cell_state = tf.identity(self.initial_hidden_state * 0, name='initial_cell_state')
    # tf.add_to_collection("initial_hidden_state", self.initial_hidden_state)

    return tf.cond(self.sy_initial_step,
                   lambda: tf.nn.rnn_cell.LSTMStateTuple(self.initial_hidden_state, self.initial_cell_state),
                   lambda: recurrent_state)

  def build_input(self):
    self.sy_input = tf.placeholder(shape=[self.batch_size, None], name="sy_input", dtype=tf.int32)

    if self.embedding_init != None:
      embedding = tf.get_variable("embedding", initializer=self.embedding_init)
    else:
      embedding_init = tf.random_normal_initializer()
      embedding = tf.get_variable("embedding", [self.output_dim, self.input_dim], dtype=tf.float32, initializer=embedding_init)

    word_embedding = tf.nn.embedding_lookup(embedding, self.sy_input)
    return word_embedding

  def build_loss(self):
    self.sy_target = tf.placeholder(shape=[self.batch_size, None], name="sy_target", dtype=tf.int64)
    self.sy_target_mask = tf.placeholder(shape=[self.batch_size, None], name="sy_target_mask", dtype=tf.float32)

    # Apply not-null mask to ignore cross-entropy on trailing padding
    target_one_hot = tf.one_hot(self.sy_target, self.output_dim, dtype=tf.int64)
    raw_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=target_one_hot, logits=self.logits)
    masked_cross_entropy = raw_cross_entropy * self.sy_target_mask

    # Get average cross entropy over caption
    self.cross_entropy = tf.reduce_mean(tf.reduce_sum(raw_cross_entropy, axis=1), name="cross_entropy")
    self.update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy, name="update_op")
    tf.add_to_collection("update_op", self.update_op)

    self.predictions = tf.argmax(self.logits, axis = 2, name="predictions")
    correct_pred = tf.equal(self.predictions, self.sy_target)
    num_correct_pred = tf.reduce_sum(tf.cast(correct_pred, tf.float32) * self.sy_target_mask)
    self.accuracy = tf.divide(num_correct_pred, tf.reduce_sum(self.sy_target_mask), name="accuracy")

  def load_model(self, sess, modelname):
    GenericLSTM.load_model(self, sess, modelname)

    graph = tf.get_default_graph()
    self.sy_image_feat_input = graph.get_tensor_by_name("sy_image_feat_input:0")

    self.sy_target = graph.get_tensor_by_name("sy_target:0")
    self.sy_target_mask = graph.get_tensor_by_name("sy_target_mask:0")

    # self.initial_hidden_state = tf.get_collection("image_proj")[0]
    self.initial_hidden_state = graph.get_tensor_by_name("initial_hidden_state:0")
    self.initial_cell_state = graph.get_tensor_by_name("initial_cell_state:0")

    self.cross_entropy = graph.get_tensor_by_name("cross_entropy:0")
    self.update_op = tf.get_collection("update_op")[0]

    self.predictions = graph.get_tensor_by_name("predictions:0")
    self.accuracy = graph.get_tensor_by_name("accuracy:0")

  def train(self, sess, data, max_iterations=200):
    """
    sess: Initialized tensorflow session
    data: instance of data object
    returns: cross_entropy and accuracy from final mini_batch
    """
    i = 0
    for mini_batch in data.training_batches(self.batch_size):
      image_features, caption_input, caption_targets, target_masks = mini_batch

      feed_dict = {
        self.sy_input            : caption_input,
        self.sy_image_feat_input : image_features,
        self.sy_target           : caption_targets,
        self.sy_target_mask      : target_masks,
        self.sy_hidden_state     : np.zeros((self.batch_size, self.hidden_dim)), # Dummy filler
        self.sy_cell_state       : np.zeros((self.batch_size, self.hidden_dim)), # Dummy filler
        self.sy_initial_step     : True
      }

      _, c, a = sess.run([self.update_op, self.cross_entropy, self.accuracy], feed_dict=feed_dict)

      if (i % 10 == 0):
        print("iter {}, cross-entropy: {}, accuracy: {}".format(i, c, a))

      i += 1
      if i > max_iterations:
        break;

    return c, a

  def test(self, sess, data, num_batches=1, max_steps=16):
    """
    sess: Initialized tensorflow session
    data: instance of data object
    returns: predictions, ground truth and logits
    """
    predictions = np.zeros((self.batch_size * num_batches, max_steps), dtype=int)
    GT     = np.zeros((self.batch_size * num_batches, max_steps), dtype=int)
    logits = np.zeros((self.batch_size * num_batches, max_steps, self.output_dim), dtype=int)
    batch_count = 0

    for mini_batch in data.testing_batches(self.batch_size):
      image_features, caption_input, caption_GT, GT_masks = mini_batch
      GT[batch_count * self.batch_size:(batch_count + 1) * self.batch_size, :] = caption_GT

      feed_dict = {
        self.sy_input            : caption_input,
        self.sy_image_feat_input : image_features,
        self.sy_hidden_state     : np.zeros((self.batch_size, self.hidden_dim)), # Dummy filler
        self.sy_cell_state       : np.zeros((self.batch_size, self.hidden_dim)), # Dummy filler
        self.sy_initial_step     : True
      }


      for i in range(max_steps):
        p, rs, l = sess.run([self.predictions, self.rnn_state, self.logits],
                              feed_dict=feed_dict)

        feed_dict[self.sy_initial_step] = False
        feed_dict[self.sy_hidden_state] = rs[0]
        feed_dict[self.sy_cell_state] = rs[1]

        assert(p.shape[1] == 1 and p.shape[0] == self.batch_size)
        assert(l.shape[2] == self.output_dim and l.shape[0] == self.batch_size)
        feed_dict[self.sy_input] = p
        predictions[batch_count * self.batch_size:(batch_count + 1) * self.batch_size, i] = p[:, 0]
        logits[batch_count * self.batch_size:(batch_count + 1) * self.batch_size, i, :] = l[:, 0, :]

      batch_count += 1
      if batch_count == num_batches:
        break

    return predictions, GT, logits

class PolicyGradientLSTM(MaxLikelihoodLSTM):

  def __init__(self,
               embedding_init=None,
               hidden_dim=512,
               output_dim=1004,
               input_dim=512,
               learning_rate=5e-4,
               batch_size=50,
               image_feature_dim=4096,
               n_layers=1,
               baseline_size=32):

    super().__init__(embedding_init, hidden_dim, output_dim, input_dim, learning_rate, batch_size, image_feature_dim)
    self.n_layers = n_layers
    self.baseline_size = baseline_size
    self.seed = 294

  def build_model(self, activation=None, loaded_mle=False):
    if not loaded_mle:
      self.sy_target = tf.placeholder(shape=[self.batch_size, None], name="sy_target", dtype=tf.int64)
      super().build_model(activation)
    else:
      self.build_loss()

    self.build_baseline()

  def build_loss(self):
    multinomial_logits = tf.reshape(self.logits, (self.batch_size * tf.shape(self.logits)[1], self.output_dim))
    self.sampled_ac = tf.reshape(tf.multinomial(multinomial_logits, 1, seed=self.seed),
                                 [self.batch_size, tf.shape(self.logits)[1]], name="sampled_ac")
    self.logprob = tf.identity(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.sy_target, logits=self.logits), name="logprob")

    # Get average cross entropy over caption
    self.sy_adv = tf.placeholder(shape=[self.batch_size, None], name="sy_adv", dtype=tf.float32)
    self.loss = tf.reduce_mean(tf.multiply(self.logprob, self.sy_adv), name="loss")
    self.update_op = tf.train.AdamOptimizer(self.learning_rate, name='adam_pg').minimize(self.loss, name="pg_update_op")
    tf.add_to_collection("pg_update_op", self.update_op)

  def build_baseline(self):
    baseline_input = tf.cond(self.sy_initial_step,
                             lambda: self.initial_hidden_state,
                             lambda: self.sy_hidden_state)
    self.baseline_prediction = tf.squeeze(build_mlp(baseline_input, 1, "baseline", n_layers=self.n_layers, size=self.baseline_size))
    self.baseline_prediction = tf.identity(self.baseline_prediction, name="baseline_prediction")
    self.baseline_targets = tf.placeholder(shape=[self.batch_size], name="baseline_targets", dtype=tf.float32)

    baseline_loss = tf.nn.l2_loss(self.baseline_targets - self.baseline_prediction)
    self.baseline_update_op = tf.train.AdamOptimizer(self.learning_rate, name='adam_bl').minimize(baseline_loss, name="baseline_update_op")
    tf.add_to_collection("baseline_update_op", self.baseline_update_op)

  def generate_paths(self, sess, data, max_steps=16, num_paths=1000, num_samples=20, gamma=1.0):
    paths = []
    num_paths = self.batch_size

    for count, mini_batch in enumerate(data.training_batches(self.batch_size)):
      actions = np.zeros((self.batch_size, max_steps), dtype=int)
      rewards = np.zeros((self.batch_size, max_steps), dtype=float)
      observation_initial_step = np.zeros((self.batch_size, max_steps), dtype=int)
      observation_image_features = np.zeros((self.batch_size, max_steps, self.image_feature_dim), dtype=float)
      observation_hidden_state = np.zeros((self.batch_size, max_steps, self.hidden_dim), dtype=float)
      observation_cell_state = np.zeros((self.batch_size, max_steps, self.hidden_dim), dtype=float)
      observation_input = np.zeros((self.batch_size, max_steps), dtype=int) # output_dim is vocab_dim

      image_features, caption_input, captions_GT, keys = mini_batch
      cand_list = []

      observation_initial_step[:, 0] = 1

      feed_dict = {
        self.sy_input            : caption_input,
        self.sy_image_feat_input : image_features,
        self.sy_hidden_state     : np.zeros((self.batch_size, self.hidden_dim)), # Dummy filler
        self.sy_cell_state       : np.zeros((self.batch_size, self.hidden_dim)), # Dummy filler
        self.sy_initial_step     : True
      }

      for i in range(max_steps):
        ac, rs = sess.run([self.sampled_ac, self.rnn_state], feed_dict=feed_dict)

        actions[:, i] = ac[:, 0]
        observation_input[:, i] = feed_dict[self.sy_input][:, 0]
        observation_image_features[:, i, :] = feed_dict[self.sy_image_feat_input]
        observation_hidden_state[:, i, :] = feed_dict[self.sy_hidden_state]
        observation_cell_state[:, i, :] = feed_dict[self.sy_cell_state]

        for j in range(num_samples):
          actions[:, i+1:max_steps] = np.random.randint(0, self.output_dim, size=(self.batch_size, max_steps - i - 1))

          candidates = data.decode(actions)
          assert(len(candidates) == len(keys))
          for idx, cap in enumerate(candidates):
            cand_list.append({
              'image_id': keys[idx],
              'caption': cap
            })

        feed_dict[self.sy_initial_step] = False
        feed_dict[self.sy_hidden_state] = rs[0]
        feed_dict[self.sy_cell_state] = rs[1]
        feed_dict[self.sy_input] = ac

      # print(cand_list[-1])
      # print(captions_GT[cand_list[-1]['image_id']])
      scorer = ciderEval(captions_GT, cand_list, "coco-val-df")
      scores = scorer.evaluate()
      reshaped_scores = np.reshape(scores, (max_steps, num_samples, self.batch_size))
      rewards = np.max(np.swapaxes(np.swapaxes(reshaped_scores, 1, 2), 0, 1), axis=2) # Should this be max instead of mean?
      end_mask = 1 - np.cumsum(actions == data.END_ID, axis=1)

      path = {
        "actions" : actions,
        "rewards" : rewards,
        "observation_initial_step" : observation_initial_step,
        "observation_input" : observation_input,
        "observation_image_features" : observation_image_features,
        "observation_hidden_state" : observation_hidden_state,
        "observation_cell_state" : observation_cell_state,
        "end_mask" : end_mask
      }

      paths.append(path)
      if (count + 1) * self.batch_size >= num_paths:
        break

    # Build arrays for observation, action for the policy gradient update by concatenating
    # across paths
    if len(paths) > 1:
      ob_initial_step = np.concatenate([path["observation_initial_step"] for path in paths])
      ob_input = np.concatenate([path["observation_input"] for path in paths])
      ob_image_features = np.concatenate([path["observation_image_features"] for path in paths])
      ob_hidden_state = np.concatenate([path["observation_hidden_state"] for path in paths])
      ob_cell_state = np.concatenate([path["observation_cell_state"] for path in paths])
      actions = np.concatenate([path["actions"] for path in paths])
      rewards = np.concatenate([path["rewards"] for path in paths])
      end_masks = np.concatenate([path["end_mask"] for path in paths])
    else:
      ob_initial_step = paths[0]["observation_initial_step"]
      ob_input = paths[0]["observation_input"]
      ob_image_features = paths[0]["observation_image_features"]
      ob_hidden_state = paths[0]["observation_hidden_state"]
      ob_cell_state = paths[0]["observation_cell_state"]
      actions = paths[0]["actions"]
      rewards = paths[0]["rewards"]
      end_masks = paths[0]["end_mask"]

    reward_to_go = rewards#np.flip(np.cumsum(np.flip(rewards, 1), 1), 1) # need to flip since cumsum begins from front

    return ob_initial_step, ob_input, ob_image_features, ob_hidden_state, ob_cell_state, actions, reward_to_go, end_masks

  def generate_smart_paths(self, sess, data, max_steps=16, num_paths=1000, num_samples=10, gamma=1.0):
    paths = []
    num_paths = self.batch_size

    for count, mini_batch in enumerate(data.training_batches(self.batch_size)):
      actions = np.zeros((self.batch_size, max_steps), dtype=int)
      rewards = np.zeros((self.batch_size, max_steps), dtype=float)
      observation_initial_step = np.zeros((self.batch_size, max_steps), dtype=int)
      observation_image_features = np.zeros((self.batch_size, max_steps, self.image_feature_dim), dtype=float)
      observation_hidden_state = np.zeros((self.batch_size, max_steps, self.hidden_dim), dtype=float)
      observation_cell_state = np.zeros((self.batch_size, max_steps, self.hidden_dim), dtype=float)
      observation_input = np.zeros((self.batch_size, max_steps), dtype=int) # output_dim is vocab_dim

      image_features, caption_input, captions_GT, keys = mini_batch
      cand_list = []

      observation_initial_step[:, 0] = 1

      feed_dict = {
        self.sy_input            : caption_input,
        self.sy_image_feat_input : image_features,
        self.sy_hidden_state     : np.zeros((self.batch_size, self.hidden_dim)), # Dummy filler
        self.sy_cell_state       : np.zeros((self.batch_size, self.hidden_dim)), # Dummy filler
        self.sy_initial_step     : True
      }

      for i in range(max_steps):
        ac, rs = sess.run([self.sampled_ac, self.rnn_state], feed_dict=feed_dict)

        actions[:, i] = ac[:, 0]
        observation_input[:, i] = feed_dict[self.sy_input][:, 0]
        observation_image_features[:, i, :] = feed_dict[self.sy_image_feat_input]
        observation_hidden_state[:, i, :] = feed_dict[self.sy_hidden_state]
        observation_cell_state[:, i, :] = feed_dict[self.sy_cell_state]

        feed_dict[self.sy_initial_step] = False
        feed_dict[self.sy_hidden_state] = rs[0]
        feed_dict[self.sy_cell_state] = rs[1]
        feed_dict[self.sy_input] = ac

      for i in range(max_steps):
        for j in range(num_samples):
          sampled_actions = np.zeros((self.batch_size, max_steps), dtype=int)
          sampled_actions[:, 0:i+1] = actions[:, 0:i+1]

          if i + 1 < max_steps:
            feed_dict[self.sy_initial_step] = False
            feed_dict[self.sy_input] = actions[:, i:i+1]
            feed_dict[self.sy_hidden_state] = observation_hidden_state[:, i+1, :]
            feed_dict[self.sy_cell_state] = observation_cell_state[:, i+1, :]
            feed_dict[self.sy_image_feat_input] = observation_image_features[:, i+1, :]

            for k in range(i + 1, max_steps):
              ac, rs = sess.run([self.sampled_ac, self.rnn_state], feed_dict=feed_dict)

              feed_dict[self.sy_hidden_state] = rs[0]
              feed_dict[self.sy_cell_state] = rs[1]
              feed_dict[self.sy_input] = ac

              sampled_actions[:, k] = ac[:, 0]

          candidates = data.decode(sampled_actions)
          assert(len(candidates) == len(keys))
          for idx, cap in enumerate(candidates):
            cand_list.append({
              'image_id': keys[idx],
              'caption': cap
            })

      print(cand_list[-1])
      print(captions_GT[cand_list[-1]['image_id']])
      scorer = ciderEval(captions_GT, cand_list, "coco-val-df")
      scores = scorer.evaluate()
      reshaped_scores = np.reshape(scores, (max_steps, num_samples, self.batch_size))
      rewards = np.mean(np.swapaxes(np.swapaxes(reshaped_scores, 1, 2), 0, 1), axis=2) # Should this be max instead of mean?
      end_mask = 1 - np.cumsum(actions == data.END_ID, axis=1)

      path = {
        "actions" : actions,
        "rewards" : rewards,
        "observation_initial_step" : observation_initial_step,
        "observation_input" : observation_input,
        "observation_image_features" : observation_image_features,
        "observation_hidden_state" : observation_hidden_state,
        "observation_cell_state" : observation_cell_state,
        "end_mask" : end_mask
      }

      paths.append(path)
      if (count + 1) * self.batch_size >= num_paths:
        break

    # Build arrays for observation, action for the policy gradient update by concatenating
    # across paths
    if len(paths) > 1:
      ob_initial_step = np.concatenate([path["observation_initial_step"] for path in paths])
      ob_input = np.concatenate([path["observation_input"] for path in paths])
      ob_image_features = np.concatenate([path["observation_image_features"] for path in paths])
      ob_hidden_state = np.concatenate([path["observation_hidden_state"] for path in paths])
      ob_cell_state = np.concatenate([path["observation_cell_state"] for path in paths])
      actions = np.concatenate([path["actions"] for path in paths])
      rewards = np.concatenate([path["rewards"] for path in paths])
      end_masks = np.concatenate([path["end_mask"] for path in paths])
    else:
      ob_initial_step = paths[0]["observation_initial_step"]
      ob_input = paths[0]["observation_input"]
      ob_image_features = paths[0]["observation_image_features"]
      ob_hidden_state = paths[0]["observation_hidden_state"]
      ob_cell_state = paths[0]["observation_cell_state"]
      actions = paths[0]["actions"]
      rewards = paths[0]["rewards"]
      end_masks = paths[0]["end_mask"]

    reward_to_go = rewards#np.flip(np.cumsum(np.flip(rewards, 1), 1), 1) # need to flip since cumsum begins from front

    return ob_initial_step, ob_input, ob_image_features, ob_hidden_state, ob_cell_state, actions, reward_to_go, end_masks

  def train(self, sess, data):
    ob_init_step, ob_in, ob_img_feat, ob_hid_state, ob_cell_state, actions, q_n, end_masks = self.generate_smart_paths(sess, data)

    b_n = np.zeros(q_n.shape)
    for i in range(b_n.shape[1]):
      b_n[:, i] = sess.run(self.baseline_prediction, feed_dict={self.sy_initial_step: i == 0,
                                                         self.sy_image_feat_input: ob_img_feat[:, i, :],
                                                         self.sy_hidden_state: ob_hid_state[:, i, :]})
    q_mean, q_std = np.mean(q_n), np.std(q_n)
    adv_n = q_n - (q_mean + q_std * b_n)

    target_q_n = (adv_n - q_mean) / q_std
    for i in range(target_q_n.shape[1]):
      sess.run([self.baseline_update_op], feed_dict={self.sy_initial_step: i==0,
                                                self.sy_image_feat_input: ob_img_feat[:, i, :],
                                                self.sy_hidden_state: ob_hid_state[:, i, :],
                                                self.baseline_targets: target_q_n[:, i]})

    losses = []
    for i in range(adv_n.shape[1]):
      loss_val, _, = sess.run([self.loss, self.update_op], feed_dict={self.sy_adv: adv_n[:, i:i+1],
                                                                      self.sy_target: actions[:, i:i+1],
                                                                      self.sy_image_feat_input: ob_img_feat[:, i, :],
                                                                      self.sy_input: ob_in[:, i:i+1],
                                                                      self.sy_hidden_state: ob_hid_state[:, i, :],
                                                                      self.sy_cell_state: ob_cell_state[:, i, :],
                                                                      self.sy_initial_step: i == 0
                                                                      })
      losses.append(loss_val)

    return losses, actions, q_n, adv_n

  def load_model(self, sess, modelname, is_PG=False):
    MaxLikelihoodLSTM.load_model(self, sess, modelname)

    if is_PG:
      graph = tf.get_default_graph()
      self.sampled_ac = graph.get_tensor_by_name("sampled_ac:0")
      self.logprob = graph.get_tensor_by_name("logprob:0")
      self.sy_adv = graph.get_tensor_by_name("sy_adv:0")
      self.loss = graph.get_tensor_by_name("loss:0")
      self.update_op = tf.get_collection("pg_update_op")[0]

      self.baseline_prediction = graph.get_tensor_by_name("baseline_prediction:0")
      self.baseline_targets = graph.get_tensor_by_name("baseline_targets:0")
      self.baseline_update_op = tf.get_collection("baseline_update_op")[0]
    else:
      self.build_model(loaded_mle=True)
