import numpy as np
import tensorflow as tf
from cococaption.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from cococaption.pycocoevalcap.cider.cider import Cider
# from cococaption.pycocoevalcap.spice.spice import Spice
# from cococaption.pycocoevalcap.bleu.bleu import Bleu
# from cococaption.pycocoevalcap.rouge.rouge import Rouge
# from cococaption.pycocoevalcap.meteor.meteor import Meteor
from collections import namedtuple
from generator.generator import Generator

GeneratorSpec = namedtuple('GeneratorSpec', ['input_dim', 'hidden_dim', 'output_dim', 'rnn_activation',
                                             'image_feature_dim', 'n_seq_steps', 'embedding_init',
                                             'n_baseline_layers', 'baseline_hidden_dim',
                                             'mle_learning_rate', 'pg_learning_rate',
                                             'baseline_learning_rate', 'batch_size', 'epsilon'])

class GeneratorWrapper(object):

  def __init__(self, generator_spec, discriminator_reward, load_model=False, reward_to_go=False,
               use_discriminator_reward=True):
    self.gen_spec = generator_spec
    self.generator = Generator(generator_spec, "training_generator", load_model)
    self.old_generator = Generator(generator_spec, "old_generator", load_model)
    if load_model:
      self.load()
    else:
      self.setup_ppo()

    self._discriminator_reward = discriminator_reward
    self.reward_to_go = reward_to_go
    self.use_discriminator_reward = use_discriminator_reward
    self.scorer = Cider()

  def save(self, sess, modelname, modeldir="models"):
    saver = tf.train.Saver()
    saver.save(sess, '%s/%s'%(modeldir, modelname))

  def load(self):
    graph = tf.get_default_graph()
    self.update_old_gen = tf.get_collection("update_old_gen")[0]
    self.ppo_loss = graph.get_tensor_by_name("generator/ppo_loss:0")
    self.ppo_update_op = tf.get_collection("ppo_update_op")[0]

  def setup_ppo(self):
    with tf.variable_scope("generator"):
      generator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='training_generator')
      old_generator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='old_generator')

      update_old_gen = []
      for var, var_target in zip(sorted(generator_vars,     key=lambda v: v.name),
                                 sorted(old_generator_vars, key=lambda v: v.name)):
          update_old_gen.append(var_target.assign(var))
      self.update_old_gen = tf.group(*update_old_gen)
      tf.add_to_collection("update_old_gen", self.update_old_gen)

      ratio = tf.exp(tf.negative(self.generator.neglogp) - tf.negative(self.old_generator.neglogp))
      clipped_loss = tf.multiply(tf.clip_by_value(ratio, 1.0 - self.gen_spec.epsilon, 1.0 + self.gen_spec.epsilon), self.generator.ph_adv)
      reg_loss = tf.multiply(ratio, self.generator.ph_adv)
      self.ppo_loss = tf.negative(tf.reduce_mean(tf.minimum(reg_loss, clipped_loss)), name="ppo_loss")

      train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "training_generator")
      self.ppo_update_op = tf.train.AdamOptimizer(self.gen_spec.pg_learning_rate).minimize(self.ppo_loss, var_list=train_vars, name="ppo_update_op")
      tf.add_to_collection("ppo_update_op", self.ppo_update_op)

  def train(self, sess, data, num_iterations, training_type='PG'):
    assert(training_type in ['PG', 'MLE', 'PPO'])
    data.set_mode(training_type).set_batch_size(self.gen_spec.batch_size)
    if training_type == 'PG':
      return self._train_pg(sess, data, num_iterations)
    elif training_type == 'MLE':
      return self._train_mle(sess, data, num_iterations)
    elif training_type == 'PPO':
      return self._train_pg(sess, data, num_iterations, PPO=True)

  def _train_mle(self, sess, data, num_iterations):
    """
    sess: Initialized tensorflow session
    data: instance of Data object
    returns: cross_entropy and accuracy from final mini_batch
    """
    for itr, mini_batch in enumerate(data.training_batches):
      if itr > num_iterations:
        break;

      image_features, caption_input, caption_targets, target_masks = mini_batch

      feed_dict = {
        self.generator.ph_input            : caption_input,
        self.generator.ph_image_feat_input : image_features,
        self.generator.ph_target           : caption_targets,
        self.generator.ph_target_mask      : target_masks,
        self.generator.ph_hidden_state     : np.zeros((self.gen_spec.batch_size, self.gen_spec.hidden_dim)), # Dummy filler
        self.generator.ph_cell_state       : np.zeros((self.gen_spec.batch_size, self.gen_spec.hidden_dim)), # Dummy filler
        self.generator.ph_initial_step     : True
      }

      _, c, a = sess.run([self.generator.mle_update_op, self.generator.cross_entropy,
                          self.generator.accuracy], feed_dict=feed_dict)

      if (itr % 10 == 0):
        print("iter {}, cross-entropy: {}, accuracy: {}".format(itr, c, a))

    return c, a

  def _train_pg(self, sess, data, num_iterations, PPO=False):
    paths_output = self.generate_paths(sess, data)

    ob_in = paths_output["observation_input"]
    ob_img_feat = paths_output["observation_image_features"]
    ob_hid_state = paths_output["observation_hidden_state"]
    ob_cell_state = paths_output["observation_cell_state"]
    actions = paths_output["actions"]
    q_n = paths_output["rewards"]
    probabilities = paths_output["probabilities"]
    img_idxs = paths_output["keys"]
    captions = paths_output["captions"]

    b_n = np.zeros(q_n.shape)
    for i in range(b_n.shape[1]):
      baseline_dict = {
        self.generator.ph_initial_step: i == 0,
        self.generator.ph_image_feat_input: ob_img_feat[:, i, :],
        self.generator.ph_hidden_state: ob_hid_state[:, i, :]
      }
      b_n[:, i] = sess.run(self.generator.baseline_prediction, feed_dict=baseline_dict)

    q_mean, q_std = np.mean(q_n), np.std(q_n)
    adv_n = q_n - (q_mean + q_std * b_n)

    # Advantage normalization
    adv_mean, adv_std = np.mean(adv_n), max(np.std(adv_n), 0.00000001)
    adv_n = (adv_n - adv_mean) / adv_std

    target_q_n = (q_n - q_mean) / q_std
    for i in range(target_q_n.shape[1]):
      baseline_update_dict = {
        self.generator.ph_initial_step: i==0,
        self.generator.ph_image_feat_input: ob_img_feat[:, i, :],
        self.generator.ph_hidden_state: ob_hid_state[:, i, :],
        self.generator.ph_baseline_targets: target_q_n[:, i]
      }
      sess.run([self.generator.baseline_update_op], feed_dict=baseline_update_dict)

    if PPO:
      sess.run(self.update_old_gen)

    for itr in range(num_iterations):
      for i in range(adv_n.shape[1]):
        pg_update_dict = {
          self.generator.ph_adv: adv_n[:, i:i+1],
          self.generator.ph_target: actions[:, i:i+1],
          self.generator.ph_image_feat_input: ob_img_feat[:, i, :],
          self.generator.ph_input: ob_in[:, i:i+1],
          self.generator.ph_hidden_state: ob_hid_state[:, i, :],
          self.generator.ph_cell_state: ob_cell_state[:, i, :],
          self.generator.ph_initial_step: i == 0
        }

        if PPO:
          old_gen_dict = {
            self.old_generator.ph_target: actions[:, i:i+1],
            self.old_generator.ph_image_feat_input: ob_img_feat[:, i, :],
            self.old_generator.ph_input: ob_in[:, i:i+1],
            self.old_generator.ph_hidden_state: ob_hid_state[:, i, :],
            self.old_generator.ph_cell_state: ob_cell_state[:, i, :],
            self.old_generator.ph_initial_step: i == 0
          }
          sess.run(self.ppo_update_op, feed_dict={**pg_update_dict, **old_gen_dict})
        else:
          sess.run(self.generator.pg_update_op, feed_dict=pg_update_dict)


    return captions, np.prod(probabilities, axis=1), img_idxs, q_n

  def generate_paths(self, sess, data, num_batches=1, gamma=1.0):
    paths = []

    for itr, mini_batch in enumerate(data.training_batches):
      if itr >= num_batches:
        break;

      actions = np.zeros((self.gen_spec.batch_size, self.gen_spec.n_seq_steps), dtype=int)
      rewards = np.zeros((self.gen_spec.batch_size, self.gen_spec.n_seq_steps), dtype=float)
      observation_initial_step = np.zeros((self.gen_spec.batch_size, self.gen_spec.n_seq_steps), dtype=int)
      observation_image_features = np.zeros((self.gen_spec.batch_size, self.gen_spec.n_seq_steps, self.gen_spec.image_feature_dim), dtype=float)
      observation_hidden_state = np.zeros((self.gen_spec.batch_size, self.gen_spec.n_seq_steps, self.gen_spec.hidden_dim), dtype=float)
      observation_cell_state = np.zeros((self.gen_spec.batch_size, self.gen_spec.n_seq_steps, self.gen_spec.hidden_dim), dtype=float)
      observation_input = np.zeros((self.gen_spec.batch_size, self.gen_spec.n_seq_steps), dtype=int)

      probabilities = np.zeros((self.gen_spec.batch_size, self.gen_spec.n_seq_steps), dtype=float)

      observation_initial_step[:, 0] = 1

      image_features, caption_input, captions_GT, img_keys = mini_batch

      feed_dict = {
        self.generator.ph_input            : caption_input[:, 0:1],
        self.generator.ph_image_feat_input : image_features,
        self.generator.ph_hidden_state     : np.zeros((self.gen_spec.batch_size, self.gen_spec.hidden_dim)), # Dummy filler
        self.generator.ph_cell_state       : np.zeros((self.gen_spec.batch_size, self.gen_spec.hidden_dim)), # Dummy filler
        self.generator.ph_initial_step     : True
      }

      for i in range(self.gen_spec.n_seq_steps):
        ac, rs, lt = sess.run([self.generator.sampled_ac, self.generator.rnn_state,
                               self.generator.logits], feed_dict=feed_dict)

        exp_lt = np.exp(np.reshape(lt, (self.gen_spec.batch_size * lt.shape[1], self.gen_spec.output_dim)))
        softmax_lt = exp_lt / np.sum(exp_lt, axis=1, keepdims=True)
        probabilities[:, i] = softmax_lt[np.arange(probabilities.shape[0]), ac[:, 0]]

        actions[:, i] = ac[:, 0]
        observation_input[:, i] = feed_dict[self.generator.ph_input][:, 0]
        observation_image_features[:, i, :] = feed_dict[self.generator.ph_image_feat_input]
        observation_hidden_state[:, i, :] = feed_dict[self.generator.ph_hidden_state]
        observation_cell_state[:, i, :] = feed_dict[self.generator.ph_cell_state]

        feed_dict[self.generator.ph_initial_step] = False
        feed_dict[self.generator.ph_hidden_state] = rs[0]
        feed_dict[self.generator.ph_cell_state] = rs[1]
        feed_dict[self.generator.ph_input] = caption_input[:, i:i+1]#ac

      end_mask = ((1 - np.cumsum(actions == data.END_ID, axis=1) + (actions == data.END_ID)) == 1).astype(int)
      assert(np.max(end_mask) == 1)

      path = {
        "keys" : [int(key) for key in img_keys],
        "probabilities" : probabilities,
        "actions" : actions,
        "observation_initial_step" : observation_initial_step,
        "observation_input" : observation_input,
        "observation_image_features" : observation_image_features,
        "observation_hidden_state" : observation_hidden_state,
        "observation_cell_state" : observation_cell_state,
        "end_mask" : end_mask,
        "captions": data.decode(actions)
      }

      if self.use_discriminator_reward:
        reward_func_out = self.discriminator_reward(sess, data, path)
      else:
        reward_func_out = self.generate_cidre_rewards(sess, data, path, img_keys, captions_GT, caption_input)

      rewards[0:reward_func_out.shape[0], 0:reward_func_out.shape[1]] = reward_func_out
      path["rewards"] = rewards

      paths.append(path)

    # Build arrays for observation, action for the policy gradient update by concatenating
    # across paths
    paths_output = {}
    for key in paths[0].keys():
      if len(paths) > 1:
        paths_output[key] = np.concatenate([path[key] for path in paths])
      else:
        paths_output[key] = paths[0][key]

    # reward_to_go, need to flip since cumsum begins from front
    if self.reward_to_go:
      paths_output["rewards"] = np.flip(np.cumsum(np.flip(paths_output["rewards"], 1), 1), 1)

    return paths_output

  def generate_cidre_rewards(self, sess, data, path, keys, captions_GT, caption_input, num_samples=3):
    actions = path["actions"]
    observation_image_features = path["observation_image_features"]
    observation_hidden_state = path["observation_hidden_state"]
    observation_cell_state = path["observation_cell_state"]

    cand_dict = {}

    for i in range(self.gen_spec.n_seq_steps):
      for j in range(num_samples):
        sampled_actions = np.zeros((self.gen_spec.batch_size, self.gen_spec.n_seq_steps), dtype=int)
        sampled_actions[:, 0:i+1] = actions[:, 0:i+1]

        if i + 1 < self.gen_spec.n_seq_steps:
          feed_dict = {
            self.generator.ph_input            : actions[:, i:i+1],
            self.generator.ph_image_feat_input : observation_image_features[:, i+1, :],
            self.generator.ph_hidden_state     : observation_hidden_state[:, i+1, :],
            self.generator.ph_cell_state       : observation_cell_state[:, i+1, :],
            self.generator.ph_initial_step     : False
          }

          for k in range(i + 1, self.gen_spec.n_seq_steps):
            ac, rs = sess.run([self.generator.sampled_ac, self.generator.rnn_state], feed_dict=feed_dict)

            feed_dict[self.generator.ph_hidden_state] = rs[0]
            feed_dict[self.generator.ph_cell_state] = rs[1]
            feed_dict[self.generator.ph_input] = caption_input[:, k:k+1]#ac

            sampled_actions[:, k] = ac[:, 0]

        candidates = data.decode(sampled_actions)
        assert(len(candidates) == len(keys))
        for idx, cap in enumerate(candidates):
          if keys[idx] not in cand_dict:
            cand_dict[keys[idx]] = []
          cand_dict[keys[idx]].append({
            'caption': cap
          })

    # scorer = ciderEval(captions_GT, cand_list, "coco-val-df")
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(captions_GT)
    res = tokenizer.tokenize(cand_dict)
    scores = self.scorer.compute_score(gts, res)
    rewards = np.mean(scores.reshape((self.gen_spec.batch_size, self.gen_spec.n_seq_steps, num_samples)), axis=2)
    return rewards

  def discriminator_reward(self, sess, data, path):
    return self._discriminator_reward(sess, path["keys"], path["captions"], image_idx_from_training=True)[1]

  def test(self, sess, data, num_batches=1):
    """
    sess: Initialized tensorflow session
    data: instance of data object
    returns: predictions, ground truth and logits
    """
    data.set_mode('PG').set_batch_size(self.gen_spec.batch_size)
    predictions = np.zeros((self.gen_spec.batch_size * num_batches, self.gen_spec.n_seq_steps), dtype=int)
    logits = np.zeros((self.gen_spec.batch_size * num_batches, self.gen_spec.n_seq_steps, self.gen_spec.output_dim), dtype=int)
    img_idxs = np.zeros((self.gen_spec.batch_size * num_batches), dtype=int)
    GT = {}

    for itr, mini_batch in enumerate(data.testing_batches):
      if itr == num_batches:
        break

      image_features, caption_input, captions_GT, img_keys = mini_batch
      img_idxs[itr * self.gen_spec.batch_size:(itr + 1) * self.gen_spec.batch_size] = np.array(img_keys)[:]
      GT = {**GT, **captions_GT}

      feed_dict = {
        self.generator.ph_input            : caption_input[:, 0:1],
        self.generator.ph_image_feat_input : image_features,
        self.generator.ph_hidden_state     : np.zeros((self.gen_spec.batch_size, self.gen_spec.hidden_dim)), # Dummy filler
        self.generator.ph_cell_state       : np.zeros((self.gen_spec.batch_size, self.gen_spec.hidden_dim)), # Dummy filler
        self.generator.ph_initial_step     : True
      }


      for i in range(self.gen_spec.n_seq_steps):
        p, rs, l = sess.run([self.generator.predictions, self.generator.rnn_state, self.generator.logits],
                              feed_dict=feed_dict)

        feed_dict[self.generator.ph_initial_step] = False
        feed_dict[self.generator.ph_hidden_state] = rs[0]
        feed_dict[self.generator.ph_cell_state] = rs[1]

        assert(p.shape[1] == 1 and p.shape[0] == self.gen_spec.batch_size)
        assert(l.shape[2] == self.gen_spec.output_dim and l.shape[0] == self.gen_spec.batch_size)
        feed_dict[self.generator.ph_input] = p
        predictions[itr * self.gen_spec.batch_size:(itr + 1) * self.gen_spec.batch_size, i] = p[:, 0]
        logits[itr * self.gen_spec.batch_size:(itr + 1) * self.gen_spec.batch_size, i, :] = l[:, 0, :]


    return predictions, logits, img_idxs, GT

  def set_scorer(self, scorer):
    """
    meteor = Meteor()
    rouge = Rouge()
    bleu = Bleu()
    cider = Cider()
    spice = Spice()
    """
    self.scorer = scorer
    return self
