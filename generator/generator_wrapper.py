import numpy as np
import tensorflow as tf
from pyciderevalcap.eval import CIDErEvalCap as ciderEval
from collections import namedtuple
from generator.generator import Generator

GeneratorSpec = namedtuple('GeneratorSpec', ['input_dim', 'hidden_dim', 'output_dim', 'rnn_activation',
                                             'image_feature_dim', 'n_seq_steps', 'embedding_init',
                                             'n_baseline_layers', 'baseline_hidden_dim',
                                             'mle_learning_rate', 'pg_learning_rate',
                                             'baseline_learning_rate', 'batch_size'])

class GeneratorWrapper(object):

  def __init__(self, generator_spec, discriminator_reward, load_model=False, reward_to_go=False,
               use_discriminator_reward=True):
    self.gen_spec = generator_spec
    self.generator = Generator(generator_spec, "training_generator", load_model)
    self.old_generator = Generator(generator_spec, "old_generator", load_model)

    self._discriminator_reward = discriminator_reward
    self.reward_to_go = reward_to_go
    self.use_discriminator_reward = use_discriminator_reward

  def save(self, sess, modelname, modeldir="models"):
    saver = tf.train.Saver()
    saver.save(sess, '%s/%s'%(modeldir, modelname))

  def train(self, sess, data, num_iterations=300, training_type='PG'):
    assert(training_type in ['PG', 'MLE', 'PPO'])
    data.set_mode(training_type).set_batch_size(self.gen_spec.batch_size)
    if training_type == 'PG':
      return self._train_pg(sess, data)
    elif training_type == 'MLE':
      return self._train_mle(sess, data, num_iterations)
    elif training_type == 'PPO':
     raise NotImplementedError

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

  def _train_pg(self, sess, data):
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

    target_q_n = (adv_n - q_mean) / q_std
    for i in range(target_q_n.shape[1]):
      baseline_update_dict = {
        self.generator.ph_initial_step: i==0,
        self.generator.ph_image_feat_input: ob_img_feat[:, i, :],
        self.generator.ph_hidden_state: ob_hid_state[:, i, :],
        self.generator.ph_baseline_targets: target_q_n[:, i]
      }
      sess.run([self.generator.baseline_update_op], feed_dict=baseline_update_dict)

    losses = []
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
      loss_val, _, = sess.run([self.generator.loss, self.generator.pg_update_op], feed_dict=pg_update_dict)
      losses.append(loss_val)


    return captions, np.prod(probabilities, axis=1), img_idxs, q_n

  def generate_paths(self, sess, data, num_iterations=0, gamma=1.0):
    paths = []

    for itr, mini_batch in enumerate(data.training_batches):
      if itr > num_iterations:
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
        self.generator.ph_input            : caption_input,
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
        feed_dict[self.generator.ph_input] = ac

      end_mask = 1 - np.cumsum(actions == data.END_ID, axis=1)

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
        reward_func_out = self.generate_cidre_rewards(sess, data, path, img_keys, captions_GT)

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

  def generate_cidre_rewards(self, sess, data, path, keys, captions_GT, num_samples=10):
    actions = path["actions"]
    observation_image_features = path["observation_image_features"]
    observation_hidden_state = path["observation_hidden_state"]
    observation_cell_state = path["observation_cell_state"]

    cand_list = []

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
            feed_dict[self.generator.ph_input] = ac

            sampled_actions[:, k] = ac[:, 0]

        candidates = data.decode(sampled_actions)
        assert(len(candidates) == len(keys))
        for idx, cap in enumerate(candidates):
          cand_list.append({
            'image_id': keys[idx],
            'caption': cap
          })

    scorer = ciderEval(captions_GT, cand_list, "coco-val-df")
    scores = scorer.evaluate()
    reshaped_scores = np.reshape(scores, (self.gen_spec.n_seq_steps, num_samples, self.gen_spec.batch_size))
    rewards = np.mean(np.swapaxes(np.swapaxes(reshaped_scores, 1, 2), 0, 1), axis=2)
    return rewards

  def discriminator_reward(self, sess, data, path):
    return self._discriminator_reward(sess, path["keys"], path["captions"], image_idx_from_training=True)[1]
