import tensorflow as tf

import layer_utils


class NetWorkInput(object):
    def __init__(self):
        self.feed_buff = {}

    def pre_feed(self):
        pass

    def feed(self):
        return self.feed_buff


class CaptionInput(NetWorkInput):
    def __init__(self, word_embedding_init, null_id):
        self.caption_input = tf.placeholder(shape=[None, None], name="caption_input", dtype=tf.int32)
        embedding_init = tf.constant(word_embedding_init, dtype=tf.float32)
        embedding = tf.get_variable("embedding", initializer=embedding_init)
        self.word_embedding = tf.nn.embedding_lookup(embedding, self.caption_input, name="embedding_lookup")
        self.sy_not_null_mask = tf.placeholder(shape=[None, None], name="mask_not_null", dtype=tf.bool)
        self.null_id = null_id

    def get_embedding(self):
        return self.word_embedding

    def get_not_null_numeric_mask(self):
        return tf.cast(self.sy_not_null_mask, tf.float32)

    def get_not_null_count(self):
        return tf.reduce_sum(self.get_not_null_numeric_mask(), axis=1)

    def pre_feed(self, caption_word_ids):
        self.feed_buff = {
            self.caption_input: caption_word_ids,
            self.sy_not_null_mask: caption_word_ids != self.null_id
        }


class ImageInput(NetWorkInput):
    def __init__(self, image_feature_dim, hidden_dim):
        self.hidden_dim = hidden_dim
        self.image_feat_input = tf.placeholder(shape=[None, image_feature_dim], name="image_feat_input",
                                               dtype=tf.float32)
        self.image_projection = layer_utils.affine_transform(self.image_feat_input, hidden_dim, 'image_proj')
        self.feed_buff = {}

    def get_projection(self):
        return self.image_projection

    def get_hidden_dim(self):
        return self.hidden_dim

    def pre_feed(self, image_proj):
        self.feed_buff = {
            self.image_feat_input: image_proj
        }


class MetadataInput(NetWorkInput):
    def __init__(self):
        self.labels = tf.placeholder(shape=[None], name="labels", dtype=tf.int32)

    def get_labels(self):
        return self.labels

    def get_signs(self, expand_dim=True):
        """
            Assume labels are 0, 1 to return -1 and 1 respectively
        """
        signs = tf.cast((self.labels * 2) - 1, tf.float32)
        return tf.expand_dims(signs, axis=1) if expand_dim else signs

    def pre_feed(self, labels):
        self.feed_buff = {
            self.labels: labels
        }


class Lstm(object):
    def __init__(self, hidden_dim, initial_state, lstm_input):
        self.hidden_dim = hidden_dim
        cell = tf.nn.rnn_cell.LSTMCell(hidden_dim)
        self.lstm_outputs, self.lstm_states = tf.nn.dynamic_rnn(cell, lstm_input,
                                                                time_major=False, dtype=tf.float32,
                                                                initial_state=initial_state)

    def get_output(self):
        return self.lstm_outputs


class LstmScalarRewardStrategy(object):
    class RewardConfig(object):

        def __init__(self, reward_scalar_transformer=None, take_difference=True):
            self.reward_scalar_transformer = reward_scalar_transformer
            self.take_difference = take_difference

    def __init__(self,
                 lstm_output,
                 reward_config=None
                 ):
        """
        lstm_output corresponds to [batch_dim, time_dim, hidden_dim]
        :param lstm_output:
        """
        if reward_config is None:
            reward_config = LstmScalarRewardStrategy.RewardConfig()

        self.lstm_output = lstm_output
        tmp_processing = self.lstm_output

        if reward_config.take_difference:
            tmp_processing = layer_utils.difference_over_time(lstm_output, "incremental_change")

        if reward_config.reward_scalar_transformer is None:
            reward_scalar_transformer = lambda x: layer_utils.affine_transform(x, 1, 'hidden_to_reward')

        self.scalar_rewards = tf.squeeze(reward_scalar_transformer(tmp_processing), axis=2)

    def get_rewards(self):
        return self.scalar_rewards


class Discriminator(object):
    def __init__(self,
                 caption_input,
                 image_input,
                 metadata_input,
                 reward_config=None,
                 learning_rate=1e-3
                 ):
        self.caption_input = caption_input
        self.image_input = image_input
        self.metadata_input = metadata_input
        self.reward_config = reward_config

        image_projection = image_input.get_projection()
        caption_embedding = caption_input.get_embedding()
        initial_lstm_state = tf.nn.rnn_cell.LSTMStateTuple(image_projection * 0, image_projection)
        lstm = Lstm(image_input.get_hidden_dim(), initial_lstm_state, caption_embedding)
        reward_strategy = LstmScalarRewardStrategy(lstm.get_output(), reward_config)
        rewards = reward_strategy.get_rewards()

        self.loss, self.signed_rewards, self.mean_reward_per_sentence = self.compute_loss(rewards)
        self.update_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def compute_loss(self, rewards):
        expanded_signs = self.metadata_input.get_signs()
        masked_reward = rewards * self.caption_input.get_not_null_numeric_mask()
        signed_rewards = expanded_signs * masked_reward
        mean_reward_for_each_sentence = tf.reduce_sum(signed_rewards, axis=1) / self.caption_input.get_not_null_count()
        mean_reward = tf.reduce_mean(mean_reward_for_each_sentence)
        loss = mean_reward * - 1
        return loss, signed_rewards, mean_reward_for_each_sentence

    def train(self, sess):
        def get_feed_dict():
            feed_dict = {}
            for n_input in [self.caption_input, self.image_input, self.metadata_input]:
                feed_dict.update(n_input.feed())
            return feed_dict

        _, loss, signed_rewards, mean_reward_per_sentence = sess.run(
            [self.update_op, self.loss, self.signed_rewards, self.mean_reward_per_sentence],
            feed_dict=get_feed_dict())

        return loss, signed_rewards, mean_reward_per_sentence
