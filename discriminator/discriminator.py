import tensorflow as tf

import layer_utils


def get_tensor_by_name(graph, name, index=0):
    return graph.get_tensor_by_name("{}:{}".format(name, index))


class NetWorkInput(object):
    def __init__(self):
        self.feed_buff = {}

    def pre_feed(self):
        pass

    def feed(self):
        to_feed = self.feed_buff
        self.feed_buff = {}
        return to_feed


class CaptionInput(NetWorkInput):
    def __init__(self, word_embedding_init, null_id, graph=None):

        caption_input_tname = "caption_input"
        embedding_tname = "embedding"
        lookup_tname = "embedding_lookup"
        mask_not_null_tname = "mask_not_null"
        if graph is not None:
            self.caption_input = get_tensor_by_name(graph, caption_input_tname)
            self.word_embedding = get_tensor_by_name(graph, lookup_tname)
            self.embedding = get_tensor_by_name(graph, embedding_tname)
            self.sy_not_null_mask = get_tensor_by_name(graph, mask_not_null_tname)
            self.null_id = null_id
        else:
            self.caption_input = tf.placeholder(shape=[None, None], name=caption_input_tname, dtype=tf.int32)
            embedding_init = tf.constant(word_embedding_init, dtype=tf.float32)
            self.embedding = tf.get_variable(name=embedding_tname, initializer=embedding_init)
            self.word_embedding = tf.nn.embedding_lookup(self.embedding, self.caption_input, name=lookup_tname)
            self.sy_not_null_mask = tf.placeholder(shape=[None, None], name=mask_not_null_tname, dtype=tf.bool)
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
    def __init__(self, image_feature_dim, graph=None):

        image_input_tname = "image_feat_input"

        if graph is not None:
            self.image_feat_input = get_tensor_by_name(graph, image_input_tname)
        else:
            self.image_feat_input = tf.placeholder(shape=[None, image_feature_dim], name=image_input_tname,
                                                   dtype=tf.float32)

    def get_image_features(self):
        return self.image_feat_input

    def get_hidden_dim(self):
        return self.hidden_dim

    def pre_feed(self, image_features):
        self.feed_buff = {
            self.image_feat_input: image_features
        }


class MetadataInput(NetWorkInput):
    def __init__(self, graph=None):
        label_tname = "labels"
        if graph is not None:
            self.labels = get_tensor_by_name(graph, label_tname)
        else:
            self.labels = tf.placeholder(shape=[None], name=label_tname, dtype=tf.int32)

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

        with tf.variable_scope("lstm_output"):
            cell = tf.nn.rnn_cell.LSTMCell(hidden_dim)
            self.lstm_outputs, _ = tf.nn.dynamic_rnn(cell, lstm_input,
                                                     time_major=False, dtype=tf.float32,
                                                     initial_state=initial_state)



    def get_output(self):
        return self.lstm_outputs


class LstmScalarRewardStrategy(object):
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
        else:
            reward_scalar_transformer = reward_config.reward_scalar_transformer

        self.scalar_rewards = tf.squeeze(reward_scalar_transformer(tmp_processing), axis=2, name="scalar_rewards")


    def get_rewards(self):
        return self.scalar_rewards

    class RewardConfig(object):

        def __init__(self, reward_scalar_transformer=None, take_difference=True):
            self.reward_scalar_transformer = reward_scalar_transformer
            self.take_difference = take_difference


class Discriminator(object):
    def __init__(self,
                 caption_input,
                 image_input,
                 metadata_input,
                 reward_config=None,
                 learning_rate=1e-3,
                 hidden_dim=512,
                 graph=None
                 ):
        self.caption_input = caption_input
        self.image_input = image_input
        self.metadata_input = metadata_input
        self.reward_config = reward_config
        self.hidden_dim = hidden_dim

        max_reward_opt_tname = "adam_max_reward"
        rewards_loss_collection_tname = "rewards_and_loss"

        if graph is not None:
            reward_tensors = graph.get_collection(rewards_loss_collection_tname)
            self.loss = reward_tensors[0]
            self.masked_reward = reward_tensors[1]
            self.mean_reward_per_sentence = reward_tensors[2]

            self.update_op = graph.get_operation_by_name(max_reward_opt_tname)
        else:
            lstm = self._combine_input_to_lstm()
            rewards = LstmScalarRewardStrategy(lstm.get_output(), reward_config).get_rewards()
            rewards_and_loss = self._compute_loss(rewards)
            self.loss, self.masked_reward, self.mean_reward_per_sentence = rewards_and_loss
            [tf.add_to_collection(rewards_loss_collection_tname, t) for t in rewards_and_loss]

            self.update_op = tf.train.AdamOptimizer(learning_rate, name=max_reward_opt_tname).minimize(self.loss)

    def _combine_input_to_lstm(self):
        image_projection = layer_utils.affine_transform(self.image_input.get_image_features(), self.hidden_dim, 'image_proj')
        initial_lstm_state = tf.nn.rnn_cell.LSTMStateTuple(image_projection * 0, image_projection)
        return Lstm(self.hidden_dim, initial_lstm_state, self.caption_input.get_embedding())

    def _compute_loss(self, rewards):
        expanded_signs = self.metadata_input.get_signs()
        unsigned_masked_reward = rewards * self.caption_input.get_not_null_numeric_mask()
        unsigned_mean_reward = tf.reduce_sum(unsigned_masked_reward, axis=1) / self.caption_input.get_not_null_count()
        signed_rewards = expanded_signs * unsigned_masked_reward
        signed_mean_reward_for_each_sentence = tf.reduce_sum(signed_rewards,
                                                             axis=1) / self.caption_input.get_not_null_count()
        mean_reward = tf.reduce_mean(signed_mean_reward_for_each_sentence)
        loss = mean_reward * - 1
        return loss, unsigned_masked_reward, unsigned_mean_reward

    def _get_feed_dict(self):
        feed_dict = {}
        for n_input in [self.caption_input, self.image_input, self.metadata_input]:
            feed_dict.update(n_input.feed())
        return feed_dict

    def train(self, sess):
        _, loss, masked_reward, mean_reward_per_sentence = sess.run(
            [self.update_op, self.loss, self.masked_reward, self.mean_reward_per_sentence],
            feed_dict=self._get_feed_dict())

        return loss, masked_reward, mean_reward_per_sentence

    def test(self, sess):
        loss, masked_reward, mean_reward_per_sentence = sess.run(
            [self.loss, self.masked_reward, self.mean_reward_per_sentence], feed_dict=self._get_feed_dict())
        return loss, masked_reward, mean_reward_per_sentence

    def save_model(self, sess, model_name):
        saver = tf.train.Saver()
        saver.save(sess, model_name)
