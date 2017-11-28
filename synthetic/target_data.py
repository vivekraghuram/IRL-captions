import numpy as np
import tensorflow as tf

class TargetData(object):
    def __init__(self, data, embedding, image_feature_dim=32, vocab_size=5000):
        self.data = data
        self._embedding = embedding
        self.image_feature_dim = image_feature_dim
        self.max_caption_len = self.data.shape[1]
        self.vocab_size = vocab_size
        # just cause
        self.START_ID = 0
        self.END_ID = -1
        self.NULL_ID = -1
        self.UNK_ID = -1
        self.START_TOKEN = 0
        self.END_TOKEN = -1
        self.NULL_TOKEN = -1
        self.UNK_TOKEN = -1
        self.image_part_num = None

    def embedding(self):
        return self._embedding

    def set_mode(self, mode):
        assert(mode in ['PG', 'MLE', 'PPO'])
        if mode == 'PG' or mode == 'PPO':
            self.mode = 'PG'
        else:
            self.mode = 'MLE'
        return self

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        return self

    def shuffle(self):
        data_len = len(self.data)
        shuffled_indices = np.arange(data_len)
        np.random.shuffle(shuffled_indices)
        self.data = self.data[shuffled_indices]

    def decode(self, sequences):
        seq_strs = []
        for i in range(len(sequences)):
            seq_strs.append(" ".join(map(str, sequences[i])))
        return seq_strs

    def decode_captions(self, sequences):
        return self.decode(sequences)

    def encode_captions(self, str_sequences):
        """Assumes already tokenized"""
        max_len = max(map(len, str_sequences))
        sequences = np.zeros((len(str_sequences), max_len))
        for idx, str_seq in enumerate(str_sequences):
            ints = [int(tok) for tok in str_seq]
            sequences[idx] = ints
        return sequences

    def get_image_features(self, idxs):
        return np.zeros((len(idxs), self.image_feature_dim))

    @property
    def training_batches(self):
        if self.mode == 'MLE':
            for mini_batch in self._mle_batches:
                yield mini_batch
        else:
            for mini_batch in self._pg_batches:
                yield mini_batch

    @property
    def testing_batches(self):
        for mini_batch in self.training_batches:
            yield mini_batch

    @property
    def _mle_batches(self):
        data_len = len(self.data)
        mask_start, mask_end = 0, self.batch_size

        while mask_end < data_len:
            data_batch = self.data[mask_start:mask_end]
            mask_start, mask_end = mask_end, mask_end + self.batch_size

            image_features = np.zeros((self.batch_size, self.image_feature_dim))
            target_masks = np.ones((self.batch_size, data_batch.shape[1] - 1))
            caption_input = data_batch[:, :-1]
            caption_targets = data_batch[:, 1:]
            yield image_features, caption_input, caption_targets, target_masks

    @property
    def _pg_batches(self):
        data_len = len(self.data)
        mask_start, mask_end = 0, self.batch_size

        while mask_end < data_len:
            data_batch = self.data[mask_start:mask_end]
            mask_start, mask_end = mask_end, mask_end + self.batch_size

            image_features = np.zeros((self.batch_size, self.image_feature_dim))
            train_captions = data_batch[:, :-1]
            target_captions = data_batch[:, 1:]
            decoded_targets = self.decode(target_captions)

            all_ref_captions, keys = {}, []
            for idx, cap in enumerate(decoded_targets):
                all_ref_captions[idx] = cap
                keys.append(idx)

            yield image_features, train_captions, all_ref_captions, keys
