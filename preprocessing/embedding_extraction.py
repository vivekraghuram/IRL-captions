import json
import os
import numpy as np
from preprocessing.constant import PreProcessingConstant


class GloveExtractor(object):

    """
    Extract glove embeddings according to ids of the vocabs specified
    """

    def __init__(self, glove_300_filepath, vocab_filepath, output_file_path, dim_size=300):
        self.glove_300_filepath = glove_300_filepath
        self.vocab_filepath = vocab_filepath
        self.output_file_path = output_file_path
        self.embedding_dim = dim_size

    def get_coco_glove_embeddings(self):
        idx_to_word, word_to_idx = self.load_vocab_mapping()
        assert len(idx_to_word) == len(word_to_idx), "Mapping between word and ids is incompatible in size"
        vocab_size = len(idx_to_word)

        word_embedding = self.extract_embedding(word_to_idx)

        special_token_additional_embedding = self.special_tokens_additional_one_hot_embedding(vocab_size)

        word_embedding = np.concatenate([word_embedding, special_token_additional_embedding], axis=1)

        self.unk_vocab_not_in_embedding(word_embedding, idx_to_word)
        return word_embedding

    def write_coco_glove_embeddings(self):

        if os.path.exists(self.output_file_path):
            print('Found Coco Glove embeddings - skip')
            return

        word_embedding = self.get_coco_glove_embeddings()

        self._write_coco_glove_embedding(word_embedding)

        return

    def _write_coco_glove_embedding(self, word_embedding):
        with open(self.output_file_path, 'w') as outfile:
            for each_word in word_embedding:
                outfile.write(' '.join(map(str, each_word)) + "\n")

    def special_tokens_additional_one_hot_embedding(self, vocab_size):

        additional_dim = np.zeros((vocab_size, len(PreProcessingConstant.special_tokens)))
        for tkn in PreProcessingConstant.special_tokens:
            special_token_id = PreProcessingConstant.special_token_ids[tkn]
            additional_dim[special_token_id, special_token_id] = 1
        return additional_dim

    def unk_vocab_not_in_embedding(self, word_embedding, idx_to_word):
        unk_id = PreProcessingConstant.special_token_ids[PreProcessingConstant.unk_token]
        special_token_ids = []
        for word_id, word_embedding_sum in enumerate(np.sum(word_embedding, axis=1)):
            if word_embedding_sum == 0:
                print("Unk-ing tokens not found in embedding: ", idx_to_word[word_id])
                word_embedding[word_id, :] = word_embedding[unk_id, :]
                special_token_ids.append(word_id)
        return special_token_ids

    def extract_embedding(self, word_to_idx):
        vocab_size = len(word_to_idx)
        word_embedding = np.zeros((vocab_size, self.embedding_dim))
        with open(self.glove_300_filepath, encoding="utf-8") as data_file:
            print("Writing GLOVE embeddings...")
            count = 0
            for line in data_file:
                toks = line.split(" ", 1)
                w = toks[0]
                if w in word_to_idx:
                    count += 1
                    values = toks[1]
                    wid = word_to_idx[w]
                    embedding_from_string = np.fromstring(values, dtype=float, sep=' ')
                    assert embedding_from_string.shape[0] == self.embedding_dim
                    word_embedding[wid, :] = embedding_from_string
                    if count % 100 == 0:
                        print("Words processed: ", count)

        return word_embedding

    def load_vocab_mapping(self):
        with open(self.vocab_filepath, 'r') as f:
            dict_data = json.load(f)
        word_to_idx = dict_data['word_to_idx']
        idx_to_word = dict_data['idx_to_word']
        return idx_to_word, word_to_idx
