from unittest import TestCase
from preprocessing.embedding_extraction import GloveExtractor
from preprocessing.constant import PreProcessingConstant
import numpy as np

class TestGloveExtractor(TestCase):


    def test_unk_vocab_not_in_embedding(self):
        glove_extractor = GloveExtractor("", "", "")

        vocab_size = 6
        embedding_dim = 10
        embedding = np.random.uniform(-1, 1, (vocab_size, embedding_dim))

        word_not_in_glove_id = 4
        word_to_idx = PreProcessingConstant.special_token_ids
        word_to_idx["hello"] = word_not_in_glove_id
        idx_to_word = PreProcessingConstant.special_tokens + ["hello"]

        # not in glove - has all zero values
        embedding[word_not_in_glove_id, :] = np.zeros((1, embedding_dim))

        glove_extractor.unk_vocab_not_in_embedding(embedding, idx_to_word)

        unk_token_id = PreProcessingConstant.special_token_ids[PreProcessingConstant.unk_token]
        assert all([a == b for a, b in zip(embedding[word_not_in_glove_id, :], embedding[unk_token_id, :])])
        assert all([a != b for a, b in zip(embedding[word_not_in_glove_id + 1, :], embedding[unk_token_id, :])])

