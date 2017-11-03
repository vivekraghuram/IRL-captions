from unittest import TestCase

from preprocessing.caption_feature_extraction import tokenize_caption, CaptionTokenizer


class TestCaptionTokenizer(TestCase):

    def test_tokenize_caption(self):
        tokens = tokenize_caption("A man's squa?tting !on his skateboard, and listening to hip-hop music")
        expected = ['a',
                     "man",
                     "'s",
                     'squatting',
                     'on',
                     'his',
                     'skateboard',
                     'and',
                     'listening',
                     'to',
                     'hip-hop',
                     'music']
        self.assertListEqual(tokens, expected)

    def test_captions_to_word_id_arrays(self):

        cap_tokenizer = CaptionTokenizer(min_dif=3, tokenizer=tokenize_caption)
        corpus = ["Hello world right", "hello woRld left", "hello  world"]
        cap_tokenizer.fit(corpus)
        self.assertNotIn("left", cap_tokenizer.words_to_idx)
        self.assertNotIn("right", cap_tokenizer.words_to_idx)
        self.assertIn("hello", cap_tokenizer.words_to_idx)
        self.assertIn("world", cap_tokenizer.words_to_idx)

        word_id_arrays, tokenized = cap_tokenizer.captions_to_word_ids(corpus)
        assert all(map(lambda x: x[0] == 1, word_id_arrays))
        expected = ["hello world <UNK>", "hello world <UNK>", "hello world"]
        assert cap_tokenizer.decode(word_id_arrays) == expected

        assert len(tokenized) == len(corpus)
        assert tokenized[0] == ["hello", "world", "right"]




