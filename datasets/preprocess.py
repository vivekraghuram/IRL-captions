import json
import os
import numpy as np


glove_300_filepath = 'glove/glove.840B.300d.txt'
coco2014_vocab_filepath = 'coco_captioning/coco2014_vocab.json'
coco2014_glove_300_filepath = 'coco_captioning/coco2014_vocab_glove.txt'
embedding_dim = 300


def write_coco_glove_embeddings():

    if os.path.exists(coco2014_glove_300_filepath):
        print('Found Coco Glove embeddings - skip')
        return

    idx_to_word, word_to_idx = load_vocab_mapping()
    assert len(idx_to_word) == len(word_to_idx), "Mapping between word and ids is incompatible in size"
    vocab_size = len(idx_to_word)

    word_embedding = extract_embedding(word_to_idx)

    special_token_ids = get_special_token_ids(word_embedding, idx_to_word)

    additional_embedding = get_additional_one_hot_embedding(special_token_ids, vocab_size)

    word_embedding = np.concatenate([word_embedding, additional_embedding], axis=1)

    write_coco_glove_embedding(word_embedding, coco2014_glove_300_filepath)

    return


def write_coco_glove_embedding(word_embedding, file_path):
    with open(file_path, 'w') as outfile:
        for each_word in word_embedding:
            outfile.write(' '.join(map(str, each_word)) + "\n")


def get_additional_one_hot_embedding(special_token_ids, vocab_size):
    additional_dim = np.zeros((vocab_size, len(special_token_ids)))
    for i, special_token_id in enumerate(special_token_ids):
        additional_dim[special_token_id, i] = 1
    return additional_dim


def get_special_token_ids(word_embedding, idx_to_word):
    special_token_ids = []
    for word_id, word_embedding_sum in enumerate(np.sum(word_embedding, axis=1)):
        if word_embedding_sum == 0:
            print("Special token: ", idx_to_word[word_id])
            special_token_ids.append(word_id)
    return special_token_ids


def extract_embedding(word_to_idx):
    vocab_size = len(word_to_idx)
    word_embedding = np.zeros((vocab_size, embedding_dim))
    with open(glove_300_filepath, encoding="utf-8") as data_file:
        print("Writing GLOVE embeddings...")
        count = 0
        for line in data_file:
            toks = line.split(" ", 1)
            w = toks[0]
            if w in word_to_idx:
                count += 1
                values = toks[1]
                wid = word_to_idx[w]
                word_embedding[wid, :] = np.fromstring(values, dtype=float, sep=' ')
                if count % 100 == 0:
                    print("Words processed: ", count)
    return word_embedding


def load_vocab_mapping():
    with open(coco2014_vocab_filepath, 'r') as f:
        dict_data = json.load(f)
    word_to_idx = dict_data['word_to_idx']
    idx_to_word = dict_data['idx_to_word']
    return idx_to_word, word_to_idx


if __name__ == '__main__':
    write_coco_glove_embeddings()
