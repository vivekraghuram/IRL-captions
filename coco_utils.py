import h5py
import os
import numpy as np
import json

"""
    From coco_utils.py from assignment3
"""


def print_meta_data(data_names, data):
    for (n, d) in zip(data_names, data):
        if type(d) == np.ndarray:
            print(n, type(d), d.shape, d.dtype)
        else:
            print(n, type(d), len(d))


class CocoData(object):
    def __init__(self, data, data_split):
        def data_key(key):
            return "{}_{}".format(data_split, key)

        self.which_split = data_split
        self.image_idx = data[data_key('image_idxs')]
        self.captions_in_word_idx = data[data_key('captions')]
        assert self.captions_in_word_idx.shape[0] == self.image_idx.shape[0]
        self.data_size = self.image_idx.shape[0]
        self.max_caption_len = self.captions_in_word_idx[0].shape[0]

        self.image_features = data[data_key('features')]
        self.image_urls = data[data_key('urls')]

        self.image_paths = data[data_key('image_paths')]

        self.unique_image_num = self.image_features.shape[0]
        unique_url_num = self.image_urls.shape[0]
        assert self.unique_image_num == unique_url_num, "Total image feature ({}) is different from urls ({})".format(self.unique_image_num, unique_url_num)

        if len(self.image_features.shape) == 4:
            self.image_part_num = self.image_features.shape[1] * self.image_features.shape[2]
        else:
            self.image_part_num = None
        self.image_feature_dim = self.image_features.shape[-1]

        print("\nLoaded {} data.".format(data_split))
        data_names = ["Captions", "Image indices", "Image features", "Image urls"]
        data = [self.captions_in_word_idx, self.image_idx, self.image_features, self.image_urls]
        print_meta_data(data_names, data)

    def get_image_features(self, indices):
        return self.image_features[indices]

    def sample(self, batch_size):
        mask = np.random.choice(self.data_size, batch_size)
        captions, image_features, urls = self.get(mask)
        return captions, image_features, urls

    def get(self, indices):
        captions = self.captions_in_word_idx[indices]
        image_idxs = self.image_idx[indices]
        image_features = self.image_features[image_idxs]
        urls = self.image_urls[image_idxs]
        return captions, image_features, urls

    def get_urls_by_data_index(self, indices):
        image_idxs = self.image_idx[indices]
        return self.image_urls[image_idxs]

    def get_urls_by_image_index(self, img_idx):
        return self.image_urls[img_idx]

    def split(self, ratio):
        print("\nSplitting {} data with ratio {}".format(self.which_split, ratio))
        split_at = int(self.data_size * ratio)
        c1, c2 = self.captions_in_word_idx[:split_at], self.captions_in_word_idx[split_at:]
        i1, i2 = self.image_idx[:split_at], self.image_idx[split_at:]

        def build_data_dict(c, i):
            d = dict()
            d['{}_features'.format(self.which_split)] = self.image_features
            d['{}_urls'.format(self.which_split)] = self.image_urls
            d['{}_captions'.format(self.which_split)] = c
            d['{}_image_idxs'.format(self.which_split)] = i
            d['{}_image_paths'.format(self.which_split)] = self.image_paths
            return d

        return CocoData(build_data_dict(c1, i1), self.which_split), CocoData(build_data_dict(c2, i2), self.which_split)


class VocabData(object):
    def __init__(self, data):
        self.word_embedding = data['word_embedding']
        self.word_to_idx = data['word_to_idx']
        self.idx_to_word = data['idx_to_word']

        assert len(self.word_to_idx) == len(self.idx_to_word)
        self.vocab_dim = len(self.word_to_idx)

        print("\nLoaded vocab data.")
        data_names = ["Embedding", "Word to index", "Index to word"]
        data = [self.word_embedding, self.word_to_idx, self.idx_to_word]
        print_meta_data(data_names, data)

        self.START_TOKEN = '<START>'
        self.END_TOKEN = '<END>'
        self.NULL_TOKEN = '<NULL>'
        self.UNK_TOKEN = '<UNK>'

        self.START_ID = self.word_to_idx[self.START_TOKEN]
        self.END_ID = self.word_to_idx[self.END_TOKEN]
        self.NULL_ID = self.word_to_idx[self.NULL_TOKEN]
        self.UNK_ID = self.word_to_idx[self.UNK_TOKEN]

    def embedding(self):
        return self.word_embedding

    def from_word_to_idx(self, word):
        if word not in self.word_to_idx:
            return self.UNK_ID
        else:
            return self.word_to_idx[word]

    def from_idx_to_word(self, idx):
        return self.idx_to_word[idx]

    def decode_captions(self, captions):

        singleton = False
        if captions.ndim == 1:
            singleton = True
            captions = captions[None]
        decoded = []
        N, T = captions.shape
        for i in range(N):
            words = []
            for t in range(T):
                word = self.idx_to_word[captions[i, t]]
                if word != self.NULL_TOKEN:
                    words.append(word)
                if word == self.NULL_TOKEN:
                    break
            decoded.append(' '.join(words))
        if singleton:
            decoded = decoded[0]
        return decoded

    def encode_captions(self, captions):
        max_len = max([len(c) for c in captions])
        caption_ids = np.ones((len(captions), max_len), dtype=np.int) * self.NULL_ID
        for i, c in enumerate(captions):
            for j, tk in enumerate(c):
                caption_ids[i, j] = self.from_word_to_idx(tk)
        return caption_ids

    def get_null_ids(self, shape):
        return np.ones(shape, dtype=np.int32) * self.NULL_ID


def load_coco_data_struct(base_dir='datasets/coco_captioning',
                          max_train=None,
                          source_image_features='vgg16_fc7',
                          is_caption_separated=False,
                          mock_val=False):

    data = load_coco_data(base_dir=base_dir, max_train=max_train,
                          pca_features=False,
                          source_image_features=source_image_features,
                          is_caption_separated=is_caption_separated,
                          image_path_processor=lambda x: "data/" + x[5:])

    vocab_data = VocabData(data)
    train_data = CocoData(data, "train")

    if mock_val:
        val_data = mock_data(real_data=train_data, image_num=1000, data_split="val")
    else:
        val_data = CocoData(data, "val")

    return vocab_data, train_data, val_data


def mock_data(real_data, image_num, data_split):

    def data_key(key):
        return "{}_{}".format(data_split, key)

    data = {}
    img_idx = range(image_num)
    data[data_key('image_idxs')] = np.array(img_idx)
    data[data_key('features')] = real_data.image_features[:image_num]
    data[data_key('captions')] = real_data.captions_in_word_idx[:image_num]
    data[data_key('urls')] = real_data.image_urls[:image_num]
    data[data_key('image_paths')] = real_data.image_paths[:image_num]

    return CocoData(data, data_split)


def load_coco_data(base_dir='datasets/coco_captioning',
                   max_train=None,
                   pca_features=True,
                   source_image_features='vgg16_fc7',
                   is_caption_separated=False,
                   image_path_processor=None):

    data = {}
    if is_caption_separated is False:
        caption_file = os.path.join(base_dir, 'coco2014_captions.h5')
        with h5py.File(caption_file, 'r') as f:
            for k, v in f.items():
                data[k] = np.asarray(v)
    else:
        # Custom caption data generated by script is separated into train/val
        caption_file = os.path.join(base_dir, 'train2014_captions.h5')
        with h5py.File(caption_file, 'r') as f:
            for k, v in f.items():
                data[k] = np.asarray(v)
        caption_file = os.path.join(base_dir, 'val2014_captions.h5')
        with h5py.File(caption_file, 'r') as f:
            for k, v in f.items():
                data[k] = np.asarray(v)

    if pca_features:
        train_feat_file = os.path.join(base_dir, 'train2014_{}_pca.h5'.format(source_image_features))
    else:
        train_feat_file = os.path.join(base_dir, 'train2014_{}.h5'.format(source_image_features))
    with h5py.File(train_feat_file, 'r') as f:
        data['train_features'] = np.asarray(f['features'])

    if pca_features:
        val_feat_file = os.path.join(base_dir, 'val2014_{}_pca.h5'.format(source_image_features))
    else:
        val_feat_file = os.path.join(base_dir, 'val2014_{}.h5'.format(source_image_features))
    with h5py.File(val_feat_file, 'r') as f:
        data['val_features'] = np.asarray(f['features'])

    dict_file = os.path.join(base_dir, 'coco2014_vocab.json')
    with open(dict_file, 'r') as f:
        dict_data = json.load(f)
        for k, v in dict_data.items():
            data[k] = v

    # Load vocab embedding
    embedding_file = os.path.join(base_dir, 'coco2014_vocab_glove.txt')
    word_embedding = load_word_embedding(embedding_file)
    for _, v in dict_data.items():
        assert len(word_embedding) == len(v), "Word embedding has different length from word/id mapping"
    data['word_embedding'] = word_embedding

    train_url_file = os.path.join(base_dir, 'train2014_urls.txt')
    with open(train_url_file, 'r') as f:
        train_urls = np.asarray([line.strip() for line in f])
    data['train_urls'] = train_urls

    val_url_file = os.path.join(base_dir, 'val2014_urls.txt')
    with open(val_url_file, 'r') as f:
        val_urls = np.asarray([line.strip() for line in f])
    data['val_urls'] = val_urls

    extract_image_file_path(data, base_dir, "train", image_path_processor)
    extract_image_file_path(data, base_dir, "val", image_path_processor)

    # Maybe subsample the training data
    if max_train is not None:
        num_train = data['train_captions'].shape[0]
        mask = np.random.randint(num_train, size=max_train)
        data['train_captions'] = data['train_captions'][mask]
        data['train_image_idxs'] = data['train_image_idxs'][mask]

    return data


def extract_image_file_path(data, base_dir, split, image_path_processor=None):
    image_file_path = os.path.join(base_dir, '{}2014_images.txt'.format(split))
    with open(image_file_path, 'r') as f:
        m_func = image_path_processor if image_path_processor else lambda x: x
        file_paths = np.asarray([m_func(line.strip()) for line in f])
    data['{}_image_paths'.format(split)] = file_paths


def load_word_embedding(embedding_file):
    """
    Load word embedding where k-th row corresponds to embedding of word k
    :param embedding_file:
    :return: matrix embedding
    """
    embedding = []
    with open(embedding_file) as data_file:
        for line in data_file:
            embedding.append(np.fromstring(line, dtype=float, sep=' '))
    return np.array(embedding)


def decode_captions(captions, idx_to_word):
    singleton = False
    if captions.ndim == 1:
        singleton = True
        captions = captions[None]
    decoded = []
    N, T = captions.shape
    for i in range(N):
        words = []
        for t in range(T):
            word = idx_to_word[captions[i, t]]
            if word != '<NULL>':
                words.append(word)
            if word == '<END>':
                break
    decoded.append(' '.join(words))
    if singleton:
        decoded = decoded[0]
    return decoded


def sample_coco_minibatch(data, batch_size=100, split='train'):
    split_size = data['%s_captions' % split].shape[0]
    mask = np.random.choice(split_size, batch_size)
    captions = data['%s_captions' % split][mask]
    image_idxs = data['%s_image_idxs' % split][mask]
    image_features = data['%s_features' % split][image_idxs]
    urls = data['%s_urls' % split][image_idxs]
    return captions, image_features, urls
