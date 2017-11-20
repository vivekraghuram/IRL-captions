import json
import re
from bisect import bisect_left
import h5py
import numpy as np
from pycocotools.coco import COCO
from sklearn.feature_extraction.text import CountVectorizer

from preprocessing.constant import PreProcessingConstant
from preprocessing.image_index_processing import get_coco_image_ids_in_order


def tokenize_caption(sentence):

    '''
        Remove trailing "." and white space, remove all punctuations except for ' and - , and lowercase and tokenize separated by white space
        eg. A man's squatting on his skateboard, and listening to hip-hop music.
        is converted to  ['A', "man", "'s", 'squatting', 'on', 'his', 'skateboard', 'and', 'listening', 'to', 'hip-hop', music']
    '''

    trimmed = sentence.lower().strip().rstrip('.')
    punctuation_removed = re.sub(r"[^\w\d'\-\s]+", '', trimmed)
    processed = punctuation_removed.replace("'s ", " 's ")
    return processed.split()


class NumpyIntJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        else:
            return super(NumpyIntJsonEncoder, self).default(obj)


class CaptionTokenizer(object):

    def __init__(self, min_dif, tokenizer):
        self.min_df = min_dif
        self.tokenizer = tokenizer
        self.count_vectorizer = CountVectorizer(min_df=self.min_df, tokenizer=self.tokenizer)
        self.idx_to_word = dict()
        self.words_to_idx = dict()

    def fit(self, corpus):
        self.count_vectorizer.fit(corpus)
        word_to_idx = self.count_vectorizer.vocabulary_
        idx_to_words = self.count_vectorizer.get_feature_names()
        self.words_to_idx, self.idx_to_word = self._adjust_special_token_mapping(word_to_idx=word_to_idx, idx_to_words=idx_to_words)

    def _adjust_special_token_mapping(self, word_to_idx, idx_to_words):
        offset = len(PreProcessingConstant.special_tokens)

        # create updated dict
        updated_word_to_idx = dict()
        for word, idx in word_to_idx.items():
            updated_word_to_idx[word] = word_to_idx[word] + offset
        # add special tokens
        for i, special_token in enumerate(PreProcessingConstant.special_tokens):
            updated_word_to_idx[special_token] = i

        # append special tokens in word list
        updated_idx_to_words = PreProcessingConstant.special_tokens + idx_to_words

        return updated_word_to_idx, updated_idx_to_words

    def write_word_mappings(self, file_path):

        print("\nWriting {} word/id mappings...".format(len(self.idx_to_word)))
        word_mapping_dict = dict()
        word_mapping_dict['idx_to_word'] = self.idx_to_word
        word_mapping_dict['word_to_idx'] = self.words_to_idx
        dict_str = json.dumps(word_mapping_dict, cls=NumpyIntJsonEncoder)
        with open(file_path, 'w') as f:
            f.write(dict_str)

    def captions_to_word_ids(self, img_cap_pairs, len_limit):

        captions = [p.caption for p in img_cap_pairs]
        print("\nConverting captions to word ids...")
        tokenized_captions = [self.tokenizer(c) for c in captions]

        sentence_length = [len(tokens) for tokens in tokenized_captions]
        max_len = max(sentence_length)
        if len_limit is None:
            len_limit = max_len
        max_len = min(max_len, len_limit)
        print("Average caption len: ", sum(sentence_length) / len(sentence_length))
        print("Max caption len: {} at sentence: {}".format(max_len, np.argmax(sentence_length)))

        adjusted_max_len = max_len + 2  # start and end token
        word_ids_2d_array = np.ones((len(captions), adjusted_max_len), dtype=np.int) * self.words_to_idx[PreProcessingConstant.null_token]

        total_unknowns = 0
        total_tokens = 0
        total_removed = 0
        processed_captions = []
        for i, tokens in enumerate(tokenized_captions):

            if len(tokens) > max_len:
                total_removed += 1
                continue
            total_tokens += len(tokens)
            # all start with start token
            word_ids_2d_array[i][0] = self.words_to_idx[PreProcessingConstant.start_token]

            for j, tk in enumerate(tokens):
                if tk in self.words_to_idx:
                    word_ids_2d_array[i][j + 1] = self.words_to_idx[tk]
                else:
                    total_unknowns += 1
                    word_ids_2d_array[i][j + 1] = self.words_to_idx[PreProcessingConstant.unk_token]

            word_ids_2d_array[i][j + 2] = self.words_to_idx[PreProcessingConstant.end_token]

            img_cap_pairs[i].set_processed(word_ids_2d_array[i], tokens)
            processed_captions.append(img_cap_pairs[i])

        print("Total unknown tokens: ({}/{}), {:.2f}".format(total_unknowns, total_tokens, total_unknowns / total_tokens))
        print("Filtering out {} sentences exceeding set length limit {}".format(total_removed, len_limit))
        return processed_captions

    def decode(self, word_id_mat, joining=True):
        decoded = []
        for word_id_array in word_id_mat:
            count = 1
            sentence = []
            while self.idx_to_word[word_id_array[count]] != PreProcessingConstant.end_token:
                sentence.append(self.idx_to_word[word_id_array[count]])
                count += 1
            if joining:
                decoded.append(" ".join(sentence))
            else:
                decoded.append(sentence)
        return decoded


def build_caption_corpus(annotation_file):
    coco = COCO(annotation_file)
    coco_caption_corpus = []
    for _, v in coco.anns.items():
        coco_caption_corpus.append(v['caption'])
    return coco_caption_corpus


class ImageCaptionPair(object):

    def __init__(self, annotation):
        self.caption = annotation['caption']
        self.img_id = annotation['image_id']
        self.caption_word_ids = None
        self.caption_tokenized = None
        self.img_idx = None

    def set_processed(self, caption_word_ids, caption_tokenized):
        self.caption_word_ids = caption_word_ids
        self.caption_tokenized = caption_tokenized

    def set_image_idx(self, img_idx):
        self.img_idx = img_idx


def build_image_caption_pairs(annotation_file):
    coco = COCO(annotation_file)
    img_cap_pair = []
    for _, v in coco.anns.items():
        img_cap_pair.append(ImageCaptionPair(v))
    return img_cap_pair


def write_image_caption_pairs(image_caption_pairs, file_path_dict):

    print("\nWriting {} image caption pairs...".format(len(image_caption_pairs)))
    caption_path = file_path_dict['caption']
    image_idx_path = file_path_dict['image_idx']
    caption_word_ids_path = file_path_dict['word_ids']

    cap_f = open(caption_path, 'w')
    img_idx_f = open(image_idx_path, 'w')
    cap_word_ids_f = open(caption_word_ids_path, 'w')

    for pair in image_caption_pairs:
        cap_f.write(' '.join(pair.caption_tokenized) + "\n")
        img_idx_f.write(str(pair.img_idx) + "\n")
        cap_word_ids_f.write(' '.join(map(str, pair.caption_word_ids)) + "\n")

    cap_f.close()
    img_idx_f.close()
    cap_word_ids_f.close()


def write_h5_image_caption_pairs(image_caption_pairs, file_path_prefix):

    if "train" in file_path_prefix:
        split = "train"
    elif "val" in file_path_prefix:
        split = "val"
    elif "test" in file_path_prefix:
        split = "test"
    else:
        raise Exception('Unable to infer unknown split type from file name: {}'.format(file_path_prefix))

    all_image_idxs = []
    all_image_ids = []
    all_captions = []
    all_word_token_captions = []
    for i_c in image_caption_pairs:
        all_image_idxs.append(i_c.img_idx)
        all_image_ids.append(i_c.img_id)
        all_captions.append(i_c.caption_word_ids)
        all_word_token_captions.append(' '.join(i_c.caption_tokenized))
    max_sentence_len = max([len(s) for s in all_word_token_captions])
    all_word_token_captions = np.array(all_word_token_captions).astype('|S{}'.format(max_sentence_len))

    with h5py.File('{}_captions.h5'.format(file_path_prefix), 'w') as f:
        f.create_dataset('{}_captions'.format(split), data=all_captions)
        f.create_dataset('{}_image_idxs'.format(split), data=all_image_idxs)
        f.create_dataset('{}_image_ids'.format(split), data=all_image_ids)
        f.create_dataset('{}_text_captions'.format(split), data=all_word_token_captions)


def process_captions(caption_tokenizer, img_cap_pairs, max_caption_length=None):
    return caption_tokenizer.captions_to_word_ids(img_cap_pairs, max_caption_length)


def find_all_indices(unique_coco_img_ids, img_ids_to_find):

    def index(a, x):
        """
        Locate the leftmost value exactly equal to x
        """
        i = bisect_left(a, x)
        if i != len(a) and a[i] == x:
            return i
        raise ValueError

    image_id_and_position_idx_pairs = [(img_id, i) for (i, img_id) in enumerate(unique_coco_img_ids)]
    image_id_and_position_idx_pairs.sort(key=lambda r: r[0])
    sorted_image_ids = [r[0] for r in image_id_and_position_idx_pairs]

    idx_in_sorted_keys = [index(sorted_image_ids, img_id) for img_id in img_ids_to_find]
    caption_img_idx = [image_id_and_position_idx_pairs[idx][1] for idx in idx_in_sorted_keys]
    return caption_img_idx


def process_captions_by_split(caption_tokenizer, annotation_captions_file_path, split, max_caption_length = None, h5_format=True):
    img_cap_pairs = build_image_caption_pairs(annotation_captions_file_path)
    coco_image_ids_in_order = get_coco_image_ids_in_order(split)
    process_image_indices(coco_image_ids_in_order, img_cap_pairs)

    img_cap_pairs = process_captions(caption_tokenizer, img_cap_pairs, max_caption_length=max_caption_length)
    if h5_format:
        write_h5_image_caption_pairs(img_cap_pairs, "{}/{}".format(PreProcessingConstant.data_output_directory, split))
    else:
        output_file_paths = {
            "word_ids": "{}/{}_captions.txt".format(PreProcessingConstant.data_output_directory, split),
            "image_idx": "{}/{}_caption_image_idxs.txt".format(PreProcessingConstant.data_output_directory, split),
            "caption": "{}/{}_text_captions.txt".format(PreProcessingConstant.data_output_directory, split)
        }
        write_image_caption_pairs(img_cap_pairs, output_file_paths)


def process_image_indices(coco_image_ids_in_order, img_cap_pairs):
    img_ids = [p.img_id for p in img_cap_pairs]
    image_indices = find_all_indices(coco_image_ids_in_order, img_ids)
    for (img_idx, img_cap_pair) in zip(image_indices, img_cap_pairs):
        img_cap_pair.set_image_idx(img_idx)