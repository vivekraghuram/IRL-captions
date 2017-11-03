import matplotlib.pyplot as plt

from preprocessing.caption_feature_extraction import build_caption_corpus, CaptionTokenizer, tokenize_caption, \
    process_captions_by_split
from preprocessing.constant import PreProcessingConstant


def cut_off_explorations(corpus_to_explore):
    cut_off = range(4, 50, 5)
    vocab_size = []
    for c in cut_off:
        word_tokenizer = CaptionTokenizer(min_dif=c, tokenizer=tokenize_caption)
        word_tokenizer.fit(corpus_to_explore)
        vocab_size.append(len(word_tokenizer.idx_to_word))

    plt.plot(cut_off, vocab_size)
    plt.ylabel('Vocab size')
    plt.xlabel('Cutoff')
    plt.show()


def preprocess_captions():

    word_mapping_output_file_path = "{}/coco2014_vocab.json".format(PreProcessingConstant.data_output_directory)
    caption_input_file_path_pattern = '{}/captions_{}.json'

    # build and save dict from train
    train_data_split = PreProcessingConstant.data_splits['train']
    train_annotation_captions_file_path = caption_input_file_path_pattern.format(PreProcessingConstant.annotations_directory, train_data_split)
    corpus = build_caption_corpus(train_annotation_captions_file_path)
    word_tokenizer = CaptionTokenizer(min_dif=35, tokenizer=tokenize_caption)
    word_tokenizer.fit(corpus)
    word_tokenizer.write_word_mappings(word_mapping_output_file_path)

    # process train
    process_captions_by_split(word_tokenizer, train_annotation_captions_file_path, train_data_split)
    # process val
    val_split = PreProcessingConstant.data_splits['val']
    val_annotation_captions_file_path = caption_input_file_path_pattern.format(PreProcessingConstant.annotations_directory, val_split)
    process_captions_by_split(word_tokenizer, val_annotation_captions_file_path, val_split)


if __name__ == '__main__':
    preprocess_captions()
