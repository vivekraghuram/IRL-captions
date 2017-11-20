from preprocessing.embedding_extraction import GloveExtractor

from preprocessing.constant import PreProcessingConstant

if __name__ == '__main__':

    default = True
    if default:
        base_dir = "datasets/coco_captioning"
    else:
        base_dir = PreProcessingConstant.data_output_directory

    coco2014_vocab_filepath = '{}/coco2014_vocab.json'.format(base_dir)
    coco2014_glove_300_filepath = '{}/coco2014_vocab_glove.txt'.format(base_dir)
    glove_300_filepath = 'datasets/glove/glove.840B.300d.txt'
    glove_embedding_size = 300
    glove_extractor = GloveExtractor(glove_300_filepath,
                                     coco2014_vocab_filepath,
                                     output_file_path=coco2014_glove_300_filepath,
                                     dim_size=glove_embedding_size)
    glove_extractor.write_coco_glove_embeddings()
