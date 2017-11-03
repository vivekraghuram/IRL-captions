from preprocessing.image_feature_extraction import VGG19Featurizer
from preprocessing.image_feature_extraction import load_image_batch as keras_load_image
import numpy as np
import h5py


class CocoImageMetaData(object):

    def __init__(self, meta_data_dict):
        self.file_name = meta_data_dict['file_name']
        self.coco_url = meta_data_dict['coco_url']
        self.original_id = meta_data_dict['id']


def preprocess_image(img_featurizer, all_images_meta, image_directory):
    """
    :param img_featurizer: image featurizer
    :param all_images_meta: list of image CocoImageMetaData
    :param image_directory: image directory containing file name described in the image
    :return:
    """

    def image_file_path(img):
        return "{}/{}".format(image_directory, img.file_name)

    batch_size = 5
    image_feats = []
    image_original_ids = []
    image_urls = []
    max_size = len(all_images_meta)

    for i in range(0, max_size, batch_size):
        if i % (batch_size * 1000) == 0:
            print("Processing batch #", i)
        if i + batch_size > max_size:
            batch = all_images_meta[i:max_size]
        else:
            batch = all_images_meta[i: i + batch_size]
        image_original_ids.append([m.original_id for m in batch])
        image_urls.append([m.coco_url for m in batch])
        file_path_batch = [image_file_path(m) for m in batch]
        img = keras_load_image(file_path_batch)
        features = img_featurizer.featurize(img)
        image_feats.append(features)

    image_feats = np.concatenate(image_feats)
    flatten = lambda l: [item for sublist in l for item in sublist]
    image_urls = list(flatten(image_urls))
    image_original_ids = list(flatten(image_original_ids))

    return image_feats, image_urls, image_original_ids


def write_image_data(image_features, image_urls, image_original_ids, file_prefix=""):

    """
    :param file_prefix: file prefix
    :param image_features: image feature in numpy array
    :param image_urls:  list of string urls
    :param image_original_ids: list of int original ids
    """
    assert image_features.shape[0] == len(image_original_ids)
    assert image_features.shape[0] == len(image_urls)

    with h5py.File('{}_vgg19_fc2.h5'.format(file_prefix), 'w') as f:
        f.create_dataset('features', data=image_features)

    with open('{}_image_original_ids.txt'.format(file_prefix), 'w') as img_idx_f:
        for img_id in image_original_ids:
            img_idx_f.write(str(img_id) + "\n")

    with open('{}_image_urls.txt'.format(file_prefix), 'w') as img_url_f:
        for url in image_urls:
            img_url_f.write(url + "\n")

    return


if __name__ == '__main__':

    from preprocessing.constant import PreProcessingConstant
    from pycocotools.coco import COCO

    data_split = PreProcessingConstant.data_splits['train']

    output_directory = '{}/{}'.format(PreProcessingConstant.data_output_directory, data_split)
    ann_file_path = '{}/instances_{}.json'.format(PreProcessingConstant.annotations_directory, data_split)
    coco = COCO(ann_file_path)

    image_directory_path = '{}/{}'.format(PreProcessingConstant.image_directory, data_split)

    all_image_ids = coco.getImgIds()
    all_images = [CocoImageMetaData(img) for img in coco.loadImgs(all_image_ids)]

    img_featurizer = VGG19Featurizer()
    image_feats, image_urls, image_original_ids = preprocess_image(img_featurizer, all_images, image_directory_path)
    write_image_data(image_feats, image_urls, image_original_ids, file_prefix=output_directory)
