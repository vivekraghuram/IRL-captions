import h5py
import numpy as np

from preprocessing.image_feature_extraction import load_image_batch as keras_load_image


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

    def image_file_path(img_meta_data):
        return "{}/{}".format(image_directory, img_meta_data.file_name)

    batch_size = 128
    all_image_features = []
    all_image_original_coco_ids = []
    all_image_urls = []
    max_size = len(all_images_meta)

    for i in range(0, max_size, batch_size):
        if i + batch_size > max_size:
            batch = all_images_meta[i:max_size]
        else:
            batch = all_images_meta[i: i + batch_size]

        all_image_original_coco_ids.append([m.original_id for m in batch])
        all_image_urls.append([m.coco_url for m in batch])
        file_path_batch = [image_file_path(m) for m in batch]
        img_batch = keras_load_image(file_path_batch)
        features = img_featurizer.featurize(img_batch)
        all_image_features.append(features)

    all_image_features = np.concatenate(all_image_features)
    flatten = lambda l: [item for sublist in l for item in sublist]
    all_image_urls = list(flatten(all_image_urls))
    all_image_original_coco_ids = list(flatten(all_image_original_coco_ids))

    return all_image_features, all_image_urls, all_image_original_coco_ids


def write_image_data(batch_id, image_features, image_urls, image_original_ids, layer_name, file_prefix=""):

    """
    :param batch_id: batch id of image processed
    :param file_prefix: file prefix
    :param image_features: image feature in numpy array
    :param image_urls:  list of string urls
    :param image_original_ids: list of int original ids
    """
    assert image_features.shape[0] == len(image_original_ids)
    assert image_features.shape[0] == len(image_urls)

    print("Writing batch image data # {}, of feature size {} to {}_...".format(batch_id, image_features.shape, file_prefix))

    with h5py.File('{}_{}_batches.h5'.format(file_prefix, layer_name), 'a') as f:
        f.create_dataset('features_batch_{}'.format(batch_id), data=image_features)

    with open('{}_image_original_ids.txt'.format(file_prefix), 'a') as img_idx_f:
        for img_id in image_original_ids:
            img_idx_f.write(str(img_id) + "\n")

    with open('{}_urls.txt'.format(file_prefix), 'a') as img_url_f:
        for url in image_urls:
            img_url_f.write(url + "\n")

    return


def merge_h5_feature_batch(batch_ids, file_prefix):

    all_feats = []
    with h5py.File('{}_batches.h5'.format(file_prefix), 'r') as f:
        for batch_id in batch_ids:
            batch_feat = np.array(f['features_batch_{}'.format(batch_id)])
            all_feats.append(batch_feat)
    all_feats = np.concatenate(all_feats, axis=0)

    print("Merging all img features to file", all_feats.shape)
    with h5py.File('{}.h5'.format(file_prefix), 'w') as f:
        f.create_dataset('features', data=all_feats)
    print("Done!")


