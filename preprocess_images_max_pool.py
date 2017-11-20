import tensorflow as tf
import tensorflow.contrib.layers as layers
import h5py
import numpy as np
from preprocessing.image_preprocessing import merge_h5_feature_batch

from preprocessing.constant import PreProcessingConstant


def max_pool_and_save_images(batch_ids, file_prefix):
    count = 0
    with tf.Session() as sess:
        with h5py.File('{}_max_pooled_batches.h5'.format(file_prefix), 'a') as max_pooled_file:
            with h5py.File('{}_batches.h5'.format(file_prefix), 'r') as batch_file:
                for batch_id in batch_ids:
                    batch_name_key = 'features_batch_{}'.format(batch_id)
                    batch_feat = np.array(batch_file[batch_name_key])
                    out = layers.max_pool2d(tf.constant(batch_feat), kernel_size=(2, 2), stride=(2, 2))
                    output = sess.run(out)
                    count += 1
                    max_pooled_file.create_dataset(batch_name_key, data=output)
                    print("Processed #{} batch of size {}: ".format(batch_id, output.shape))


if __name__ == '__main__':

    """
        Assumes the unbatched image processed as different "datasets" (key in h5 file) 
    """

    data_split = PreProcessingConstant.data_splits['train']
    output_directory = '{}/{}'.format(PreProcessingConstant.processing_output_directory, data_split)
    layer_name = 'block5_conv4'

    start = 0
    batch_num = 83
    batch_ids = [i * 1000 for i in range(start, batch_num)]
    file_prefix = "{}_{}".format(output_directory, layer_name)
    # max_pool_and_save_images(batch_ids, file_prefix)

    max_pool_file_prefix = "{}_max_pooled".format(file_prefix)

    with h5py.File('{}_batches.h5'.format(max_pool_file_prefix), 'r') as f:
        for i, k in f.items():
            print(i, k.shape)

    merging_batch_ids = [i * 1000 for i in range(batch_num)]
    merge_h5_feature_batch(merging_batch_ids, max_pool_file_prefix)

