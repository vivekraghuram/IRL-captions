from preprocessing.constant import PreProcessingConstant
from preprocessing.image_feature_extraction import MockFeaturizer, VGG19Featurizer
from preprocessing.image_preprocessing import CocoImageMetaData, preprocess_image, write_image_data, merge_h5_feature_batch
from preprocessing.image_index_processing import get_coco_image_ids_in_order
from pycocotools.coco import COCO

if __name__ == '__main__':

    """
    This assumes having coco original images at directory specified in "image_directory_path" variable below
    The data can be obtained following instruction in http://cocodataset.org/#download
    """

    mock_feat = False
    data_split = PreProcessingConstant.data_splits['train']

    output_directory = '{}/{}'.format(PreProcessingConstant.processing_output_directory, data_split)
    ann_file_path = '{}/instances_{}.json'.format(PreProcessingConstant.annotations_directory, data_split)
    image_directory_path = '{}/{}'.format(PreProcessingConstant.image_directory, data_split)

    coco = COCO(ann_file_path)
    all_image_ids = coco.getImgIds()
    coco_image_ids_in_order = get_coco_image_ids_in_order(data_split)
    assert len(all_image_ids) == len(coco_image_ids_in_order)
    assert set(all_image_ids) == set(coco_image_ids_in_order)

    all_images = [CocoImageMetaData(img) for img in coco.loadImgs(coco_image_ids_in_order)]

    layer_name = 'block5_conv4'
    if mock_feat:
        img_featurizer = MockFeaturizer((2, 2, 100))
    else:
        img_featurizer = VGG19Featurizer(layer_name)

    max_size = len(all_images)
    batch_size = 1000

    batch_ids = range(0, max_size, batch_size)

    for i in batch_ids:
        print("Processing batch # {}".format(i))
        if i + batch_size > max_size:
            batch = all_images[i:max_size]
        else:
            batch = all_images[i: i + batch_size]

        image_feats, image_urls, image_original_ids = preprocess_image(img_featurizer, batch, image_directory_path)
        write_image_data(i, image_feats, image_urls, image_original_ids, layer_name, file_prefix=output_directory)

    merge_h5_feature_batch(batch_ids, layer_name, output_directory)

    # With keras processing here, and upon deleting the session sometimes there's tensorflow exception
    # It's an unresolved bug on tf: https://github.com/tensorflow/cleverhans/issues/17
    # This however should not affect this functionality.

