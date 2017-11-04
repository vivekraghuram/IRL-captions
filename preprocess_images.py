from preprocessing.constant import PreProcessingConstant
from preprocessing.image_preprocessing import CocoImageMetaData, VGG19Featurizer, preprocess_image, write_image_data
from pycocotools.coco import COCO

if __name__ == '__main__':

    """
    This assumes having coco original images at directory specified in "image_directory_path" variable below
    The data can be obtained following instruction in http://cocodataset.org/#download
    """

    data_split = PreProcessingConstant.data_splits['train']
    output_directory = '{}/{}'.format(PreProcessingConstant.data_output_directory, data_split)
    ann_file_path = '{}/instances_{}.json'.format(PreProcessingConstant.annotations_directory, data_split)
    image_directory_path = '{}/{}'.format(PreProcessingConstant.image_directory, data_split)

    coco = COCO(ann_file_path)
    all_image_ids = coco.getImgIds()
    all_images = [CocoImageMetaData(img) for img in coco.loadImgs(all_image_ids)]

    img_featurizer = VGG19Featurizer()
    image_feats, image_urls, image_original_ids = preprocess_image(img_featurizer, all_images, image_directory_path)
    write_image_data(image_feats, image_urls, image_original_ids, file_prefix=output_directory)

    # With keras processing here, and upon deleting the session sometimes there's tensorflow exception
    # It's an unresolved bug on tf: https://github.com/tensorflow/cleverhans/issues/17
    # This however should not affect this functionality.

