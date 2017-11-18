import re


def get_coco_image_ids_in_order(split, base_dir='datasets/coco_captioning'):
    """
    :return: list of coco image ids - extracted from the image file name
    """
    file_path = "{}/{}_images.txt".format(base_dir, split)
    pattern = re.compile("coco/images/{}/COCO_{}_([\d]+).jpg".format(split, split))

    all_ids = []
    with open(file_path, 'r') as f:
        for line in f:
            img_id = int(re.findall(pattern, line)[0])
            all_ids.append(img_id)

    return all_ids