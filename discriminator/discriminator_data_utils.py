import numpy as np
from discriminator.mini_batcher import MiniBatcher


def create_demo_sampled_batcher(coco_data):
    # create random pairing
    grouped_data_index_by_image_idx = group_data_idx_by_image_idx(
        coco_data.image_idx)
    grouped_captions_by_image_idx = map_grouped_data_index_to_grouped_captions(
        coco_data.captions_in_word_idx,
        grouped_data_index_by_image_idx)
    scrambled_grouped_captions_by_image_idx = scramble_image_caption_pairs(
        coco_data.captions_in_word_idx,
        grouped_data_index_by_image_idx)

    # merge both, and retain flag
    image_idx_demo, captions_demo = flatten_image_caption_groups(
        grouped_captions_by_image_idx)
    image_idx_sampled, caption_sampled = flatten_image_caption_groups(
        scrambled_grouped_captions_by_image_idx)

    demo_batcher = MiniBatcher((image_idx_demo, captions_demo, np.ones(image_idx_demo.shape)))
    sample_batcher = MiniBatcher((image_idx_sampled, caption_sampled, np.zeros(image_idx_sampled.shape)))
    return demo_batcher, sample_batcher


def group_data_idx_by_image_idx(image_ids_in_caption_data_order):
    """
        Group image index, which is the same index position as corresponding caption data, by image id.
        i.e {8731 : [0,60,99,305]} caption and image features at these 4 positions correspond to image id 8731
    """
    image_positions_by_image_id = {}
    for i, img_id in enumerate(image_ids_in_caption_data_order):
        if img_id in image_positions_by_image_id:
            image_positions_by_image_id[img_id].append(i)
        else:
            image_positions_by_image_id[img_id] = [i]
    return image_positions_by_image_id


def map_grouped_data_index_to_grouped_captions(caption_data, grouped_image_index):
    """
        For each image id "key", get its corresponding captions, retrieved by "value" of the image instance indices.
        Image instance indices is assumed to be in the same order as the caption data.
    """
    captions_by_image_id = {}
    for k, v in grouped_image_index.items():
        captions_by_image_id[k] = caption_data[v]
    return captions_by_image_id


def scramble_image_caption_pairs(caption_data, grouped_image_index):

    to_shuffle_caption = np.copy(caption_data)
    np.random.shuffle(to_shuffle_caption)
    return map_grouped_data_index_to_grouped_captions(to_shuffle_caption, grouped_image_index)


def flatten_image_caption_groups(grouped_captions):
    all_captions = []
    all_image_ids = []
    for k, v in grouped_captions.items():
        all_image_ids += [k] * len(v)  # duplicate lables
        all_captions.append(v)
    flat_image_ids = np.array(all_image_ids)
    flat_captions = np.concatenate(all_captions, axis=0)
    assert flat_image_ids.shape[0] == flat_captions.shape[0]
    return flat_image_ids, flat_captions


def merge_demo_sampled(demo_image, demo_caption, sampled_image, sampled_caption):
    is_demo = np.ones(demo_image.shape)
    is_sample = np.ones(sampled_image.shape) * -1
    return np.concatenate([demo_image, sampled_image], axis=0), np.concatenate([demo_caption, sampled_caption],
                                                                               axis=0), np.concatenate(
        [is_demo, is_sample], axis=0)
