import numpy as np
from coco_utils import load_coco_data, decode_captions
import json

def generate_image_index_to_reference_captions(base_dir="datasets/self_process"):
  data = load_coco_data(base_dir=base_dir, pca_features=False, is_caption_separated=True)

  gts_train = {}
  for cap_idx, img_idx in enumerate(data['train_image_idxs']):
    img_idx = str(img_idx)
    if img_idx not in gts_train:
      gts_train[img_idx] = []

    gts_train[img_idx].append({'caption': decode_captions(data['train_captions'][cap_idx][1:], data['idx_to_word'])})

  with open('train_img_idx_to_captions.json', 'w') as f:
    f.write(json.dumps(gts_train))


  gts_val = {}
  for cap_idx, img_idx in enumerate(data['val_image_idxs']):
    img_idx = str(img_idx)
    if img_idx not in gts_val:
      gts_val[img_idx] = []

    gts_val[img_idx].append({'caption': decode_captions(data['val_captions'][cap_idx][1:], data['idx_to_word'])})

  with open('val_img_idx_to_captions.json', 'w') as f:
    f.write(json.dumps(gts_val))


class GeneratorData(object):

  def __init__(self,
               mode='PG',
               batch_size=50,
               START_TOKEN='<START>',
               END_TOKEN='<END>',
               NULL_TOKEN='<NULL>',
               UNK_TOKEN='<UNK>'):
    self.mode = mode
    self.batch_size = batch_size

    self.data = load_coco_data(base_dir="datasets/self_process", pca_features=False, is_caption_separated=True)

    self.vocab_dim          = len(self.data['word_to_idx'])
    self.image_feature_dim  = self.data['val_features'].shape[1]
    self.word_embedding_dim = self.data['word_embedding'].shape[1]

    self.NULL_ID  = self.data['word_to_idx'][NULL_TOKEN]
    self.START_ID = self.data['word_to_idx'][START_TOKEN]
    self.END_ID   = self.data['word_to_idx'][END_TOKEN]
    self.UNK_ID   = self.data['word_to_idx'][UNK_TOKEN]

    self.valid_splits = ['val', 'train']
    self.index_orders = {}
    self.prep_index_orders()

    self.build_image_idx_to_caption_idxs('train')
    self.build_image_idx_to_caption_idxs('val')

    self.caption_length = self.data['train_captions'].shape[1]

  @property
  def training_batches(self):
    """ Returns an iterator over training data with given batch size and
        format (image_features, inputs, targets, target_masks) """
    for mini_batch in self.batches('train'):
      yield mini_batch

  @property
  def testing_batches(self):
    """ Returns an iterator over testing data with given batch size and
        format (image_features, inputs, targets, target_masks) """
    for mini_batch in self.batches('val'):
      yield mini_batch

  @property
  def word_embedding(self):
    return self.data['word_embedding']

  def build_image_idx_to_caption_idxs(self, split):
    map_dict = {}
    for cap_idx, img_idx in enumerate(self.data['%s_image_idxs' % split]):
      if img_idx not in map_dict:
        map_dict[img_idx] = []
      map_dict[img_idx].append(cap_idx)
    self.data['%s_image_idx_to_caption_idxs' % split] = map_dict

  def decode(self, seq):
    singleton = False
    if seq.ndim == 1:
      singleton = True
      seq = seq[None]

    decoded = []
    N, T = seq.shape
    vocab_length = len(self.data['idx_to_word'])
    for i in range(N):
      words = []
      for t in range(T):
        word = self.data['idx_to_word'][seq[i, t] % vocab_length]
        if word != '<NULL>':
          words.append(word)
        if word == '<END>':
          break
      decoded.append(' '.join(words))
    if singleton:
      decoded = decoded[0]
    return decoded

  def encode(self, captions):
    caption_ids = np.ones((len(captions), self.caption_length), dtype=np.int) * self.NULL_ID
    for i, c in enumerate(captions):
      for j, tk in enumerate(c):
        if tk not in self.data['word_to_idx']:
          tk_idx = self.UNK_ID
        else:
          tk_idx = self.data['word_to_idx'][tk]
        caption_ids[i, j] = tk_idx
    return caption_ids

  def shuffle(self):
    """ Randomly shuffles the data """
    for key in self.index_orders.keys():
      np.random.shuffle(self.index_orders[key])
    return self

  def set_mode(self, mode):
    assert(mode in ['PG', 'MLE', 'PPO'])
    if mode == 'PG' or mode == 'PPO':
      self.mode = 'PG'
    else:
      self.mode = 'MLE'
    return self

  def set_batch_size(self, batch_size):
    self.batch_size = batch_size
    return self

  def split_size(self, split):
    assert(split in self.valid_splits)
    if self.mode == 'PG':
      return self.data['%s_features' % split].shape[0]
    return self.data['%s_captions' % split].shape[0]

  def prep_index_orders(self):
    starting_mode = self.mode
    for split in self.valid_splits:
      for mode in ['PG', 'MLE']:
        split_size = self.set_mode(mode).split_size(split)
        self.index_orders[split + self.mode] = np.arange(split_size)
        np.random.shuffle(self.index_orders[split + self.mode])
    self.set_mode(starting_mode)

  def batches(self, split):
    split_size = self.split_size(split)
    mask_start, mask_end = 0, self.batch_size

    while mask_end < split_size:
      mask = self.index_orders[split + self.mode][mask_start:mask_end]
      a, b, c, d = self.sample_minibatch(mask, split)
      mask_start, mask_end = mask_end, mask_end + self.batch_size
      yield (a, b, c, d)

  def sample_minibatch(self, mask, split):
    if self.mode == 'PG':
      return self.sample_pg_minibatch(mask, split)
    return self.sample_mle_minibatch(mask, split)

  def sample_mle_minibatch(self, mask, split):
    mask = np.random.choice(self.split_size(split), self.batch_size)
    captions = self.data['%s_captions' % split][mask]
    image_idxs = self.data['%s_image_idxs' % split][mask]
    image_features = self.data['%s_features' % split][image_idxs]
    train_captions, target_captions, target_mask = self.get_train_target_caption(captions)
    if split == 'val':
      train_captions = np.ones((self.batch_size, 1)) * self.START_ID
    return image_features, train_captions, target_captions, target_mask

  def sample_pg_minibatch(self, mask, split):
    mask = np.sort(mask)
    assert(mask.shape[0] == self.batch_size)
    image_features = self.data['%s_features' % split][mask]
    keys, all_ref_captions, input_captions = [], {}, np.zeros((self.batch_size, self.caption_length - 1))
    for idx, data_idx in enumerate(mask):
      data_idx = int(data_idx)
      keys.append(data_idx)
      all_ref_captions[data_idx] = []
      caption_idxs = self.data['%s_image_idx_to_caption_idxs' % split][data_idx]
      train_captions, target_captions, _ = self.get_train_target_caption(self.data['%s_captions' % split][caption_idxs])

      decoded_target_captions = self.decode(target_captions)
      for caption in decoded_target_captions:
        all_ref_captions[data_idx].append({'caption': caption})

      input_captions[idx] = train_captions[np.random.choice(train_captions.shape[0])]

    assert(keys == sorted(keys))
    return image_features, input_captions, all_ref_captions, keys

  def get_train_target_caption(self, train_captions_as_word_ids):
    """
    Convert training data: '<START> a variety of fruits and vegetables sitting on a kitchen counter'
    to target: 'a variety of fruits and vegetables sitting on a kitchen counter <END>'
    """
    target_captions_as_word_ids = train_captions_as_word_ids[:, 1:]
    train_captions_as_word_ids = train_captions_as_word_ids[:, :-1]
    not_null_target_mask = target_captions_as_word_ids != self.NULL_ID
    return train_captions_as_word_ids, target_captions_as_word_ids, not_null_target_mask
