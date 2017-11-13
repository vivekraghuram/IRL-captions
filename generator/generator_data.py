import numpy as np
from coco_utils import load_coco_data, decode_captions
import json

def generate_image_index_to_reference_captions():
  data = load_coco_data()

  gts_train = {}
  for cap_idx, img_idx in enumerate(data['train_image_idxs']):
    img_idx = str(img_idx)
    if img_idx not in gts_train:
      gts_train[img_idx] = []

    gts_train[img_idx].append({'caption': decode_captions(data['train_captions'][cap_idx][1:], data['idx_to_word'])})

  with open('train_img_idx_to_captions.json', 'wb') as f:
    f.write(json.dumps(gts_train).encode('ascii'))


  gts_val = {}
  for cap_idx, img_idx in enumerate(data['val_image_idxs']):
    img_idx = str(img_idx)
    if img_idx not in gts_val:
      gts_val[img_idx] = []

    gts_val[img_idx].append({'caption': decode_captions(data['val_captions'][cap_idx][1:], data['idx_to_word'])})

  with open('val_img_idx_to_captions.json', 'wb') as f:
    f.write(json.dumps(gts_val).encode('ascii'))


class GeneratorData(object):

  def __init__(self,
               mode='PG',
               batch_size=50,
               START_TOKEN='<START>',
               END_TOKEN='<END>',
               NULL_TOKEN='<NULL>'):
    self.mode = mode
    self.batch_size = batch_size

    self.data = load_coco_data(pca_features=False)

    self.vocab_dim          = len(self.data['word_to_idx'])
    self.image_feature_dim  = self.data['val_features'].shape[1]
    self.word_embedding_dim = self.data['word_embedding'].shape[1]

    self.NULL_ID  = self.data['word_to_idx'][NULL_TOKEN]
    self.START_ID = self.data['word_to_idx'][START_TOKEN]
    self.END_ID   = self.data['word_to_idx'][END_TOKEN]

    self.valid_splits = ['val', 'train']
    self.index_orders = {}
    self.prep_index_orders()

    with open('train_img_idx_to_captions.json', 'rb') as f:
      self.data['train_image_idx_to_captions'] = json.load(f)

    with open('val_img_idx_to_captions.json', 'rb') as f:
      self.data['val_image_idx_to_captions'] = json.load(f)

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
    for split in self.valid_splits:
      split_size = self.split_size(split)
      self.index_orders[split] = np.arange(split_size)
      np.random.shuffle(self.index_orders[split])

  def batches(self, split):
    split_size = self.split_size(split)
    mask_start, mask_end = 0, self.batch_size

    while mask_end < split_size:
      mask = self.index_orders[split][mask_start:mask_end]
      a, b, c, d = self.sample_minibatch(mask, split)
      mask_start, mask_end = mask_end, mask_end + self.batch_size
      yield (a, b, c, d)

  def sample_minibatch(self, mask, split):
    if self.mode == 'PG':
      return self.sample_pg_minibatch(mask, split)
    return self.sample_mle_minibatch(mask, split)

  def sample_mle_minibatch(self, mask, split):
    captions = self.data['%s_captions' % split][mask]
    image_idxs = self.data['%s_image_idxs' % split][mask]
    image_features = self.data['%s_features' % split][image_idxs]
    train_captions, target_captions, target_mask = self.get_train_target_caption(captions)
    if split == 'val':
      training_captions = np.ones((self.batch_size, 1)) * self.START_ID
    return image_features, train_captions, target_captions, target_mask

  def sample_pg_minibatch(self, mask, split):
    image_features = self.data['%s_features' % split][mask]
    keys = []
    captions = {}
    for idx in mask:
      keys.append(str(idx))
      captions[keys[-1]] = self.data['%s_image_idx_to_captions' % split][keys[-1]]

    inputs = np.ones((self.batch_size, 1)) * self.START_ID
    return image_features, inputs, captions, keys

  def get_train_target_caption(self, train_captions_as_word_ids):
    """
    Convert training data: '<START> a variety of fruits and vegetables sitting on a kitchen counter'
    to target: 'a variety of fruits and vegetables sitting on a kitchen counter <END>'
    """
    target_captions_as_word_ids = train_captions_as_word_ids[:, 1:]
    train_captions_as_word_ids = train_captions_as_word_ids[:, :-1]
    not_null_target_mask = target_captions_as_word_ids != self.NULL_ID
    return train_captions_as_word_ids, target_captions_as_word_ids, not_null_target_mask
