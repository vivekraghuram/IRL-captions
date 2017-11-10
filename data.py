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


class Data(object):

  def training_batches(self, batch_size):
    """ Returns an iterator over training data with given batch size and
        format (image_features, inputs, targets, target_masks) """
    raise NotImplementedError

  def testing_batches(self, batch_size):
    """ Returns an iterator over testing data with given batch size and
        format (image_features, inputs, targets, target_masks) """
    raise NotImplementedError

  def shuffle(self):
    """ Shuffles the training data """
    raise NotImplementedError

  @property
  def num_sequences(self):
    raise NotImplementedError

  @property
  def max_sequence_length(self):
    raise NotImplementedError

class COCOData(Data):

  def __init__(self,
               START_TOKEN='<START>',
               END_TOKEN='<END>',
               NULL_TOKEN='<NULL>'):

    self.data = load_coco_data(pca_features=False)

    self.vocab_dim          = len(self.data['word_to_idx'])
    self.image_feature_dim  = self.data['val_features'].shape[1]
    self.word_embedding_dim = self.data['word_embedding'].shape[1]

    self.NULL_ID  = self.data['word_to_idx'][NULL_TOKEN]
    self.START_ID = self.data['word_to_idx'][START_TOKEN]
    self.END_ID   = self.data['word_to_idx'][END_TOKEN]

    self.index_orders = {}

  @property
  def word_embedding(self):
    return self.data['word_embedding']

  def shuffle(self):
    for key in self.index_orders.keys():
      np.random.shuffle(self.index_orders[key])

  def get_train_target_caption(self, train_captions_as_word_ids):
    """
    Convert training data: '<START> a variety of fruits and vegetables sitting on a kitchen counter'
    to target: 'a variety of fruits and vegetables sitting on a kitchen counter <END>'
    """
    target_captions_as_word_ids = train_captions_as_word_ids[:, 1:]
    train_captions_as_word_ids = train_captions_as_word_ids[:, :-1]
    not_null_target_mask = target_captions_as_word_ids != self.NULL_ID
    return train_captions_as_word_ids, target_captions_as_word_ids, not_null_target_mask

  def split_size(self, split):
    return self.data['%s_captions' % split].shape[0]

  def sample_minibatch(self, mask, split):
    captions = self.data['%s_captions' % split][mask]
    image_idxs = self.data['%s_image_idxs' % split][mask]
    image_features = self.data['%s_features' % split][image_idxs]
    urls = self.data['%s_urls' % split][image_idxs]
    return captions, image_features, urls

  def batches(self, batch_size, split):
    split_size = self.split_size(split)

    if split not in self.index_orders:
      self.index_orders[split] = np.arange(split_size)
      np.random.shuffle(self.index_orders[split])

    mask_start, mask_end = 0, batch_size

    while mask_end < split_size:
      mask = self.index_orders[split][mask_start:mask_end]
      captions, image_features, urls = self.sample_minibatch(mask, split)
      train_captions, target_captions, target_mask = self.get_train_target_caption(captions)
      mask_start, mask_end = mask_end, mask_end + batch_size
      yield (image_features, train_captions, target_captions, target_mask)

  def training_batches(self, batch_size):
    for mini_batch in self.batches(batch_size, 'train'):
      yield mini_batch

  def testing_batches(self, batch_size):
    for (i, trc, tac, tm) in self.batches(batch_size, 'val'):
      trc = np.ones((batch_size, 1)) * self.START_ID
      yield (i, trc, tac, tm)

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

class PGData(COCOData):

  def __init__(self,
               START_TOKEN='<START>',
               END_TOKEN='<END>',
               NULL_TOKEN='<NULL>'):
    super().__init__(START_TOKEN, END_TOKEN, NULL_TOKEN)

    with open('train_img_idx_to_captions.json', 'rb') as f:
      self.data['train_image_idx_to_captions'] = json.load(f)

    with open('val_img_idx_to_captions.json', 'rb') as f:
      self.data['val_image_idx_to_captions'] = json.load(f)

  def split_size(self, split):
    return self.data['%s_features' % split].shape[0]

  def sample_minibatch(self, mask, split):
    image_features = self.data['%s_features' % split][mask]
    keys = []
    captions = {}
    for idx in mask:
      keys.append(str(idx))
      captions[keys[-1]] = self.data['%s_image_idx_to_captions' % split][keys[-1]]

    return captions, image_features, keys

  def batches(self, batch_size, split):
    split_size = self.split_size(split)

    if split not in self.index_orders:
      self.index_orders[split] = np.arange(split_size)
      np.random.shuffle(self.index_orders[split])

    mask_start, mask_end = 0, batch_size

    while mask_end < split_size:
      mask = self.index_orders[split][mask_start:mask_end]
      captions, image_features, keys = self.sample_minibatch(mask, split)
      mask_start, mask_end = mask_end, mask_end + batch_size
      inputs = np.ones((batch_size, 1)) * self.START_ID
      yield (image_features, inputs, captions, keys)

  def testing_batches(self, batch_size):
    for mini_batch in self.batches(batch_size, 'val'):
      yield mini_batch
