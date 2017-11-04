import numpy as np

class Data(object):

  def training_batches(self, batch_size):
    """ Returns an iterator over training data with given batch size and
        format (inputs, targets, target_masks) """
    raise NotImplementedError

  def testing_batches(self, batch_size):
    """ Returns an iterator over testing data with given batch size and
        format (inputs, targets, target_masks) """
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
