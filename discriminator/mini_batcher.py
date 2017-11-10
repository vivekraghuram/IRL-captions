import numpy as np


class MiniBatcher(object):

    def __init__(self, batchable_tuple):
        shapes = [b.shape[0] for b in batchable_tuple]
        assert all([s == shapes[0] for s in shapes]), "Data to be batched together should have the same first dimension"
        self.batchable_tuple = batchable_tuple
        self.data_size = shapes[0]

    def sample(self, batch_size=100):
        assert batch_size <= self.data_size
        mask = np.random.choice(self.data_size, batch_size)
        batch_list = [b[mask] for b in self.batchable_tuple]
        return tuple(batch_list)

    def mix_new_data(self, new_batchable_tuple, mix_ratio=1):

        shapes = [b.shape[0] for b in new_batchable_tuple]
        assert all([s == shapes[0] for s in shapes]), "Data to be batched together should have the same first dimension"
        given_size = new_batchable_tuple[0].shape[0]
        old_batchable_tuple = self.sample(given_size * mix_ratio)
        mixed = []
        for i in range(len(old_batchable_tuple)):
            mixed.append(np.concatenate([old_batchable_tuple[i], new_batchable_tuple[i]]))
        return MiniBatcher(tuple(mixed))



