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


class MixedMiniBatcher(object):

    def __init__(self, batchers, ratios):

        assert sum(ratios) == 1, "ratio add up to one"
        assert len(batchers) == len(ratios), "batcher to mix should have same length as ratios"

        self.ratios = ratios
        self.batchers = batchers
        self.mixture_num = len(ratios)

    def sample(self, batch_size=100):

        sizes = self._get_mixture_batch_size(batch_size)

        from_each_batcher = []
        for b_i, b in enumerate(self.batchers):
            from_each_batcher.append(b.sample(sizes[b_i]))

        item_num = len(from_each_batcher[0])

        mixed = []
        for item_i in range(item_num):
            mixed.append(np.concatenate([result[item_i] for result in from_each_batcher], axis=0))

        idx = np.arange(mixed[0].shape[0])
        np.random.shuffle(idx)
        mixed = [m[idx] for m in mixed]
        return tuple(mixed)

    def _get_mixture_batch_size(self, batch_size):
        sample_sizes = []
        remaining = batch_size
        for i, r in enumerate(self.ratios):
            if i == self.mixture_num - 1:
                sample_sizes.append(remaining)
            else:
                taken = int(r * batch_size)
                sample_sizes.append(taken)
                remaining -= taken
        return sample_sizes


def test_sizes():
    mixed_batcher = MixedMiniBatcher(["a", "b", "c"], [0.33, 0.33, 0.34])
    sizes = mixed_batcher._get_mixture_batch_size(10)
    assert np.array_equal(np.array(sizes), np.array([3, 3, 4]))


def test_mixture_batch():
    content_b1 = (np.array(["a", "b", "c"]), np.array([11, 22, 33]), np.array([True, True, True]))
    content_b2 = (np.array(["x", "y", "z"]), np.array([100, 200, 300]), np.array([False, False, False]))
    batcher1 = MiniBatcher(content_b1)
    batcher2 = MiniBatcher(content_b2)
    mixed_batcher = MixedMiniBatcher([batcher1, batcher2], [0.4, 0.6])
    res = mixed_batcher.sample(3)
    idx1 = []
    idx2 = []
    for i, letter in enumerate(res[0]):
        if letter in content_b1[0]:
            idx1.append(i)
        elif letter in content_b2[0]:
            idx2.append(i)
    assert len(idx1) == 1
    assert len(idx2) == 2
    assert all([v for v in content_b1[2][idx1]])
    assert all([False == v for v in content_b2[2][idx2]])
    assert all([v % 11 == 0 for v in content_b1[1][idx1]])
    assert all([v % 100 == 0 for v in content_b2[1][idx2]])


if __name__ == '__main__':

    test_sizes()

    test_mixture_batch()
