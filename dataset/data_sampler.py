from torch.utils.data import Sampler
import torch.distributed as dist
from torch.utils.data.sampler import BatchSampler as TorchBatchSampler
import itertools
import torch
import pickle
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm


__all__ = ["BatchSampler", "InfiniteSampler", 'TestBatchSampler', 'AspectRatioBatchSampler', 'SequentialSampler']

class AspectRatioBatchSampler(Sampler):
    """
    按照图片的长宽比对输入网络训练图片进行从小到大的重新排列

    copy from: https://github.com/yhenon/pytorch-retinanet
    """

    def __init__(self, data_source, hyp):
        super(AspectRatioBatchSampler, self).__init__(data_source)
        assert hasattr(data_source, "aspect_ratio"), f"data_source should has method of aspect_ratio"
        self.data_source = data_source
        self.batch_size = hyp['batch_size']
        self.drop_last = hyp['drop_last']

        if hyp.get('aspect_ratio_path', None) is not None and Path(hyp['aspect_ratio_path']).exists():
                self.ar = pickle.load(open(hyp['aspect_ratio_path'], 'rb'))
        else:
            self.ar = None

        self.cwd = hyp['current_work_dir']
        self.groups = self.group_images()

    def __iter__(self): 
        """
        必须要被重载的方法。

        Return:
            组成每个batch的image indxies
        """
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        """
        必须要被重载的方法。

        Reutn:
            根据设定的batch size, 返回数据集总共可以分成多少batch。
        """
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        """
        按照图片的长宽比对输入网络训练图片进行从小到大的重新排列。

        Return:
            返回一个list, list中的每一个元素表示的是组成对应batch的image indexies。
        """
        # determine the order of the images
        order = list(range(len(self.data_source))) 
        if self.ar is None:
            idx = list(range(len(self.data_source)))
            tbar = tqdm(idx, total=len(idx))
            tbar.set_description("Sorting dataset by aspect ratio")
            ar = []
            for i in tbar:
                ar.append(self.data_source.aspect_ratio(i))
                tbar.update()
            tbar.close()
            order = np.asarray(ar).argsort()
            pickle.dump(order, open(str(Path(self.cwd) / "dataset" / 'pkl' / 'aspect_ratio.pkl'), 'wb'))
        else:
            sort_i = np.argsort(self.ar)
            order = np.array(order)[sort_i]

        groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)]
                  for i in range(0, len(order), self.batch_size)]
        # divide into groups, one group = one batch
        return groups


class BatchSampler(TorchBatchSampler):
    """
    This batch sampler will generate mini-batches of (mosaic, index) tuples from another sampler.
    It works just like the :class:`torch.utils.data.sampler.BatchSampler`,
    but it will turn on/off the mosaic aug.
    """

    def __init__(self, *args, enable_data_aug=True, **kwargs):
        super(BatchSampler, self).__init__(*args, **kwargs)
        self.enable_data_aug = enable_data_aug

    def __iter__(self):
        for batch in super().__iter__():
            yield [(self.enable_data_aug, idx) for idx in batch]


class InfiniteSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.
    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """

    def __init__(
        self,
        size: int,
        shuffle: bool = True,
        seed = 0,
        rank=0,
        world_size=1,
    ):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._size = size
        assert size > 0
        self._shuffle = shuffle
        self._seed = int(seed)

        if dist.is_available() and dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
        else:
            self._rank = rank
            self._world_size = world_size

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g)
            else:
                yield from torch.arange(self._size)

    def __len__(self):
        return self._size // self._world_size


class SequentialSampler(Sampler):
    """
    SequentialSampler是一个可迭代对象
    """
    def __init__(self, 
        size, 
        shuffle: bool = True,
        seed = 0,
        rank=0,
        world_size=1,):

        self._size = size
        assert size > 0
        self._shuffle = shuffle
        self._seed = int(seed)

        if dist.is_available() and dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
        else:
            self._rank = rank
            self._world_size = world_size
 
    def __iter__(self):
        if not self._shuffle:
            return iter(range(self._size))
        else:
            return iter(np.random.permutation(self._size))
 
    def __len__(self):
        return self._size // self._world_size

# -------------------------------------------------------------------------------------------------------------------------

class TestBatchSampler(TorchBatchSampler):
    """
    This batch sampler will generate mini-batches of (mosaic, index) tuples from another sampler.
    It works just like the :class:`torch.utils.data.sampler.BatchSampler`,
    but it will turn on/off the mosaic aug.
    """

    def __init__(self, *args, **kwargs):
        super(TestBatchSampler, self).__init__(*args, **kwargs)

    def __iter__(self):
        for batch in super().__iter__():
            yield [idx for idx in batch]