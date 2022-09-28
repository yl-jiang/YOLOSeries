#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import itertools
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data.sampler import BatchSampler as torchBatchSampler
from torch.utils.data.sampler import Sampler

import random
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

__all__ = ["YOLOBatchSampler", "InfiniteSampler", "InfiniteAspectRatioBatchSampler", "InfiniteSampler"]

class YOLOBatchSampler(torchBatchSampler):
    """
    This batch sampler will generate mini-batches of (mosaic, index) tuples from another sampler.
    It works just like the :class:`torch.utils.data.sampler.BatchSampler`,
    but it will turn on/off the mosaic aug.
    """

    def __init__(self, *args, do_mosaic=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.do_mosaic = do_mosaic

    def __iter__(self):
        for batch in super().__iter__():
            yield [(self.do_mosaic, idx) for idx in batch]


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

    def __init__(self, size: int, shuffle: bool = True, seed: Optional[int] = 0, rank=0, world_size=1):
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
            self._world_size = dist.get_world_size()  # the number of processes in the current process group
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



class InfiniteAspectRatioBatchSampler(Sampler):
    """
    按照图片的长宽比对输入网络训练图片进行从小到大的重新排列

    copy from: https://github.com/yhenon/pytorch-retinanet
    """

    def __init__(self, dataset, drop_last, aspect_ratio_filepath=None, seed=0, rank=0, world_size=1):
        super(InfiniteAspectRatioBatchSampler, self).__init__(dataset)
        self._seed = int(seed)

        if dist.is_available() and dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()  # the number of processes in the current process group
        else:
            self._rank = rank
            self._world_size = world_size

        self.drop_last = drop_last
        self.data_source = dataset
        if aspect_ratio_filepath is not None and Path(aspect_ratio_filepath).exists():
            self.ar = pickle.load(open(aspect_ratio_filepath, 'rb'))
        else:
            self.ar = None

    def __iter__(self): 
        """
        必须要被重载的方法。

        Return:
            组成每个batch的image indxies
        """
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def __len__(self):
        """
        必须要被重载的方法。

        """
        return len(self.data_source) // self._world_size

    def _infinite_indices(self):
        """
        按照图片的长宽比对输入网络训练图片进行从小到大的重新排列。

        Return:
            返回一个list, list中的每一个元素表示的是组成对应batch的image indexies。
        """
        # determine the order of the images
        ordered = list(range(len(self.data_source))) 
        if self.ar is None:
            idx = list(range(len(self.data_source)))
            tbar = tqdm(idx, total=len(idx))
            tbar.set_description("Sorting dataset by aspect ratio")
            ar = []
            for i in tbar:
                img_arr = self.data_source.load_img(i)
                h, w = img_arr.shape[0], img_arr.shape[1]
                ar.append(w / h)
                tbar.update()
            tbar.close()
            ordered = np.asarray(ar).argsort()
            pickle.dump(ordered, open(str(Path("./data") / 'pkl' / 'aspect_ratio.pkl'), 'wb'))
        else:
            sort_i = np.argsort(self.ar)
            ordered = np.array(ordered)[sort_i]

        while True:
            yield from (i for i in ordered)



class RangeItertor:
    def __init__(self, size: int, shuffle: bool = True, seed: Optional[int] = 0, rank=0, world_size=1):
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
            self._world_size = dist.get_world_size()  # the number of processes in the current process group
        else:
            self._rank = rank
            self._world_size = world_size
        
        self._gen_range_indeies()
        self.index = self._rank

    def __next__(self):
        try:
            _out =  self._indeies[self.index]
        except IndexError:
            raise StopIteration()
        self.index += self._world_size
        return _out
        

    def __iter__(self):
        return self

    def _gen_range_indeies(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        if self._shuffle:
            self._indeies = torch.randperm(self._size, generator=g)
        else:
            self._indeies = torch.arange(self._size)

    def __len__(self):
        return self._size // self._world_size


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

    def __init__(self, size: int, shuffle: bool = True, seed: Optional[int] = 0, rank=0, world_size=1):
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
            self._world_size = dist.get_world_size()  # the number of processes in the current process group
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

