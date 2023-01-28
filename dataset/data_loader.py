from functools import partial
from .datasets import *
from torch.utils.data import DataLoader as TorchDataLoader
from .data_augument import *
from .data_sampler import *
from .data_prefetcher import *
from .data_collater import *
from utils import wait_for_the_master
import uuid
import torch
import random
import numpy as np

__all__ = ['build_dataloader', 'build_test_dataloader']



def worker_init_reset_seed(worker_id):
    seed = uuid.uuid4().int % 2**32
    random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    np.random.seed(seed)



class DataLoader(TorchDataLoader):
    def __init__(self, *args, **kwargs):
        self.__initialized = False
        shuffle = False
        batch_sampler = None
        if "shuffle" in kwargs:
            shuffle = kwargs["shuffle"]
        if "sampler" in kwargs:
            sampler = kwargs["sampler"]
        if "batch_sampler" in kwargs:
            batch_sampler = kwargs["batch_sampler"]

        # Use custom BatchSampler
        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = torch.utils.data.sampler.RandomSampler(self.dataset)
                else:
                    sampler = torch.utils.data.sampler.SequentialSampler(self.dataset)
            batch_sampler = BatchSampler(sampler, self.batch_size, self.drop_last)

        self.batch_sampler = batch_sampler
        
        super(DataLoader, self).__init__(*args, **kwargs)
        self.__initialized = True

    def close_data_aug(self):
        self.batch_sampler.enable_data_aug = False


        
def build_dataloader(img_dir, lab_dir, name_path, input_dim, aug_hyp, cache_num, enable_data_aug, 
                     seed, batch_size, num_workers, pin_memory, shuffle, drop_last):
    """
    pytorch dataloader for cocodataset.
    :param kwargs:
    :return:
    """

    if enable_data_aug:
        transforms = Transforms(aug_hyp)
    else:
        transforms = None

    with wait_for_the_master():
        dataset = YOLODataset(img_dir, lab_dir, name_path, input_dim, aug_hyp, cache_num, enable_data_aug, transforms)
        
    sampler = InfiniteSampler(size=len(dataset), shuffle=shuffle, seed=seed if seed else 7)
    batch_sampler = BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=drop_last, enable_data_aug=enable_data_aug)
    collate_fn = partial(fixed_imgsize_collate_fn, dst_size=input_dim)
    dataloader_kwargs = {'num_workers': num_workers, 
                         'pin_memory': pin_memory, 
                         'batch_sampler': batch_sampler, 
                         'worker_init_fn': worker_init_reset_seed, 
                         'collate_fn':collate_fn}

    dataloader = DataLoader(dataset, **dataloader_kwargs)
    if torch.cuda.is_available():
        prefetcher = DataPrefetcher(dataloader)
    else:
        prefetcher = None

    return dataset, dataloader, prefetcher


# -------------------------------------------------------------------------------------------------------------------------

class TestDataLoader(TorchDataLoader):
    def __init__(self, *args, **kwargs):
        self.__initialized = False
        shuffle = False
        batch_sampler = None
        if "shuffle" in kwargs:
            shuffle = kwargs["shuffle"]
        if "sampler" in kwargs:
            sampler = kwargs["sampler"]
        if "batch_sampler" in kwargs:
            batch_sampler = kwargs["batch_sampler"]

        # Use custom BatchSampler
        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = torch.utils.data.sampler.RandomSampler(self.dataset)
                else:
                    sampler = torch.utils.data.sampler.SequentialSampler(self.dataset)
            batch_sampler = TestBatchSampler(sampler, self.batch_size, self.drop_last)

        self.batch_sampler = batch_sampler
        
        super(TestDataLoader, self).__init__(*args, **kwargs)
        self.__initialized = True


def build_test_dataloader(img_dir, input_dim, batch_size=1, num_workers=0):
    # 因为在inference模式下使用letter_resize_img函数对输入图片进行resize，不会将所有输入的图像都resize到相同的尺寸，而是只要符合输入网络的要求即可
    # assert batch_size == 1, f"use inference mode, so please set batch size to 1"
    with wait_for_the_master():
        dataset = TestDataset(img_dir, input_dim)
    
    # 这里强制设定word_size = 1是因为test时不需要dpp
    sampler = InfiniteSampler(len(dataset), shuffle=False, world_size=1)

    # if dist.is_available() and dist.is_initialized():
    #     batch_size = batch_size * 2 // dist.get_world_size()
    batch_sampler = TestBatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False)
    dataloader_kwargs = {"num_workers": num_workers, 
                         "pin_memory": True, 
                         "batch_sampler": batch_sampler, 
                         "worker_init_fn": worker_init_reset_seed,
                         "collate_fn": test_dataset_collate_fn, 
                        }
    dataloader = TestDataLoader(dataset, **dataloader_kwargs)
    if torch.cuda.is_available():
        prefetcher = TestDataPrefetcher(dataloader)
    else:
        prefetcher = None
    return dataset, dataloader, prefetcher
