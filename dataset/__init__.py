from numpy import isin
from .base_generator import Generator
from .CommonDataloader import YoloDataloader
from .testdataloader import testdataloader
from .auxiliary_classifier_dataloader import auxiliary_classifier_dataloader
from .samplers import YOLOXBatchSampler, InfiniteSampler
from .data_loading import YOLOXDataLoader, worker_init_reset_seed
from .data_prefetcher import DataPrefetcher
import torch

def build_data_prefetcher(dataset, batch_size, seed, num_workers):
    assert isinstance(dataset, torch.utils.data.Dataset)
    
    sampler = InfiniteSampler(len(dataset), seed=seed if seed else 0)

    batch_sampler = YOLOXBatchSampler(
        sampler=sampler,
        batch_size=batch_size,
        drop_last=False,
        mosaic=not no_aug,
    )

    dataloader_kwargs = {"num_workers": num_workers, "pin_memory": True, "shuffle": True}
    dataloader_kwargs["batch_sampler"] = batch_sampler

    # Make sure each process has different random seed, especially for 'fork' method.
    # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
    dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

    train_loader = YOLOXDataLoader(dataset, **dataloader_kwargs)
    prefetcher = DataPrefetcher(train_loader)
    # max_iter means iters per epoch
    max_iter = len(train_loader)

    # this code show how to use prefetcher in training:
    # inps, targets = prefetcher.next()
    # inps = inps.to(self.data_type)
    # targets = targets.to(self.data_type)
    # targets.requires_grad = False
    return max_iter, prefetcher