#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch


__all__ = ["DataPrefetcher"]


class DataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = DataPrefetcher._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            # 这里根据自己的dataset的__getitem__()函数的输出或者collect_fn函数来定
            outdict = next(self.loader)
        except StopIteration:
            self.next_batch_img = None
            self.next_batch_ann = None
            self.next_batch_img_id = None
            self.next_batch_resize_info = None
            return
        else:
            self.next_batch_img = outdict['img']
            self.next_batch_ann = outdict['ann']
            self.next_batch_img_id = outdict["img_id"]
            self.next_batch_resize_info = outdict["resize_info"]

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.next_batch_ann = self.next_batch_ann.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        next_batch_img = self.next_batch_img
        next_batch_ann = self.next_batch_ann
        next_batch_img_id = self.next_batch_img_id
        next_batch_resize_info = self.next_batch_resize_info
        if next_batch_img is not None:
            self.record_stream(next_batch_img)
        if next_batch_ann is not None:
            next_batch_ann.record_stream(torch.cuda.current_stream())
        self.preload()
        return {"img": next_batch_img, "ann": next_batch_ann, "img_id": next_batch_img_id, 'resize_info': next_batch_resize_info}

    def _input_cuda_for_image(self):
        self.next_batch_img = self.next_batch_img.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())
