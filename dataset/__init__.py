#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/8/20 11:15 下午
# @Author  : jyl
# @File    : __init__.py.py
from .base_generator import Generator
from .coco_generator import COCOGenerator


# from .anchor import GPUAnchor, CPUAnchor

from .COCODataLoader import cocodataloader
from .COCOTestDataLoader import cocotestdataloader
from .CommonDataloader import YoloDataloader

from .testdataloader import testdataloader

from .auxiliary_classifier_dataloader import auxiliary_classifier_dataloader

from .WheatTrainDataLoader import WheatTrainDataloader, WheatDataset
from .WheatTestDataLoader import WheatTestDataLoader


