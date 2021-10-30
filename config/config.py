#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/8/20 11:20 下午
# @Author  : jyl
# @File    : config.py
import yaml
from pathlib import Path

cwd = Path("__file__").absolute().parent

class Config:
    def __init__(self) -> None:
        self.config = {}

    def update_config(self, args):
        if args.batch_size:
            self.config.update({'batch_size': args.batch_size})
        if args.input_img_size:
            self.config.update({'input_img_size': [args.input_img_size, args.input_img_size]})
        if args.use_focal_loss:
            self.config.update({'use_focal_loss': args.use_focal_loss})
        if args.img_dir:
            self.config.update({'img_dir': args.img_dir})
        if args.lab_dir:
            self.config.update({'lab_dir': args.lab_dir})
        if args.name_path:
            self.config.update({'name_path': args.name_path})
        if args.cache_num:
            self.config.update({'cache_num': args.cache_num})
        if args.total_epoch:
            self.config.update({'total_epoch': args.total_epoch})
        if args.do_warmup:
            self.config.update({'do_warmup': args.do_warmup})
        if args.optimizer:
            self.config.update({'optimizer': args.optimizer})
        if args.iou_threshold:
            self.config.update({'iou_threshold': args.iou_threshold})
        if args.conf_threshold:
            self.config.update({'conf_threshold': args.conf_threshold})
        if args.cls_threshold:
            self.config.update({'cls_threshold': args.cls_threshold})
        if args.agnostic:
            self.config.update({'agnostic': args.agnostic})
        if args.init_lr:
            self.config.update({'init_lr': args.init_lr})


    def get_config(self, cfg, args=None):
        configs = yaml.load(open(str(cfg)), Loader=yaml.FullLoader)
        for k, v in configs.items():
            self.config.update(v)
        if args:
            self.update_config(args)
        return self.config


if __name__ == "__main__":
    config = Config()
