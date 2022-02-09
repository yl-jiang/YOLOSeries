#!/bin/bash
source conda activate troch1.7
python ./detect_yolov5.py \
--cfg ./config/detection_yolov5.yaml \
--test_img_dir /home/uih/JYL/Dataset/VOC/val2012/image \
--name_path /home/uih/JYL/Dataset/VOC/val2012/names.txt \
--pretrained_model_path ./checkpoints/every_small.pth