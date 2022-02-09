#!/bin/bash
source conda activate troch1.7
python ./val_yolov5.py \
--cfg /home/uih/JYL/Programs/YOLO/config/validation.yaml \
--val_img_dir /home/uih/JYL/Dataset/VOC/val2012/image \
--val_lab_dir /home/uih/JYL/Dataset/VOC/val2012/label \
--pretrained_model_path /home/uih/JYL/Programs/YOLO_ckpts/yolov5s_for_voc.pth \
--model_type small \
--name_path /home/uih/JYL/Dataset/VOC/val2012/names.txt