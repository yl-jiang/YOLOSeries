#!/bin/bash
source conda activate troch1.7
python ./val_yolov5.py \
--cfg xxx/Programs/YOLO/config/validation.yaml \
--val_img_dir xxx/Dataset/VOC/val2012/image \
--val_lab_dir xxx/Dataset/VOC/val2012/label \
--pretrained_model_path xxx/Programs/YOLO_ckpts/yolov5s_for_voc.pth \
--model_type small \
--name_path xxx/Dataset/VOC/val2012/names.txt