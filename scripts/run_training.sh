#!/bin/bash
source conda activate torch1.7
python ./train_yolov5.py \
--cfg /home/uih/JYL/Programs/YOLO/config/train_yolov5.yaml \
--pretrained_model_path /home/uih/JYL/Programs/YOLO_ckpts/yolov5_small_for_voc.pth \
--train_img_dir /home/uih/JYL/Dataset/VOC/train2012/image \
--train_lab_dir /home/uih/JYL/Dataset/VOC/train2012/label \
--val_img_dir /home/uih/JYL/Dataset/VOC/val2012/image \
--val_lab_dir /home/uih/JYL/Dataset/VOC/val2012/label \
--test_img_dir ./result/coco_test_imgs \
--name_path /home/uih/JYL/Dataset/VOC/train2012/names.txt