#!/bin/bash
source conda activate fun
python ./train_yolov5.py \
--cfg ./config/train_yolov5.yaml \
--train_img_dir /Volumes/Samsung/Dataset/COCO/train_dataset/image \
--train_lab_dir /Volumes/Samsung/Dataset/COCO/train_dataset/label \
--val_img_dir /Volumes/Samsung/Dataset/COCO/val_dataset/image \
--val_lab_dir /Volumes/Samsung/Dataset/COCO/val_dataset/label \
--test_img_dir ./result/coco_test_imgs \
--name_path /Volumes/Samsung/Dataset/COCO/train_dataset/names.txt
# --pretrained_model_path xxx/Programs/YOLO_ckpts/yolov5s_for_voc.pth \