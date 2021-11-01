#!/bin/bash
source conda activate troch1.7
python /home/uih/Downloads/Yolov5-main/detect.py --cfg "/home/uih/Downloads/Yolov5-main/config/detection.yaml" --img_dir "/home/uih/JYL/Dataset/COCO2017/val2017" --pretrained_model_path "/home/uih/Downloads/checkpoints/every_for_coco_xlarge.pth" --model_type "xlarge" --name_path "/home/uih/JYL/Dataset/COCO2017/train/names.txt"