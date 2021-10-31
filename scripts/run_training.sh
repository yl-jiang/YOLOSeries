#!/bin/bash
conda activate fun
python /Users/ylj/Programs/Python/Yolov5Git/scripts/train.py --cfg "/Users/ylj/Programs/Python/Yolov5Git/config/train.yaml" --img_dir "/Volumes/Samsung/Dataset/GlobalWheat/image" --lab_dir "/Volumes/Samsung/Dataset/GlobalWheat/label" --name_path "/Volumes/Samsung/Dataset/GlobalWheat/names.txt"