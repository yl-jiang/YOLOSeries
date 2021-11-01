#!/bin/bash
source conda activate troch1.7
python xxx/detect.py --cfg "xxx/config/detection.yaml" --img_dir "xxx/image" --pretrained_model_path "xxx/model_xlarge.pth" --model_type "xlarge" --name_path "xxx/names.txt"