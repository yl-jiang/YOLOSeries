#!/bin/bash
source conda activate troch1.7
python xxx/val.py --cfg "xxx/config/validation.yaml" --img_dir "xxx/image/" --lab_dir "xxx/label" --pretrained_model_path "xxx/model_xlarge.pth" --model_type "xlarge" --name_path "xxx/names.txt"