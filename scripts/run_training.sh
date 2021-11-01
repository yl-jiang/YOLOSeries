#!/bin/bash
source conda activate torch1.7
python xxx/train.py --cfg "xxx/config/train.yaml" --img_dir "xxx/image/" --lab_dir "xxx/label" --name_path "xxx/names.txt" --aspect_ratio_path "xxx/dataset/pkl/aspect_ratio.pkl"