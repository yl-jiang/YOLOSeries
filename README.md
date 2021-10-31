# Yolov5

## 数据集准备
### image以及label文件夹
将image数据和label数据分别存放到不同的文件夹，对应的文件名相同：
```
├── image  
│   ├── 000001.jpg
│   ├── 000002.jpg
|   ├── ...
├── label
│   ├── 000001.txt
│   ├── 000002.txt
│   ├── ...
```
其中000001.txt文件中内容的保存格式按照如下：
```
class_id xmin ymin xmax ymax
```
分隔符使用空格，例如：
```
23 385.53 60.03 600.5 357.19
23 53.01 356.49 185.04 411.68
```
### names文件
还需要制作一个names.txt的文件，其中保存的是每个class_id对应的名称，例如：
```
0 person
1 bicycle
2 car
3 motorcycle
4 airplane
5 bus
6 train
7 truck
8 boat
```
---
## 模型训练
数据集准备好之后只需传入合适的参数，并运行Yolov5/train.py文件即可：
```
$ conda activate your-environmention
$ python train.py --img_dir "your-image-dir" --lab_dir "your-label_dir" --name_path "your-names-path"
```
其他参数（优化器、NMS参数、损失函数参数以及其他训练参数等）的配置可到xxx/Yolov5/config/train.yaml文件中找到对应的参数名称并修改即可。
## 模型测试
