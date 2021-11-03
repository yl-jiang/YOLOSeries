该项目是个人在学习[yolov5官方代码](https://github.com/ultralytics/yolov5)过程中，加入自己的一些理解以及必要注释，并根据个人习惯对代码结构进行重新组织，该项目主要目的是为了记录学习的过程。

## 网络结构
以Yolov5s为例，下面分别是该网络的整体结构图、backbone、Neck以及构成这些模块用到的小组件的结构图。
### 总体结构
![blue-print](https://github.com/yl-jiang/Yolov5/blob/main/figures/yolov5.svg)
### 计算过程
![blue-print](https://github.com/yl-jiang/Yolov5/blob/main/figures/detail.svg)
### Neck
![blue-print](https://github.com/yl-jiang/Yolov5/blob/main/figures/neck.png)
### 小组件
![blue-print](https://github.com/yl-jiang/Yolov5/blob/main/figures/detail2.svg)

## 数据集准备
### 1. image以及label文件夹
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
其中000001.txt文件内容的保存格式如下：
```
class_id xmin ymin xmax ymax
```
分隔符使用空格，例如：
```
23 385.53 60.03 600.5 357.19
23 53.01 356.49 185.04 411.68
```
### 2. names文件
另外还需要制作一个names.txt的文件，其保存的是每个class_id对应的名称，例如：
```
0 person
1 bicycle
2 car
...
```
---
## 模型训练
数据集准备好之后只需传入合适的参数，并运行Yolov5/train.py文件即可：
```
$ conda activate your-pytorch-environmention
$ git clone https://github.com/yl-jiang/Yolov5.git
$ cd Yolov5
$ python train.py --img_dir "your-image-dir" --lab_dir "your-label_dir" --name_path "your-names-path"
```

类似输出为：
```
 epoch       tot       box       cof       cls      tnum     imgsz        lr     AP@.5       mAP   time(s)
#   1       2.475     0.111     1.127     0.000      776       640     0.000069    0.0       0.0              :   2%|▌                         | 36/1686 [00:11<05:34,  4.93it/s]
```

其他参数（优化器、NMS参数、损失函数参数以及其他训练参数等）的配置可到xxx/Yolov5/config/train.yaml文件中找到对应的参数名称并修改即可。

---
## 模型测试
将预训练模型文件放到```checkpoints```文件夹，只有测试图片，没有对应的label，运行如下命令（需要指定测试的模型```--model_type```，因为要根据该参数构建对应的网络结构）：
```
$ conda activate your-pytorch-environmention
$ git clone https://github.com/yl-jiang/Yolov5.git
$ cd Yolov5
$ python detect.py --cfg "./config/detection.yaml" --img_dir "your-image-dir" --pretrained_model_path "xxx/model_xlarge.pth" --model_type "xlarge" --name_path "xxx/names.txt"
```

类似输出为：
```
[00001/5000] ➡️ 1 :tennis_racket:; 1 ⚾; 1 🧑 (2.71s)
[00002/5000] ➡️ 8 🧑; 2 :bench:; 1 💼; 1 🚆 (2.71s)
[00003/5000] ➡️ 1 🛥 (0.20s)
```

预测结果默认会保存在```xxx/result/predictions```文件夹下，如果想保存到自定义目录，请到```xxx/config/detection.yaml```文件中，修改修改相关。
有关预测代码相关的配置参数可到```xxx/config/detection.yaml```文件中修改。其中```xxx```代表存放该项目代码文件的文件夹路径。

---
## 模型性能评估（mAP）

需要评估训练模型的性能（mAP）时，需要准备好测试图片以及对应标签，并分别将图片和标签数据放到不同的文件夹，并运行如下命令(使用xlarge模型为例)：
```
$ conda activate your-pytorch-environmention
$ git clone https://github.com/yl-jiang/Yolov5.git
$ cd Yolov5
$ python val.py --cfg "./config/validation.yaml" --img_dir "your-image-dir" --lab_dir "your-label-dir" --pretrained_model_path "./checkpoints/model_xlarge.pth" --model_type "xlarge" --name_path "./dataset/other/coco_names.txt"
```
类似输出为：
```
[00001/5000] ➡️ 39 :sink:; 19 🚽; 13 📱; 9 🍷; 7 ⏰; 7 👔; 5 🥤; 5 🧑; 3 🍼; 2 :refrigerator:; 1 🛏; 1 🏺 (0.70s)
[00002/5000] ➡️ 77 🧑; 41 🏏; 34 🪑; 20 🧤; 20 ⚾; 11 :bench:; 3 🚗; 1 🍼; 1 🪴 (0.70s)
[00003/5000] ➡️ 43 🧑; 32 :bench:; 24 🪁; 10 🚗; 3 ⚾; 2 🚦; 2 🥏; 1 ☂; 1 🚚 (0.70s)
```