## 前言
该项目是在学习[YOLOV5官方代码](https://github.com/ultralytics/yolov5)过程中，加入自己的一些理解以及必要注释，并根据个人习惯对代码结构进行重新组织，该项目主要目的是为了记录学习的过程。

其实，从代码结构上看，YOLO系列的模型代码结构大部分是可以复用的，在学习了官方[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)代码后，也将YOLOX集成到本项目中，使用方法与YOLOV5类似。

目前该项目支持YOLOv5、YOLOv7、RetinaNet、YOLOX。
## 网络结构
以Yolov5s为例，下面分别是该网络的整体结构、backbone、Neck以及构成这些模块用到的小组件的结构。

**注意：**

以下结构图是当时我在较早时候学习yolov5代码时画的，方便对照代码一起查看（对应的模型参见：`YOLOSeries/models/normal/yolov5s_plain_bscp.py`）。该结构图与当前最新版本的yolov5结构有些许差别，但到目前为止模型整体架构并没有特别大的变化，因此仍然具有参考意义。

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

保存的bbox左上角和右下角点的坐标，该坐标点是基于原始图像尺度的，不需要归一化或其他操作。

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
数据集准备好之后只需传入合适的参数，并运行YoloSeries/train_yolov5.py文件即可：
```
$ conda activate your-pytorch-environmention
$ git clone git@github.com:yl-jiang/YOLOSeries.git
$ cd YoloSeries
$ python train_yolov5.py --img_dir "your-image-dir" --lab_dir "your-label_dir" --name_path "your-names-path"
```

类似输出为：
```
Use Nvidia GPU Quadro P5000, find 1 GPU devices, current device id: 0, total memory=15251.6MB, major=6, minor=1, multi_processor_count=20
Checking the consistency of dataset!
- Use time 0.031s
Parser names!
- Use time 0.000s
 epoch       tot       box       cof       cls      tnum     imgsz        lr     AP@.5       mAP   time(s)
#   1       2.475     0.111     1.127     0.000      776       640     0.000069    0.0       0.0              :   2%|▌                         | 36/1686 [00:11<05:34,  4.93it/s]
```

其他参数（优化器、NMS参数、损失函数参数以及其他训练参数等）的配置可到YoloSeries/config/train_yolov5.yaml文件中找到对应的参数名称并修改即可。

---
## 模型测试
将预训练模型文件放到```checkpoints```文件夹，只有测试图片，没有对应的label，运行如下命令（需要指定测试的模型```--model_type```，因为要根据该参数构建对应的网络结构）：
```
$ conda activate your-pytorch-environmention
$ git clone git@github.com:yl-jiang/YOLOSeries.git
$ cd YoloSeries
$ python detect_yolov5.py --cfg "./config/detection.yaml" --img_dir "your-image-dir" --pretrained_model_path "xxx/model_xlarge.pth" --model_type "xlarge" --name_path "xxx/names.txt"
```

类似输出为：
```
[00001/5000] ➡️ 1 :tennis_racket:; 1 ⚾; 1 🧑 (2.71s)
[00002/5000] ➡️ 8 🧑; 2 :bench:; 1 💼; 1 🚆 (2.71s)
[00003/5000] ➡️ 1 🛥 (0.20s)
```

预测结果默认会保存在```YoloSeries/result/predictions```文件夹下，如果想保存到自定义目录，请到```YoloSeries/config/detection.yaml```文件中，修改```output_dir```参数即可。其它与预测代码相关的配置参数可到该文件中修改，根据参数的名称即可大概得知该参数的作用。

---
## 模型性能评估（mAP）

需要评估训练模型的性能（mAP）时，需要准备好测试图片以及对应标签，并分别将图片和标签数据放到不同的文件夹，并运行如下命令(以使用xlarge模型为例)：
```
$ conda activate your-pytorch-environmention
$ git clone git@github.com:yl-jiang/YOLOSeries.git
$ cd YoloSeries
$ python val_yolov5.py --cfg "./config/validation.yaml" --img_dir "your-image-dir" --lab_dir "your-label-dir" --pretrained_model_path "./checkpoints/model_xlarge.pth" --model_type "xlarge" --name_path "./dataset/other/coco_names.txt"
```
类似输出为：
```
[00001/5000] ➡️ 39 :sink:; 19 🚽; 13 📱; 9 🍷; 7 ⏰; 7 👔; 5 🥤; 5 🧑; 3 🍼; 2 :refrigerator:; 1 🛏; 1 🏺 (0.70s)
[00002/5000] ➡️ 77 🧑; 41 🏏; 34 🪑; 20 🧤; 20 ⚾; 11 :bench:; 3 🚗; 1 🍼; 1 🪴 (0.70s)
[00003/5000] ➡️ 43 🧑; 32 :bench:; 24 🪁; 10 🚗; 3 ⚾; 2 🚦; 2 🥏; 1 ☂; 1 🚚 (0.70s)
```
---

## 模型初始化参数

| Model | Baidu | pwd |
| ----------- | ----------- | -------- |
| yolov5 small for wheat | https://pan.baidu.com/s/1rBRPZBA2NzABqCnDNwpYGA | 4brr |
| yolov5 small for voc | https://pan.baidu.com/s/11syrvVvdPSt93M96KCiyYw | a0fe |
| yolox small for wheat | https://pan.baidu.com/s/1lCZ3ZhWzw-haK7m7RtWzvQ  | 7ge4 |
| yolox small for voc | https://pan.baidu.com/s/1I72-oWXJ1xdiatUxjx53jQ | mug9 |
| yolox small for coco | https://pan.baidu.com/s/11G8VBeghpFKU7sIuvJ63YA | 55kg |
| retinanet small for wheat | https://pan.baidu.com/s/1prbKi8xJQI5uGAQynSQ_Lw | frw8 |
| retinanet small for voc | https://pan.baidu.com/s/1ZbKGiGGo6z0Xtul1T4Myng  | rop1 |
| retinanet small for coco | https://pan.baidu.com/s/1yKmwW1M6zk67VVD6WYD6gg  | w1nx |

yoloxs, yoloxm, yoloxl, yoloxx使用的backbone分别与yolov5s, yolov5m, yolov5l, yolov5x一致，有关yolox的预训练模型只在backbone部分载入了yolov5官方的预训练参数，剩余部分layer参数使用随机初始化。

又由于yolov5官方只提供了基于coco数据集的预训练模型，因此本项目中yolov5关于voc数据集的预训练模型在detect部分的layer参数也是使用随机初始化。所有这些预训练模型仅可作为finetune使用，请知悉。

---
## Reference
1. [YOLOV5-Pytorch](https://github.com/ultralytics/yolov5)
2. [YOLOX-Pytorch](https://github.com/Megvii-BaseDetection/YOLOX)
3. [RetinaNet-Pytorch](https://github.com/yhenon/pytorch-retinanet)