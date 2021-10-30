from models import Yolov5Small, Yolov5Middle, Yolov5Large, Yolov5SmallWithPlainBscp, Yolov5XLarge
from models import Yolov5SmallDW, Yolov5MiddleDW, Yolov5LargeDW, Yolov5XLargeDW

def build_model(model_type):
    if model_type.lower() == "plainsmall":
        return Yolov5SmallWithPlainBscp
    elif model_type.lower() == "middle":
        return Yolov5Middle
    elif model_type.lower() == "large":
        return Yolov5Large
    elif model_type.lower() == "xlarge":
        return Yolov5XLarge
    elif model_type.lower() == "smalldw":
        return Yolov5SmallDW
    elif model_type.lower() == "middledw":
        return Yolov5MiddleDW
    elif model_type.lower() == "largedw":
        return Yolov5LargeDW
    elif model_type.lower() == "xlargedw":
        return Yolov5XLargeDW
    else:
        return Yolov5Small