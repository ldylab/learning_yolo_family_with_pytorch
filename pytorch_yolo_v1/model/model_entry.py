# from model.base.fcn import CustomFcn
# from model.best.fcn import DeepLabv3Fcn
# from model.better.fcn import Resnet101Fcn
# from model.sota.fcn import LightFcn
from model.resnet.resnet34_model import resnet34
from model.resnet.resnet101_model import resnet101, resnet50
from model.yolov1.yolo_v1 import yolov1
import torch.nn as nn


def select_model(args):
    type2model = {
        'resnet34_fcn': resnet34(num_classes=args.classes_num),
        'resnet101_fcn': resnet101(num_classes=args.classes_num),
        'resnet50_fcn': resnet50(num_classes=args.classes_num),
        'yolov1_fcn': yolov1(num_classes=args.classes_num, )
    }
    model = type2model[args.model_type]
    return model


def equip_multi_gpu(model, args):
    model = nn.DataParallel(model, device_ids=args.gpus)
    return model
