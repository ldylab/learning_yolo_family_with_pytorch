import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from data import *
import numpy as np
import cv2
import tools
import time
from data.voc_dataset import VOC_CLASSES, VOC_ROOT
from data.data_entry import select_train_loader, select_eval_loader
from model.model_entry import select_model
from options import prepare_train_args
from model.model_entry import select_model
from data.voc_dataset_transform import BaseTransform
from data.voc_dataset import VOCDetection


parser = argparse.ArgumentParser(description='YOLO Detection')
parser.add_argument('-v', '--version', default='yolo',
                    help='yolo')
parser.add_argument('-d', '--dataset', default='voc',
                    help='voc, coco-val.')
parser.add_argument('-size', '--input_size', default=416, type=int,
                    help='输入图像尺寸')
parser.add_argument('--trained_model', default='02_000000.pth',
                    type=str, help='模型权重的路径')
parser.add_argument('--conf_thresh', default=0.1, type=float,
                    help='得分阈值')
parser.add_argument('--nms_thresh', default=0.50, type=float,
                    help='NMS 阈值')
parser.add_argument('--visual_threshold', default=0.3, type=float,
                    help='用于可视化的阈值参数')
parser.add_argument('--cuda', type=bool, default=True,
                    help='use cuda.')
parser.add_argument('--data_type', type=str, default='voc', help='used in data_entry.py')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--classes_num', type=int, default=20, help='num of object to identify')
parser.add_argument('--model_type', type=str, default='yolov1_fcn', help='used in model_entry.py')

args = parser.parse_args()


def vis(img, bboxes, scores, cls_inds, thresh, class_colors, class_names, class_indexs=None, dataset='voc'):
    if dataset == 'voc':
        for i, box in enumerate(bboxes):
            cls_indx = cls_inds[i]
            xmin, ymin, xmax, ymax = box
            if scores[i] > thresh:
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_colors[int(cls_indx)], 1)
                cv2.rectangle(img, (int(xmin), int(abs(ymin)-20)), (int(xmax), int(ymin)), class_colors[int(cls_indx)], -1)
                mess = '%s' % (class_names[int(cls_indx)])
                cv2.putText(img, mess, (int(xmin), int(ymin-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    elif dataset == 'coco-val' and class_indexs is not None:
        for i, box in enumerate(bboxes):
            cls_indx = cls_inds[i]
            xmin, ymin, xmax, ymax = box
            if scores[i] > thresh:
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_colors[int(cls_indx)], 1)
                cv2.rectangle(img, (int(xmin), int(abs(ymin)-20)), (int(xmax), int(ymin)), class_colors[int(cls_indx)], -1)
                cls_id = class_indexs[int(cls_indx)]
                cls_name = class_names[cls_id]
                # mess = '%s: %.3f' % (cls_name, scores[i])
                mess = '%s' % (cls_name)
                cv2.putText(img, mess, (int(xmin), int(ymin-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    return img
        

def test(net, device, testset, transform, thresh, class_colors=None, class_names=None, class_indexs=None, dataset='voc'):
    num_images = len(testset)
    for index in range(num_images):
        print('Testing image {:d}/{:d}....'.format(index+1, num_images))
        img, _ = testset.pull_image(index)
        h, w, _ = img.shape

        # 预处理图像，并将其转换为tensor类型
        x = torch.from_numpy(transform(img)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
        x = x.unsqueeze(0).to(device)

        t0 = time.time()
        # 前向推理
        bboxes, scores, cls_inds = net(x)
        print("detection time used ", time.time() - t0, "s")
        
        # 将预测的输出映射到原图的尺寸上去
        scale = np.array([[w, h, w, h]])
        bboxes *= scale

        # 可视化检测结果
        img_processed = vis(img, bboxes, scores, cls_inds, thresh, class_colors, class_names, class_indexs, dataset)
        cv2.imshow('detection', img_processed)
        cv2.waitKey(0)
        # print('Saving the' + str(index) + '-th image ...')
        # cv2.imwrite('test_images/' + args.dataset+ '3/' + str(index).zfill(6) +'.jpg', img)


if __name__ == '__main__':
    # 是否使用cuda
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(device)

    # 输入图像的尺寸
    input_size = args.input_size

    torch.manual_seed(args.seed)

    print('test on voc ...')
    class_names = VOC_CLASSES
    class_indexs = None
    num_classes = 20
    dataset = VOCDetection(root=VOC_ROOT, img_size=input_size, image_sets=[('2007', 'test')], transform=None)



    # 用于可视化，给不同类别的边界框赋予不同的颜色，为了便于区分。
    num_classes = 20
    class_names = VOC_CLASSES
    class_indexs = None
    class_colors = [(np.random.randint(255),np.random.randint(255),np.random.randint(255)) for _ in range(num_classes)]

    # 构建模型
    model = select_model(args)
    # OK
    model.trainable = False


    # 加载已训练好的模型权重
    model.load_state_dict(torch.load(args.trained_model, map_location=device))
    model.to(device).eval()
    print('Finished loading model!')

    # 开始测试
    test(net=model,
        device=device, 
        testset=dataset,
        transform=BaseTransform(input_size),
        thresh=args.visual_threshold,
        class_colors=class_colors,
        class_names=class_names,
        class_indexs=class_indexs,
        dataset=args.dataset
        )
