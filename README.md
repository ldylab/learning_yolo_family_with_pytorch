# 基于Pytorch的相关YOLO实现
### YOLOV1基于Pytorch的实现及其改进
- Backbone使用ResNet18；
- 加入SPP的Neck结构；
- 改进Detection head：全连接层->卷积层；
- 改进Prediction层：官方YOLOv1最后使用全连接层来做预测->用1x1卷积在特征图上去做预测；
- 损失函数：
  - 对于objectness，仍使用L2 loss；
  - 对于class，改成使用Cross-entropy loss；
  - 对于tx、ty，改成使用Binary cross-entropy；
  - 对于tw、th，使用L2 loss。


### Reference
- [1] [知乎专栏-YOLO原理](https://www.zhihu.com/column/c_1364967262269693952)
- [2] [YOLOV1-PyTorch](https://github.com/yjh0410/new-YOLOv1_PyTorch)
