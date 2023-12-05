# Faster R-CNN

## borrow from
* https://github.com/pytorch/vision/tree/master/torchvision/models/detection

## setup：
* Python3.6/3.7/3.8
* Pytorch1.7.1
* pycocotools(Linux:```pip install pycocotools```; Windows:```pip install pycocotools-windows```(不需要额外安装vs))
* Ubuntu


```

##（Put pretrain model into backbone folder）：
* MobileNetV2 backbone: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
* ResNet50+FPN backbone: https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
 
 
## PASCAL VOC2012 dataset 
* Pascal VOC2012 train/val download：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

## train script
* train mobilenetv2+fasterrcnn，use train_mobilenet.py
* train resnet50+fpn+fasterrcnn，use train_resnet50_fpn.py

## dataset path
* '--data-path'(VOC_root) to store the root folder of 'VOCdevkit' 


## Faster RCNN framework
![Faster R-CNN](fasterRCNN.png) 
