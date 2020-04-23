#!/bin/sh

# 普通のYOLOv3のweightsをダウンロード
wget -c https://pjreddie.com/media/files/yolov3.weights

# YOLOv3-tinyのweightsをダウンロード
wget -c https://pjreddie.com/media/files/yolov3-tiny.weights

# backbone network用のweightsをダウンロード
wget -c https://pjreddie.com/media/files/darknet53.conv.74
