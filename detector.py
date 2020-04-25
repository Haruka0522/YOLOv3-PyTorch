from __future__ import division
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from datasets import *
import numpy as np
import cv2
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random


def arg_parse():
    """
    実行オプションを取得する関数
    """
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--images", dest='images', help="Image / Directory containing images to perform detection upon",
                        default="imgs", type=str)
    parser.add_argument("--det", dest='det', help="Image / Directory to store detections to",
                        default="det", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence",
                        help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh",
                        help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help="Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile",
                        default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=int)

    return parser.parse_args()


args = arg_parse()
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 80

classes = load_classes("data/coco.names")

# ニューラルネットワークのセットアップ
print("Loading network......")
model = Darknet(args.cfgfile).to(device)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.eval()

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# 保存先がないときは作成
if not os.path.exists(args.det):
    os.makedirs(args.det)

dataloader = DataLoader(
        ImageFolder(args.images),
        batch_size = int(args.bs),
        shuffle=False,
        num_workers=4)

imgs = []
img_detections = []

#推論
for batch_i,(img_paths,input_imgs) in enumerate(dataloader):
    input_imgs = Variable(input_imgs.type(Tensor))

    with torch.no_grad():
        detections = model(input_imgs)
        detections = non_max_suppres_thres_process(detections,confidence,nms_thesh)

    imgs.extend(img_paths)
    img_detections.extend(detections)

#結果を画像に描画
for img_i,(path,detections) in enumerate(zip(imgs,img_detections)):
    img_cv = cv2.imread(path)
    if detections is not None:
        detections = rescale_boxes(detections,int(args.reso),img_cv.shape[:2])
        unique_labels = detections[:,-1].cpu().unique()
        n_cls_preds = len(unique_labels)
        colors = pkl.load(open("pallete","rb"))
        for x1,y1,x2,y2,conf,cls_conf,cls_pred in detections:
            color = random.choice(colors)
            cv2.rectangle(img_cv,(x1,y1),(x2,y2),color,thickness=2)
            label = classes[int(cls_pred)]
            cv2.putText(img_cv,label,(x1,y1),cv2.FONT_HERSHEY_PLAIN,1.5,color,2)
            cv2.imwrite(args.det+"/result_"+str(img_i)+".jpg",img_cv)

"""
# 結果のprint
print("SUMMARY")
print("----------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
print("{:25s}: {:2.3f}".format(
    "Detection (" + str(len(imlist)) + " images)", output_recast - start_det_loop))
print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
print("{:25s}: {:2.3f}".format(
    "Average time_per_img", (end - load_batch)/len(imlist)))
print("----------------------------------------------------------")
"""
torch.cuda.empty_cache()
