from __future__ import division
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utilyties.datasets import GetImages
import numpy as np
import cv2
from utilyties.util import load_classes, non_max_suppres_thres_process, rescale_boxes
import argparse
import os
from darknet import Darknet
import pickle as pkl
import random
from tqdm import tqdm


def arg_parse():
    """
    実行オプションを取得する関数
    """
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--images", dest='images', help="Image / Directory containing images to perform detection upon",
                        default="images", type=str)
    parser.add_argument("--det", dest='det', help="Image / Directory to store detections to",
                        default="result", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence",
                        help="Object Confidence to filter predictions", default=0.7, type=float)
    parser.add_argument("--nms_thresh", dest="nms_thresh",
                        help="NMS Threshhold", default=0.4, type=float)
    parser.add_argument("--cfg", dest='cfgfile', help="Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile",
                        default="weights/yolov3.weights", type=str)
    parser.add_argument("--img_size", dest="img_size", help="each image dimension size",
                        default="416", type=int)
    parser.add_argument("--class", dest="cls", help="path to class label",
                        default="data/coco.names", type=str)
    parser.add_argument("--cuda", dest="cuda", help="use cuda flag True or False",
                        default=True, type=bool)

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    print("\n--- running options ---")
    print(args)
    print("")

    use_cuda = torch.cuda.is_available() and args.cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    # ニューラルネットワークのセットアップ
    print("Loading network......")
    model = Darknet(args.cfgfile).to(device)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    # モデルをevaluationモードにセット
    model.eval()

    # 保存先がないときは作成
    if not os.path.exists(args.det):
        os.makedirs(args.det)

    # 画像を読み込み始めた時間
    load_imgs_time = time.time()

    # 検出画像をロード
    dataset = GetImages(args.images)
    dataloader = DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=4)
    print("\n--- detection images list ---")
    print(dataset.files)
    print(f"\nAll data loaded in {time.time()-load_imgs_time:6.4f} seconds\n")

    # 推論を開始した時間
    start_det_time = time.time()

    # 推論結果を入れるリスト
    imgs = []
    img_detections = []
    det_time_list = []

    # 推論
    with tqdm(dataloader) as pbar:
        for batch_i, (img_paths, input_imgs) in enumerate(pbar):
            batch_start_time = time.time()  # batch一つあたりの時間計測始まり

            input_imgs = Variable(input_imgs.type(Tensor))

            with torch.no_grad():
                detections = model(input_imgs)
                detections = non_max_suppres_thres_process(
                    detections, args.confidence, args.nms_thresh)

            imgs.extend(img_paths)
            img_detections.extend(detections)
            batch_end_time = time.time()  # batch一つあたりの時間計測終わり
            det_time_list.append((batch_start_time, batch_end_time))
            batch_info = f"batch{batch_i} predicted in {batch_end_time-batch_start_time:6.4f} seconds"
            pbar.postfix = batch_info
            pbar.update()

    # 結果を画像に描画
    classes = load_classes(args.cls)
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        img_cv = cv2.imread(path)
        if detections is not None:
            detections = rescale_boxes(
                detections, int(args.img_size), img_cv.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            colors = pkl.load(open("utilyties/pallete", "rb"))
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                color = random.choice(colors)
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, thickness=2)
                label = classes[int(cls_pred)]
                cv2.putText(img_cv, label, (x1, y1),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2)
                cv2.imwrite(f"{args.det}/result_{img_i}.jpg", img_cv)

    torch.cuda.empty_cache()
