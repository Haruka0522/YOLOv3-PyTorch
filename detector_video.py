from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cv2
from darknet import Darknet
import argparse
from utilyties.datasets import pad_to_square, resize
from utilyties.util import load_classes, non_max_suppres_thres_process, rescale_boxes, cv2pil
import torchvision.transforms as transforms
import pickle as pkl
import random


def arg_parse():
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--confidence", dest="confidence",
                        help="Object Confidence to filter predictions", default=0.7, type=float)
    parser.add_argument("--video",dest="videopath",default="IMG-0011.avi",type=str)
    parser.add_argument("--preview",dest="preview",default=False,type=bool)
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

    return parser.parse_args()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # 実行オプション
    args = arg_parse()
    print("\n--- running options ---")
    print(args)
    print("")

    # ニューラルネットワークのセットアップ
    print("Loading network......")
    model = Darknet(args.cfgfile).to(device)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")
    inp_dim = int(model.net_info["height"])

    # モデルをevaluationモードにセット
    model.eval()

    video = cv2.VideoCapture(args.videopath)

    classes = load_classes(args.cls)

    while video.isOpened():
        ret, frame = video.read()
        # frameがなかったらbreak
        if not ret:
            break
        result = frame.copy()

        # frameを扱いやすい形式に変換する
        img = transforms.ToTensor()(cv2pil(frame))
        img, _ = pad_to_square(img, 0)
        img = resize(img, args.img_size)
        dataloader = DataLoader([img, ], batch_size=1,
                                shuffle=False, num_workers=4)

        # 推論
        for i in dataloader:
            i = Variable(i.type(Tensor))
            with torch.no_grad():
                detections = model(i)
                detections = non_max_suppres_thres_process(
                    detections, args.confidence, args.nms_thresh)
            # 結果描画
            detections = detections[0]
            if detections is not None:
                detections = rescale_boxes(
                    detections, args.img_size, result.shape[:2])
                unique_labels = detections[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                colors = pkl.load(open("utilyties/pallete", "rb"))
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    color = random.choice(colors)
                    cv2.rectangle(result, (x1, y1), (x2, y2),
                                  color, thickness=2)
                    label = classes[int(cls_pred)]
                    cv2.putText(result, label, (x1, y1),
                                cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2)

        cv2.imshow("result", result)
        # qを押すと終了
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
