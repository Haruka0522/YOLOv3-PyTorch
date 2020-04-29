from __future__ import division

from utilyties.datasets import ListDataset
from utilyties.util import xywh2xyxy, non_max_suppres_thres_process, calc_predict_scores, calc_evaluation_index

import torch
from torch.autograd import Variable

import numpy as np


def evaluate(model, img_list_path, img_size, batch_size, iou_thres, obj_thres, nms_thres):
    """モデルの評価を行う"""
    model.eval()

    # データローダーの作成
    dataset = ListDataset(list_path=img_list_path,
                          img_size=img_size,
                          multiscale=False)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=1,
                                             collate_fn=dataset.collate_fn)

    # CUDA対応のTensor型を定義
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # この中に(TP,confs,pred)のタプルを入れる

    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        labels += targets[:, 1].tolist()

        # rescale
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        # 画像をPyTorchで扱える形式に
        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppres_thres_process(
                outputs, obj_thres=obj_thres, nms_thres=nms_thres)

        sample_metrics += calc_predict_scores(outputs,
                                              targets, iou_thres=iou_thres)

    true_positives, pred_scores, pred_labels = \
        [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, ap, f1, ap_class = \
        calc_evaluation_index(true_positives, pred_labels,labels,pred_scores)

    return precision, recall, ap, f1, ap_class
