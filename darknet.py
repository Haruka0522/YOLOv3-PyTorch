from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

from util import *

from parse_config import *


def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416, 416))
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :] / 255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    return img_


def create_modules(blocks):
    """
    parse_model_config()関数で読み込んだ情報をPyTorchのレイヤーを重ねてモジュール化する
    """
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3  # RGBで３層だから3を設定
    output_filters = []

    for idx, x in enumerate(blocks[1:]):  # block[0]はハイパーパラメータなので除いている
        module = nn.Sequential()

        # 畳み込み層の場合
        if x["type"] == "convolutional":
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # 畳み込み層を追加
            conv = nn.Conv2d(prev_filters, filters,
                             kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{0}".format(
                idx), conv)  # indexの名前で層を追加している

            # batch normalization層を追加
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(idx), bn)

            # activationをチェック
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(idx), activn)

        # upsampling層の場合
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=stride, mode="bilinear")
            module.add_module("upsample_{0}".format(idx), upsample)

        # route層の場合
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(",")
            start = int(x["layers"][0])
            # endの指定があるときはその値、ないときは0を補完
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            if start > 0:
                start = start - idx
            if end > 0:
                end = end - idx
            route = EmptyLayer()
            module.add_module("route_{0}".format(idx), route)
            if end < 0:
                filters = output_filters[idx+start] + output_filters[idx+end]
            else:
                filters = output_filters[idx+start]

        # short cut層の場合
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{0}".format(idx), shortcut)

        # yolo層の場合
        elif x["type"] == "yolo":
            num_classes = int(x["classes"])
            img_size = int(net_info["height"])
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(x) for x in anchors]
            anchors = [(anchors[i], anchors[i+1])
                       for i in range(0, len(anchors), 2)]  # 2つづつのタプルに区切る
            anchors = [anchors[i] for i in mask]

            detection = YOLOLayer(anchors,num_classes,img_size)
            module.add_module("yolo_{0}".format(idx), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()  # loss関数のインスタンス
        self.bce_loss = nn.BCELoss()  # loss関数のインスタンス
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size

        # 各gridのオフセットを計算
        self.grid_x = torch.arange(g).repeat(
            g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(
            g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor(
            [(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view(
            (1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view(
            (1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None):
        # CUDA対応のTensorに上書きする
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor

        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors,
                   self.num_classes+5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # 出力
        print(prediction.shape)
        x = torch.sigmoid(prediction[..., 0])  # center x
        y = torch.sigmoid(prediction[..., 1])  # center y
        w = prediction[..., 2]  # width
        h = prediction[..., 3]  # height
        pred_conf = torch.sigmoid(prediction[..., 0])  # conf
        pred_cls = torch.sigmoid(prediction[..., 5])  # 予想されたclass

        # grid_sizeと今が一致していない場合、新しくオフセットを計算する
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # anchorsを使ってオフセットとスケールを追加する
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x  # x
        pred_boxes[..., 1] = y.data + self.grid_y  # y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w  # w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h  # h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes)
            ),
            -1
        )

        # targetsが指定されなかった場合
        if targets is None:
            return output, 0

        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )

            # 存在しないオブジェクトを無視するようにする
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(
                pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metricsの処理
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / \
                (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / \
                (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / \
                (obj_mask.sum() + 1e-16)

            # 計算した値をself.metricsにまとめる
            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }

            return output, total_loss


class Darknet(nn.Module):
    def __init__(self, cfg_file,img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(cfg_file)
        self.net_info, self.module_list = create_modules(self.module_defs)
        self.img_size = img_size

    def forward(self, x, targets=None):
        img_dim = x.shape[2]
        loss = 0
        layer_outputs,yolo_outputs = [],[]
        for i ,(module_def,module) in enumerate(zip(self.module_defs,self.module_list)):
            module_type = module_def["type"]

            # 畳み込み層またはupsample層の場合
            if module_type == "convolutional" or module_type == "upsample":
                x = module(x)

            # route層の場合
            elif module_type == "route":
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
                """
                layers = module_def["layers"]
                layers = [int(a) for a in layers]

                if layers[0] > 0:
                    layers[0] = layers[0] - i
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i
                    map1 = outputs[i+layers[0]]
                    map2 = outputs[i+layers[1]]

                    x = torch.cat((map1, map2), 1)
                """

            # shortcut層の場合
            elif module_type == "shortcut":
                from_ = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[from_]

            # yolo層の場合
            elif module_type == "yolo":
                x,layer_loss = module[0](x,targets,img_dim)
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x)

        yolo_outputs = to_cpu(torch.cat(yolo_outputs,1))
        return detections if targets is None else(loss,yolo_outputs)

    def load_weights(self, weight_file):
        fp = open(weight_file, "rb")
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        weights = np.fromfile(fp, dtype=np.float32)

        # weightsファイルをネットワークのモジュールに読み込む
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.module_defs[i+1]["type"]

            # 畳み込みのブロックの場合
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.module_defs[b+1]["batch_normalize"])
                except:
                    batch_normalize = 0
                conv = model[0]
                if batch_normalize:
                    bn = model[1]

                    num_bn_biases = bn.bias.numel()

                    bn_biases = torch.from_numpy(
                        weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(
                        weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(
                        weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(
                        weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    num_biases = conv.bias.numel()

                    conv_biases = torch.from_numpy(weights[ptr:ptr+num_biases])
                    ptr += num_biases

                    conv_biases = conv_biases.view_as(conv.bias.data)

                    conv.bias.data.copy_(conv_biases)

                num_weights = conv.weight.numel()
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr += num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)


"""このコードが正常にかけているかのテスト"""
"""
#blocks = parse_model_config("cfg/yolov3.cfg")
#print(create_modules(blocks))
"""
"""
model = Darknet("cfg/yolov3.cfg")
inp = get_test_input()
pred = model(inp, torch.cuda.is_available())
print(pred)
"""
"""
model = Darknet("cfg/yolov3.cfg")
model.load_weights("yolov3.weights")
"""
