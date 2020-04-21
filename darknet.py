from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def parse_cfg(cfgfile):
    """
    configファイルを読み取ってくる関数
    cfgファイルのパスを渡すと解析して全てのブロックをdictとして保存する
    """
    cfg_file = open(cfgfile,"r")
    lines = cfg_file.read().split("\n")
    lines = [x for x in lines if len(x) > 0]   #空の行を無視する
    lines = [x for x in lines if x[0] != "#"]  #コメントアウトを無視する
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key,value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks

def create_modules(blocks):
    """
    parse_cfg()関数で読み込んだ情報をPyTorchのレイヤーを重ねてモジュール化する
    """
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3   #RGBで３層だから3を設定
    output_filters = []

    for idx,x in enumerate(blocks[1:]):
        module = nn.Sequential()

        #畳み込み層の場合
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

            #畳み込み層を追加
            conv = nn.Conv2d(prev_filters,filters,kernel_size,stride,pad,bias=bias)
            module.add_module("conv_{0}".format(idx),conv)   #indexの名前で層を追加している

            #batch normalization層を追加
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(idx),bn)

            #activationをチェック
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1,inplace=True)
                module.add_module("leaky_{0}".format(idx),activn)

        #upsampling層の場合
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=2,mode="bilinear")
            module.add_module("upsample_{0}".format(idx),upsample)

        #route層の場合
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(",")
            start = int(x["layers"][0])
            #endの指定があるときはその値、ないときは0を補完
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            if start > 0:
                start = start - idx
            if end > 0:
                end  = end - idx
            route = EmptyLayer()
            module.add_module("route_{0}".format(idx),route)
            if end < 0:
                filters = output_filters[idx+start]  + output_filters[idx+end]
            else:
                filters = output_filters[idx+start]

        #short cut層の場合
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{0}".format(idx),shortcut)

        #yolo層の場合
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(x) for x in anchors]
            anchors = [(anchors[i],anchors[i+1]) for i in range(0,len(anchors),2)]  #2つづつのタプルに区切る
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{0}".format(idx),detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info,module_list)

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer,self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self,anchors):
        super(DetectionLayer,self).__init__()
        self.anchors = anchors

class Darknet(nn.Module):
    def __init__(self,cfg_file):
        super(Darknet,self).__init__()
        self.blocks = parse_cfg(cfg_file)
        self.net_info,self.module_list = create_modules(self.blocks)

    def forward(self,x,CUDA):
        modules = self.blocks[1:]
        outputs = {}   #route層のキャッシュ用

        write = 0
        for i,module in enumerate(modules):
            module_type = module["type"]

            #畳み込み層またはupsample層の場合
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)

            #route層の場合
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if layers[0] > 0:
                    layers[0] = layers[0] - i
                if len(leyers) == 1:
                    x = outputs[i + (layers[0])]
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i
                    map1 = outputs[i+layers[0]]
                    map2 = outputs[i+layers[1]]

                    x = torch.cat((map1,map2),1)

            #shortcut層の場合
            elif module_type == "shortcut":
                from module_type == "shortcut":
                    from_ = int(module["from"])
                    x = outputs[i-1] + outputs[i+from_]




"""このコードが正常にかけているかのテスト
blocks = parse_cfg("cfg/yolov3.cfg")
print(create_modules(blocks))
"""
