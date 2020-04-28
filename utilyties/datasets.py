import random
import os
import glob
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h-w)
    pad1, pad2 = dim_diff//2, dim_diff-dim_diff//2
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size,
                          mode="nearest").squeeze(0)
    return image


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, multiscale=True, normalized_label=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        # list_pathのファイルを読み込んで画像のpathのリストを作る
        self.label_files = [
            path.replace("images", "labels").replace(
                ".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]

        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.normalized_label = normalized_label
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # 画像に対する処理ここから
        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # 画像をPyTorch tensorとして扱う
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        img_ = cv2.imread(img_path)
        img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        img = transforms.ToTensor()(img_)

        # img.shapeが2次元だったら3次元にする
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape  # colorチャネル情報は使わないので_に割り当てている
        h_factor, w_factor = (h, w) if self.normalized_label else (1, 1)

        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape
        # 画像に対する処理ここまで

        # labelに対する処理ここから
        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # paddingされていないかつscalingされていない画像の座標を抽出
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)

            # paddingの調整
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]

            boxes[:, 1] = ((x1+x2) / 2) / padded_w
            boxes[:, 2] = ((y1+y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        return img_path, img, targets

    def collate_fn(self, batch):
        """
        collate_fnはtorch.utils.data.DataLoaderの引数
        この引数を指定することでDataLoaderが普通のTensorを返すだけでなく、高度なbatchを作ることができる
        """
        paths, imgs, targets = list(zip(*batch))

        # multiscaleがTrueならランダムに選ぶようのリストを作っておく
        if self.multiscale:
            random_scale = list(range(self.min_size, self.max_size+1, 32))

        # targetsから空の要素を取り除く
        targets = [boxes for boxes in targets if boxes is not None]

        #
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)

        # 10番目のbatch毎に新しい画像サイズを選ぶ
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(random_scale)

        # 入力された形状に画像サイズを変更
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1

        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, idx):
        img_path = self.files[idx % len(self.files)]
        # 画像をPyTorch tensorで読み込む
        img = transforms.ToTensor()(Image.open(img_path))
        # padを正方形に
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)
