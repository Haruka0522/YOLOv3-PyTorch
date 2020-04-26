from __future__ import division

from darknet import *
from util import *
from logger import *
from parse_config import *
from datasets import *
from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int,
                        default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str,
                        default="cfg/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str,
                        default="cfg/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str,
                        help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416,
                        help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int,
                        default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int,
                        default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False,
                        help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True,
                        help="allow for malti-scale training")

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    print(args)

    logger = Logger("logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 出力用ディレクトリの作成
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # configファイルを受け取る
    data_config = parse_data_config(args.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # 初期modelを構成する
    model = Darknet(args.model_def).to(device)
    model.apply(weights_init_normal)

    # dataloaderを作成
    dataset = ListDataset(train_path, multiscale=args.multiscale_training)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    # 最適化アルゴリズムはAdamを使う
    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj"
    ]

    # ここから実際にループを回して学習する
    for epoch in range(args.epochs):
        model.train()
        start_time = time.time()  # ログ用に始めた時間を記録
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outpus = model(imgs, targets=targets)
            loss.backward()

            if batches_done % args.gradient_accumulations:
                # 各ステップの前に勾配を累積する
                optimizer.step()
                optimizer.zero_grad()

            # ここからログを残す処理
            log_str = "\n --- [Epoch{0}/{1}, Batch {2}/{3}] --- \n"\
                .format(epoch, args.epochs, batch_i, len(dataloader))

            metric_table = [
                ["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            for i, metric in enumerate(metrics):
                formats = {m: "%.6F" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(
                    metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # tensorboard用のログ
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # epochの残り時間を推定する
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(
                seconds=epoch_batches_left*(time.time()-start_time)/(batch_i+1))
            log_str += f"\n --- ETA {time_left}"

            # ログ情報をprint
            print(log_str)

            model.seen += imgs.size(0)

        # args.evaluation_interval回毎にモデルの評価を表示する
        if epoch % args.evaluation_interval == 0:
            print("\n --- Evaluating Model --- ")
            print("モデル評価は未実装")

        if epoch % args.checkpoint_interval == 0:
            torch.save(model.state_dict(),
                       f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
