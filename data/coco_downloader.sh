#!/bin/sh

# CREDIT: https://github.com/pjreddie/darknet/tree/master/scripts/get_coco_dataset.sh

# coco apiをクローン
git clone https://github.com/pdollar/coco
cd coco

mkdir images
cd images

# 画像のダウンロード
wget -c https://pjreddie.com/media/files/train2014.zip
wget -c https://pjreddie.com/media/files/val2014.zip

# 解凍
unzip -q train2014.zip
unzip -q val2014.zip

cd ..

# COCO Metadataをダウンロード
wget -c https://pjreddie.com/media/files/instances_train-val2014.zip
wget -c https://pjreddie.com/media/files/coco/5k.part
wget -c https://pjreddie.com/media/files/coco/trainvalno5k.part
wget -c https://pjreddie.com/media/files/coco/labels.tgz
tar xzf labels.tgz
unzip -q instances_train-val2014.zip

# セットアップ
paste <(awk "{print \"$PWD\"}" <5k.part) 5k.part | tr -d '\t' > 5k.txt
paste <(awk "{print \"$PWD\"}" <trainvalno5k.part) trainvalno5k.part | tr -d '\t' > trainvalno5k.txt
