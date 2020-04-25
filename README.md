## YOLOv3-PyTorch
YOLOv3をPyTorchを使って実装していきます。
まだ開発段階です。

### 環境構築
#### cloneとpythonのライブラリ
```
git clone https://github.com/Haruka0522/YOLOv3-PyTorch.git
cd YOLOv3-PyTorch
pip install -r requirements.txt
```
#### weightsファイルのダウンロード
```
cd weights
sh weights_downloader.sh
```
#### COCO datasetのダウンロード
```
cd data
sh coco_downloader.sh
```

### 物体検出
```
python detector.py
```