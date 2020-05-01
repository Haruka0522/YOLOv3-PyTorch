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

### 物体検出デモ（画像）
```
python detector_image.py
```

### 物体検出デモ（Webカメラ）
```
python detector_webcam.py
```

### 物体検出デモ（動画）
```
python detector_video.py --video 任意の動画へのパス
```

### 学習デモ
環境構築でダウンロードしたCOCOデータセットを学習するデモです。
```
python train.py
```
#### tensorboardで確認
```
tensorboard --logdir="logs"
```
ブラウザで http://localhost:6006/ にアクセス