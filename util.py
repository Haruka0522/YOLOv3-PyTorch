from __future__ import division

import torch
import numpy as np
import cv2


def bbox_iou(box1, box2):
    """
    IoUの計算をする関数
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.max(b1_x2, b2_x2)
    inter_rect_y2 = torch.max(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
        torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def predict_transform(
        prediction, inp_dim, anchors, num_classes, CUDA=None):
    """
    検出マップを受け取って２次元テンソルに変換する
    """
    if CUDA is None:
        CUDA = torch.cuda.is_available()
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(
        batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(
        batch_size, grid_size ** 2 * num_anchors, bbox_attrs)

    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    # sigmoid関数を通してobjectness scoreを算出する
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # グリッドオフセットを中心座標予測に追加
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)
    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)
    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(
        1, num_anchors).view(-1, 2).unsqueeze(0)
    prediction[:, :, :2] += x_y_offset

    # bounding boxの寸法にanchorを適用
    anchors = torch.FloatTensor(anchors)
    if CUDA:
        anchors = anchors.cuda()
    anchors = anchors.repeat(grid_size**2, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    # class scoreにsigmoid activationを適用
    prediction[:, :, 5:5 +
               num_classes] = torch.sigmoid((prediction[:, :, 5:5+num_classes]))

    # 検出マップのサイズを入力画像のサイズに変更するために、ストライドをかける
    prediction[:, :, :4] *= stride

    return prediction


def non_max_suppres_thres_process(prediction, obj_thres=0.5, nms_thres=0.4):
    """
    predict結果を受け取って、objectness scoreのしきい値処理とNon-maximal suppressionを行う
    また、その結果を書き出す
    obj_thresはobjectness scoreのしきい値
    nms_thresはIoUのしきい値
    """
    # (center_x,center_y,width,height)から(x1,y1,x2,y2)に変換する
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])

    # Noneで埋められた初期outputを生成
    output = [None for _ in range(len(prediction))]

    for image_i, image_pred in enumerate(prediction):
        # objectness scoreのしきい値以下のものを除外
        image_pred = image_pred[image_pred[:, 4] >= obj_thres]

        # Noneが残っている場合はcontinue
        if not image_pred.size(0):
            continue

        # objectness scoreでソート
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat(
            (image_pred[:, :5], class_confs.float(), class_preds.float()), 1)

        # non-maximum suppressionの処理
        keep_boxes = []
        while detections.size(0):
            # IoUの計算としきい値処理
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(
                0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            detections[0, :4] = (
                weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output


def letterbox_image(img, inp_dim):
    """画像をリサイズします"""
    img_h = img.shape[0]
    img_w = img.shape[1]
    w, h = inp_dim
    new_h = int(img_h * min(w/img_w, h/img_h))
    new_w = int(img_w * min(w/img_w, h/img_h))
    resized_image = cv2.resize(
        img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
    canvas[(h-new_h)//2:(h-new_h)//2 + new_h, (w-new_w) //
           2: (w-new_w)//2 + new_w, :] = resized_image

    return canvas


def prep_image(img, inp_dim):
    """
    ニューラルネットワークで扱える形式に画像を変換します
    """
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)

    return img


def load_classes(path):
    """
    クラスのconfigファイルを読み込む
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def to_cpu(tensor):
    """
    PyTorchのTensor型には勾配やGPUの情報が含まれている。
    このままではただの値を取り出しにくいためGPU情報や勾配情報を捨てる
    """
    return tensor.detach().cpu()


def build_targets(pred_boxes, pred_cls, target, anchors, iou_thres):

    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(-1)
    nG = pred_boxes.size(2)

    # 出力用Tensor
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    # boxを基準とする座標情報に変換
    target_boxes = target[:, 2:6] * nG
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]

    # IoUの計算結果からanchorを取得
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)

    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()

    # mask
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # IoUのしきい値処理
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > iou_thres, gj[i], gi[i]] = 0

    # 座標
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()

    # widthとheight
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)

    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1

    # 一番良いanchorを計算
    class_mask[b, best_n, gj, gi] = (
        pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    
    iou_scores[b, best_n, gj, gi] = bbox_iou(
        xywh2xyxy(pred_boxes[b, best_n, gj, gi]), xywh2xyxy(target_boxes))

    tconf = obj_mask.float()

    return_info = \
        [iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf]

    return return_info


def rescale_boxes(boxes, current_dim, original_shape):
    """ bouding boxを元画像の形状に合うように計算する """
    orig_h, orig_w = original_shape

    # 追加されたpaddingの量
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))

    # 追加されたpaddingを削除
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x

    # 元の画像のサイズに修正
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h

    return boxes


def xywh2xyxy(xywh):
    """
    座標の表記を(center_x,center_y,width,height)から(x1,y1,x2,y2)に変換する
    """
    xyxy = xywh.new(xywh.shape)
    xyxy[..., 0] = xywh[..., 0] - xywh[..., 2] / 2
    xyxy[..., 1] = xywh[..., 1] - xywh[..., 3] / 2
    xyxy[..., 2] = xywh[..., 0] + xywh[..., 2] / 2
    xyxy[..., 3] = xywh[..., 1] + xywh[..., 3] / 2

    return xyxy


def calc_predict_scores(outputs,targets,iou_thres):
    """各サンプル毎に予測スコアとラベルを計算する"""
    batch_metrics = []
    for sample_i in range(len(outputs)):

        #Noneだったら何もせずに次へ
        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:,:4]
        pred_scores = output[:,4]
        pred_labels = output[:,-1]

        #true_positivesは全て0で初期化
        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:,0] == sample_i][:,1:]


        #アノテーションされたデータがあるときには
        if len(annotations) >= 1:
            detected_boxes = []
            target_labels = annotations[:,0]
            target_boxes = annotations[:,1]

            for pred_i,(pred_box,pred_label) in enumerate(zip(pred_boxes,pred_labels)):
                if len(detected_boxes) == len(annotations):
                    break
                if pred_label not in target_labels:
                    continue

                #IoUを計算
                iou,box_index = bbox_iou(pred_box.unsqueeze(0),target_boxes).max(0)
                if iou >= iou_thres and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives,pred_scores,pred_labels])

    return batch_metrics


def calc_evaluation_index(tp,pred_cls,target_cls,obj_conf):
    """各クラスごとに評価指標を計算する"""

    #objectness scoreでソートする
    i = np.argsort(-obj_conf)
    tp,obj_conf,pred_cls = tp[i],obj_conf[i],pred_cls[i]

    #クラスのset
    unique_classes = np.unique(target_cls)

    #Precision-Recall曲線を作って、APを計算
    ap,p,r = [],[],[]
    for c in unique_classes:
        i = True if pred_cls == c else False
        num_ground_truth = (target_cls==c).sum()
        num_predicted = i.sum()

        #何もなかったら何もしない
        if num_predicted == 0 and num_ground_truth == 0:
            continue

        #どちらかがなかったら0で補う
        if num_predicted == 0 or num_ground_truth == 0:
            ap.append(0)
            r.append(0)
            p.append(0)

        else:
            #FPとTP
            fpc = (1-tp[i]).cumsum()
            tpc = (pt[i]).cumsum()

            #Recall
            recall_curve = tpc / (num_ground_truth + 1e-16)
            r.append(recall_curve[-1])

            #Precition
            precition_curve = tpc / (tpc + fpc)
            p.append(precition_curve[-1])

            #AP
            mrec = np.concatenate(([0.0],recall_curve,[1.0]))
            mpre = np.concatenate(([0.0],precition_curve,[0.0]))
            for i in range(mpre.size-1,0,-1):
                mpre[i-1] = np.maximum(mpre[i-1],mpre[i])
            i = np.where(mrec[1:] != mrec[:-1])[0]
            ap_ = np.sum((mrec[i+1]-mrec[i])*mpre[i+1])
            ap.append(ap_)

        #F1
        p,r,ap = np.array(p),np.array(r),np.array(ap)
        f1 = 2 * p * r / (p + r + 1e16)

        return p,r,ap,f1,unique_classes.astype("int32")
