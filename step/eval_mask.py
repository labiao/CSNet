import os

import cv2
import imageio
# matplotlib.use('agg')
# import matplotlib.pyplot as plt  # matplotlib.use('agg')必须在本句执行前运行
import numpy as np
from chainercv.evaluations import calc_semantic_segmentation_confusion


def run(args):
    preds = []
    labels = []
    ids = []
    n_img = 0
    root = r'dataset/BCD/change_label'
    with open(r'dataset/BCD/amn_train_change.txt', 'r') as f:
        file = f.readlines()
        for i in range(0, len(file)):
            file[i] = file[i].rstrip('\n')
            img = cv2.imread(os.path.join(root, file[i] + '.png'), cv2.IMREAD_GRAYSCALE)
            ids.append(file[i])
            new_img = img.copy()
            new_img = new_img // 255
            new_img = new_img.astype(int)
            # print(new_img.dtype)
            # new_img.dtype = 'int64' 报错
            labels.append(new_img)
    for i, id in enumerate(ids):
        cls_labels = imageio.imread(os.path.join(args.mask, id + '.png')).astype(np.uint8)
        cls_labels[cls_labels == 255] = 1
        preds.append(cls_labels.copy())
        n_img += 1

    confusion = calc_semantic_segmentation_confusion(preds, labels)[:2, :2]
    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    fp = 1. - gtj / denominator
    fn = 1. - resj / denominator
    iou = gtjresj / denominator
    print("total images", n_img)

    precision = gtjresj / (fp * denominator + gtjresj)
    recall = gtjresj / (fn * denominator + gtjresj)
    F_score = 2 * (precision * recall) / (precision + recall)
    print({'precision': precision, 'recall': recall, 'F_score': F_score})
    print({'iou': iou, 'miou': np.nanmean(iou)})
