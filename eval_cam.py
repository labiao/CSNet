import argparse
import os

import numpy as np
from PIL import Image
from chainercv.evaluations import calc_semantic_segmentation_confusion
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataloader
from misc import pyutils


def run(args, predict_dir, num_cls, flag):
    preds = []
    masks = []
    n_images = 0
    for iter, pack in tqdm(enumerate(dataloader)):
        n_images += 1
        cam_dict = np.load(os.path.join(predict_dir, pack['name'][0] + '.npy'), allow_pickle=True).item()
        if flag == 0:
            cams = cam_dict['IS_CAM']
            keys = cam_dict['keys']
        elif flag == 1:
            cams = cam_dict['IS_CAM']
            cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
            keys = cam_dict['keys']
            keys = np.pad(keys, (1, 0), mode='constant')
        else:
            cams = cam_dict['IS_CAM']
            cams = np.pad(cams[1:, ...], ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
            keys = cam_dict['keys']

        cls_labels = np.argmax(cams, axis=0)
        cls_labels = keys[cls_labels]
        preds.append(cls_labels.copy())

        mask = np.array(Image.open(os.path.join(args.gt_path, pack['name'][0] + '.png'))) // 255
        masks.append(mask.copy())

    confusion = calc_semantic_segmentation_confusion(preds, masks)[:num_cls, :num_cls]

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator
    print({'th:': args.cam_eval_thres, 'iou': iou, 'miou': np.nanmean(iou)})


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", default="dataset/BCD/amn_train_change.txt", type=str)
    parser.add_argument("--gt_path", default='dataset/BCD/change_label', type=str)
    parser.add_argument('--session_name', default="exp", type=str)
    args = parser.parse_args()

    CAM_root = 'dataset/BCD'
    num_cls = 2
    dataset = dataloader.VOC12ImageDataset(args.train_list,
                                           CAM_root=CAM_root,
                                         img_normal=None, to_torch=False)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    pyutils.Logger(os.path.join(args.session_name, 'eval_' + args.session_name + '.log'))
    flag = 2
    if flag == 0:
        print('不使用阈值(Argmax)')
    elif flag == 1:
        print('结合背景和前景类别的cam去预测')
    else:
        print('只用前景类别的cam去预测')
    for i in range(26, 27):
        t = i / 100.0
        args.cam_eval_thres = t
        run(args, args.session_name + "/npy", num_cls, flag)
