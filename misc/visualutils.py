import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np


def visual_attention_map(att_maps, name):
    # print(att_maps.shape) # 2 256 256

    refined_unary = np.reshape(att_maps[1, ...], (256, 256))
    fg_img = np.uint8(255 * refined_unary)
    vis_result = cv2.applyColorMap(fg_img, cv2.COLORMAP_JET)
    # img = cv2.imread(
    #     os.path.join('/mnt/sdb1/zhengdaoyuan/PycharmProjects/CS-WSCDNet/dataset/BCD/change_label', name + '.png'))
    # fuse = cv2.addWeighted(img, 0.5, vis_result, 0.5, 0)
    # cv2.imwrite('/mnt/sdb1/zhengdaoyuan/PycharmProjects/CS-WSCDNet/result_RSP/visual_cam/' + name + '.jpg', fuse)
    cv2.imwrite('/mnt/sdb1/zhengdaoyuan/PycharmProjects/CS-WSCDNet/result_RSP_spie/' + name + '.jpg', vis_result)


def visual_seeds(belongings):  # belongings:[n,21,h,w]
    fg_img = np.uint8(255 * belongings[0])
    vis_result = cv2.applyColorMap(fg_img, cv2.COLORMAP_JET)
    cv2.imwrite('/mnt/sdb1/zhengdaoyuan/PycharmProjects/CS-WSCDNet/exp/belongings/' + 'name' + '.jpg', vis_result)


def visual_norm_cam(norm_cam):
    bg_img = np.uint8(255 * norm_cam[0][0])
    fg_img = np.uint8(255 * norm_cam[0][1])
    bg_img = cv2.resize(bg_img, (256, 256), interpolation=cv2.INTER_NEAREST)
    fg_img = cv2.resize(fg_img, (256, 256), interpolation=cv2.INTER_NEAREST)
    vis_result1 = cv2.applyColorMap(bg_img, cv2.COLORMAP_JET)
    vis_result2 = cv2.applyColorMap(fg_img, cv2.COLORMAP_JET)

    cv2.imwrite('/mnt/sdb1/zhengdaoyuan/PycharmProjects/CS-WSCDNet/exp/belongings/' + 'bg' + '.jpg', vis_result1)
    cv2.imwrite('/mnt/sdb1/zhengdaoyuan/PycharmProjects/CS-WSCDNet/exp/belongings/' + 'fg' + '.jpg', vis_result2)
