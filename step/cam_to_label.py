import argparse

import cv2
import numpy as np
import os


def cam_to_label(args, sec=False):
    # 指定待处理的文件夹路径
    folder_path = args.ir_label_out_dir
    save_path = args.mask
    os.makedirs(save_path, exist_ok=True)
    # 循环读取文件夹下所有图片并处理
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.png') and not file_name.endswith('tte.png'):
            # 读取灰度影像
            img_gray = cv2.imread(os.path.join(folder_path, file_name), cv2.IMREAD_GRAYSCALE)
            # if np.sum(img_gray) == 0:
            #     continue
            # 进行标签值的映射和转换
            if not sec:
                img_gray[img_gray == 0] = 254
                img_gray[img_gray == 1] = 254
                img_gray[img_gray == 255] = 254
                img_gray[img_gray == 2] = 255
                img_gray[img_gray == 254] = 0
            else:
                img_gray[img_gray == 1] = 255

            # 保存影像
            save_file_name = f'{file_name}'
            save_path1 = os.path.join(save_path, save_file_name)
            cv2.imwrite(save_path1, img_gray)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # args.ir_label_out_dir = r'../result_our/DenseCRF_label'
    args.ir_label_out_dir = r'../result_our/amn_ir_label'
    # args.ir_label_out_dir = r'../result_RSP_sec/amn_ir_label_final'
    # args.mask = r'../result_our/mask'
    args.mask = r'../result_our/amn_mask'
    # args.mask = r'../result_RSP_sec/mask_final'
    cam_to_label(args, sec=True)
