import os

import cv2
import numpy as np

def visual_refined_unary(refined_unary, name):
    refined_unary = np.reshape(refined_unary, (256, 256))
    fg_img = np.uint8(255 * refined_unary)
    vis_result = cv2.applyColorMap(fg_img, cv2.COLORMAP_JET)
    img = cv2.imread(os.path.join('../dataset/BCD/change_label', name + '.png'))
    fuse = cv2.addWeighted(img, 0.5, vis_result, 0.5, 0)
    cv2.imwrite('../result_RSP/visual_cam/' + name + '.jpg', fuse)
    # cv2.imwrite('../result_RSP_sec/visual_amn_cam/' + name + '.jpg', fuse)


def run():
    ids = []
    # ---------------------------------------------------------------------------------------
    root = r'../dataset/BCD/change_label'
    with open(r'../dataset/BCD/train5.txt', 'r') as f:
        file = f.readlines()
        for i in range(0, len(file)):
            file[i] = file[i].rstrip('\n')
            img = cv2.imread(os.path.join(root, file[i] + '.png'), cv2.IMREAD_GRAYSCALE)
            if (img == 0).all():
                continue
            ids.append(file[i])
    # ---------------------------------------------------------------------------------------
    print(len(ids))
    n_images = 0
    for i, id in enumerate(ids):
        n_images += 1
        # cam_dict = np.load(os.path.join(r'/mnt/sdb1/zhengdaoyuan/PycharmProjects/AMN/result/lpcam_mask', id + '.npy'),
        #                    allow_pickle=True).item()
        # cams = cam_dict['high_res']  # high_res指的是high_resolution
        # if id == 'top_mosaic_09cm_area32_4_9':
        # cam_dict = np.load(os.path.join(r'../result_RSP/cam', id + '.npy'), allow_pickle=True).item()
        cam_dict = np.load(os.path.join(r'../result_RSP/amn_cam', id + '.npy'), allow_pickle=True).item()
        # print(cam_dict['high_res'].shape)  # (2, 256, 256)
        cams = cam_dict['high_res'][1:, ...]
        # cams = cam_dict['high_res']
        # cams = cam_dict['cam'][1:, ...].numpy()

        visual_refined_unary(cams, id)

if __name__ == '__main__':

    run()
