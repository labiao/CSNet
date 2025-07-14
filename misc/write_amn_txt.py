import os
import shutil

import numpy as np


def write_txt(save_path1, save_path2):
    SAMlabel_path = r'/mnt/sdb3/zhengdaoyuan/PycharmProjects/CS-WSCDNet/result_our/SAMlabelV1'
    ori_txt = r'/mnt/sdb3/zhengdaoyuan/PycharmProjects/CS-WSCDNet/dataset/BCD/train5.txt'
    cls_labels_dict = np.load('/mnt/sdb3/zhengdaoyuan/PycharmProjects/CS-WSCDNet/dataset/BCD/npy5.npy',
                              allow_pickle=True).item()
    f1 = open(save_path1, "w+")
    f2 = open(save_path2, "w+")
    img_name_list = np.loadtxt(ori_txt, dtype=np.str_)
    for img_name in img_name_list:
        if cls_labels_dict[img_name] == [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]:
            if os.path.exists(os.path.join(SAMlabel_path, img_name + '.png')):
                f1.writelines(img_name + '\n')
                f2.writelines(img_name + '\n')
        elif cls_labels_dict[img_name] == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
            f1.writelines(img_name + '\n')
            shutil.copyfile('../result_our/DenseCRF_label/' + img_name + '.png', os.path.join(SAMlabel_path, img_name + '.png'))

    f1.close()
    f2.close()


def write_txt_sec(save_path1):
    SAMlabel_path = r'/mnt/sdb1/zhengdaoyuan/PycharmProjects/CS-WSCDNet/result_RSP_sec/SAMlabelV1'
    ori_txt = r'/mnt/sdb1/zhengdaoyuan/PycharmProjects/CS-WSCDNet/dataset/BCD/train5.txt'
    cls_labels_dict = np.load('/mnt/sdb1/zhengdaoyuan/PycharmProjects/CS-WSCDNet/dataset/BCD/npy5.npy',
                              allow_pickle=True).item()
    f1 = open(save_path1, "w+")
    img_name_list = np.loadtxt(ori_txt, dtype=np.str_)
    for img_name in img_name_list:
        if cls_labels_dict[img_name] == [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]:
            if os.path.exists(os.path.join(SAMlabel_path, img_name + '.png')):
                f1.writelines(img_name + '\n')
        elif cls_labels_dict[img_name] == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
            f1.writelines(img_name + '\n')
            shutil.copyfile('../result_RSP/DenseCRF_label/' + img_name + '.png', os.path.join(SAMlabel_path, img_name + '.png'))

    f1.close()


if __name__ == '__main__':
    write_txt(r'../dataset/BCD/amn_train.txt', r'../dataset/BCD/amn_train_change.txt')
    # write_txt_sec(r'../dataset/BCD/amn_train_sec.txt')
