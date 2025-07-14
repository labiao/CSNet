import os

import cv2
import numpy as np

import random


def split_dataset(data, test_ratio):
    """
    将数据集划分为训练集和测试集
    :param data: 待划分的数据集
    :param test_ratio: 测试集的比例
    :return: 训练集和测试集
    """
    n = len(data)
    indices = list(range(n))
    random.shuffle(indices)  # 随机打乱索引顺序

    test_set_size = int(n * test_ratio)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]

    test_data = [data[i] for i in test_indices]
    train_data = [data[i] for i in train_indices]

    return train_data, test_data


def count_building(label, npy_path, txt1_path, txt2_path):
    # 1.读取标签路径下的掩膜文件
    for _, _, filenames in os.walk(label):
        # 2.循环遍历并统计每个png文件中属于mask = 1的数量
        count1 = count2 = 0
        ll_change = []
        ll_no_change = []
        e = dict()
        for i in range(0, len(filenames)):
            png = os.path.join(label, filenames[i])  # png
            img = cv2.imread(png, cv2.IMREAD_GRAYSCALE)

            if img.shape[0] == 256 and img.shape[1] == 256:
                # print(img.shape)
                No_building = np.sum(img == 0)  # 统计背景像素
                building = np.sum(img != 0)  # 统计建筑物像素
                # 如果当前的 building > Nobuilding，则将该图放置1文件夹中
                if building >= 256 * 256 * 0.05:
                    # if building:
                    # shutil.copyfile(png, os.path.join(save_label, filenames[i]))
                    # f.writelines(filenames[i].split(".")[0] + " 1\n")
                    ll_change.append(filenames[i].split(".")[0] + "\n")
                    e[filenames[i].split('.')[0]] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
                    count1 += 1
                # 否则，则将该图放置0文件夹中
                elif building == 0:
                    # shutil.copyfile(png, os.path.join(save_label, filenames[i]))
                    # f.writelines(filenames[i].split(".")[0] + " 0\n")
                    ll_no_change.append(filenames[i].split(".")[0] + "\n")
                    e[filenames[i].split('.')[0]] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    count2 += 1
        f1 = open(txt1_path, "w+")
        f2 = open(txt2_path, "w+")
        change_train, change_test = split_dataset(ll_change, 0.3)
        no_change_train, no_change_test = split_dataset(ll_no_change, 0.3)

        for data in change_train:
            f1.writelines(data)

        for data in change_test:
            f2.writelines(data)

        for data in no_change_train:
            f1.writelines(data)

        for data in no_change_test:
            f2.writelines(data)

        f1.close()
        f2.close()

        np.save(npy_path, e)
        print(count1, count2)


if __name__ == '__main__':
    label = '../dataset/BCD/change_label'
    npy_path = '../dataset/BCD/npy5.npy'
    txt1_path = '../dataset/BCD/train5.txt'
    txt2_path = '../dataset/BCD/val5.txt'
    # 1917 9009 > 0.05
    # 3075 9009 > 0
    count_building(label, npy_path, txt1_path, txt2_path)
