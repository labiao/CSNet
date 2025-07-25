import os
import numpy as np
import imageio

from torch import multiprocessing
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataloader
from misc import torchutils, imutils
from PIL import Image

palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128,
           64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128,
           0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128,
           64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0]


def _work(process_id, infer_dataset, args):
    visualize_intermediate_cam = False
    databin = infer_dataset[process_id]
    infer_data_loader = DataLoader(databin, shuffle=False, num_workers=0, pin_memory=False)

    for iter, pack in enumerate(infer_data_loader):
        # img_name = dataloader.decode_int_filename(pack['name'][0])
        img_name = pack['name'][0]
        imgA = pack['imgA'][0].numpy()
        imgB = pack['imgB'][0].numpy()
        str = os.path.join(args.cam_out_dir, img_name + '.npy')
        if not os.path.exists(str):
            imageio.imwrite(os.path.join(args.ir_label_out_dir, img_name + '.png'),
                            np.zeros((imgA.shape[:2])).astype(np.uint8))
            continue
        cam_dict = np.load(os.path.join(args.cam_out_dir, img_name + '.npy'), allow_pickle=True).item()
        # cams = cam_dict['high_res']
        cams = cam_dict['IS_CAM']
        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')

        # 1. find confident fg & bg
        fg_conf_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.conf_fg_thres)
        fg_conf_cam = np.argmax(fg_conf_cam, axis=0)

        pred = imutils.crf_inference_label(imgB - imgA, fg_conf_cam, n_labels=keys.shape[0])
        # print(fg_conf_cam)
        # exit(-1)

        fg_conf = keys[pred]
        bg_conf_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.conf_bg_thres)
        bg_conf_cam = np.argmax(bg_conf_cam, axis=0)
        pred = imutils.crf_inference_label(imgB - imgA, bg_conf_cam, n_labels=keys.shape[0])
        bg_conf = keys[pred]

        # 2. combine confident fg & bg
        conf = fg_conf.copy()
        conf[fg_conf == 0] = 255
        conf[bg_conf + fg_conf == 0] = 0

        # out = Image.fromarray(conf.astype(np.uint8), mode='P')
        # out.putpalette(palette)
        # out.save(os.path.join(args.ir_label_out_dir, img_name + '_palette.png'))

        imageio.imwrite(os.path.join(args.ir_label_out_dir, img_name + '.png'), conf.astype(np.uint8))

        if process_id == args.num_workers - 1 and iter % (len(databin) // 20) == 0:
            print("%d " % ((5 * iter + 1) // (len(databin) // 20)), end='')


def run(args):
    dataset = dataloader.VOC12ImageDataset(args.train_list, CAM_root=args.CAM_root, img_normal=None,
                                                 to_torch=False)
    dataset = torchutils.split_dataset(dataset, args.num_workers)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=args.num_workers, args=(dataset, args), join=True)
    print(']')
