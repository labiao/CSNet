import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os
import os.path as osp

import dataloader
from misc import torchutils, imutils
import net.resnet50_cam
import cv2

cudnn.enabled = True


def _work(process_id, model, dataset, args):
    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)
    recam_predictor = net.resnet50_cam.Class_Predictor(10, 2048)
    recam_predictor.load_state_dict(
        torch.load(osp.join(args.recam_weight_dir, 'recam_predictor_' + str(args.recam_num_epoches) + '.pth')))

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()
        recam_predictor.cuda()
        for iter, pack in enumerate(data_loader):

            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']

            n_classes = torch.nonzero(label).squeeze(1)
            if n_classes[0] == 0:
                print('类别为0')
                continue

            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)

            outputs = [model.forward2(imgA[0].cuda(non_blocking=True), imgB[0].cuda(non_blocking=True),
                                      recam_predictor.classifier.weight) for imgA, imgB in
                       zip(pack['imgA'], pack['imgB'])]
            # outputs = [model.forward2(img[0].cuda(non_blocking=True),recam_predictor.classifier.weight) for img in pack['img']] # b x 20 x w x h
            # print(outputs.shape)
            # exit(-1)

            strided_cam = torch.sum(torch.stack(
                [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o in
                 outputs]), 0)

            highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                         mode='bilinear', align_corners=False) for o in outputs]
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]
            valid_cat = torch.nonzero(label)[:, 0]

            strided_cam = strided_cam[valid_cat]
            strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5

            # save cams
            np.save(os.path.join(args.recam_out_dir, img_name + '.npy'),
                    {"keys": valid_cat, "cam": strided_cam.cpu(), "high_res": highres_cam.cpu().numpy()})

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5 * iter + 1) // (len(databin) // 20)), end='')


def run(args):
    model = getattr(importlib.import_module(args.cam_network), 'CAM')()
    model.load_state_dict(
        torch.load(osp.join(args.recam_weight_dir, 'res50_recam_' + str(args.recam_num_epoches) + '.pth')))
    model.eval()

    n_gpus = torch.cuda.device_count()

    dataset = dataloader.VOC12ClassificationDatasetMSF(args.train_list, CAM_root=args.CAM_root,
                                                       scales=args.cam_scales)
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print(']')

    torch.cuda.empty_cache()
