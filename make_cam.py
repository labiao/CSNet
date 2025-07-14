import argparse
import importlib
import os
import sys
from builtins import bool

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import multiprocessing, cuda
from torch.backends import cudnn
from torch.utils.data import DataLoader

import dataloader
from misc import torchutils, pyutils

cudnn.enabled = True


def overlap(img, hm):
    hm = plt.cm.jet(hm)[:, :, :3]
    hm = np.array(
        Image.fromarray((hm * 255).astype(np.uint8), 'RGB').resize((img.shape[1], img.shape[0]), Image.BICUBIC)).astype(
        np.float) * 2
    if hm.shape == np.array(img).astype(np.float).shape:
        out = (hm + np.array(img).astype(np.float)) / 3
        out = (out / np.max(out) * 255).astype(np.uint8)
    else:
        print(hm.shape)
        print(np.array(img).shape)
    return out


def draw_heatmap(norm_cam, gt_label, orig_img, save_path, img_name):
    gt_cat = np.where(gt_label[[0, 2]] == 1)[0]
    # for _, gt in enumerate(gt_cat):
    gt = gt_cat[1]
    heatmap = overlap(orig_img, norm_cam[gt])
    cam_viz_path = os.path.join(save_path, img_name + '_{}.png'.format(gt))
    imageio.imsave(cam_viz_path, heatmap)


def _work(process_id, model, dataset, args):
    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()

        for iter, pack in enumerate(data_loader):
            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']
            label = F.pad(label, (1, 0), 'constant', 1.0)

            outputs = [model(imgA[0].cuda(non_blocking=True), imgB[0].cuda(non_blocking=True),
                             label.cuda(non_blocking=True).unsqueeze(-1).unsqueeze(-1))
                       for imgA, imgB in zip(pack['imgA'], pack['imgB'])]  # 长度为4的列表

            # multi-scale fusion
            IS_CAM_list = [output[1].cpu() for output in outputs]
            IS_CAM_list = [F.interpolate(torch.unsqueeze(o, 1), size, mode='bilinear', align_corners=False) for o in
                           IS_CAM_list]
            IS_CAM = torch.sum(torch.stack(IS_CAM_list, 0), 0)[:, 0]
            IS_CAM /= F.adaptive_max_pool2d(IS_CAM, (1, 1)) + 1e-5
            IS_CAM = IS_CAM.cpu().numpy()

            # visualize IS-CAM
            if args.visualize:
                orig_img = np.array(Image.open('dataset/BCD/change_label/' + img_name + '.png').convert('RGB'))
                # orig_img = np.array(Image.open(pack['img_path'][0]).convert('RGB'))
                draw_heatmap(IS_CAM.copy(), label, orig_img, os.path.join(args.session_name, 'visual'), img_name)

            # save IS_CAM
            valid_cat = torch.nonzero(label)[:, 0].cpu().numpy()
            valid_cat[valid_cat == 2] = 1
            IS_CAM = IS_CAM[valid_cat]
            np.save(os.path.join(args.session_name, 'npy', img_name + '.npy'), {"keys": valid_cat, "IS_CAM": IS_CAM})

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5 * iter + 1) // (len(databin) // 20)), end='')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", default="dataset/BCD/amn_train_change.txt", type=str)
    parser.add_argument("--network", default="net.repvggb1g2_SIPE", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--session_name", default="exp_rep", type=str)
    parser.add_argument("--ckpt", default="final.pth", type=str)
    parser.add_argument("--visualize", default=True, type=bool)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.session_name, 'npy'), exist_ok=True)
    os.makedirs(os.path.join(args.session_name, 'visual'), exist_ok=True)
    pyutils.Logger(os.path.join(args.session_name, 'infer.log'))
    print(vars(args))

    CAM_root = 'dataset/BCD'
    dataset = dataloader.VOC12ClassificationDatasetMSF(
        args.train_list,
        CAM_root=CAM_root,
        scales=(1.0, 0.5, 1.5, 2.0))
    checkpoint = torch.load(args.session_name + '/ckpt/' + args.ckpt)
    # model = getattr(importlib.import_module(args.network), 'CAM')()
    # model.load_state_dict(checkpoint['net'], strict=True)
    # model.eval()

    ############################deployment#####################
    # model = getattr(importlib.import_module(args.network), 'CAM')(None, deploy=False, pretrained=False)
    # model.eval()
    # model.load_state_dict(checkpoint['net'])
    # for module in model.modules():
    #     if hasattr(module, 'switch_to_deploy'):
    #         module.switch_to_deploy()
    # torch.save(model.state_dict(), args.session_name + '/ckpt/' + args.ckpt.split('.')[0] +'_deploy.pth')
    # del model
    # sys.exit(0)

    model = getattr(importlib.import_module(args.network), 'CAM')(None, deploy=True, pretrained=False)
    model.load_state_dict(torch.load(args.session_name + '/ckpt/' + args.ckpt.split('.')[0] + '_deploy2.pth'))
    model.eval()
    #####################################################################

    n_gpus = torch.cuda.device_count()

    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print(']')

    torch.cuda.empty_cache()
