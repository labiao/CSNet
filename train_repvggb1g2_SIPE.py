from torch.backends import cudnn
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from chainercv.evaluations import calc_semantic_segmentation_confusion
from misc import pyutils, torchutils, visualization
import argparse
import importlib
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import dataloader

cudnn.enabled = True


def validate(model, data_loader):
    print('validating ... ', flush=True, end='')

    model.eval()

    with torch.no_grad():
        preds = []
        preds_cam = []
        labels = []
        for iter, pack in tqdm(enumerate(data_loader)):
            imgA = pack['imgA'].cuda()
            imgB = pack['imgB'].cuda()
            img_name = pack['name'][0]
            label = pack['label'].cuda(non_blocking=True).unsqueeze(-1).unsqueeze(-1)
            label = F.pad(label, (0, 0, 0, 0, 1, 0), 'constant', 1.0)

            outputs = model.forward(imgA, imgB, label)
            IS_cam = outputs['IS_cam']
            cam = outputs['cam']
            IS_cam = F.interpolate(IS_cam, imgA.shape[2:], mode='bilinear')
            cam = F.interpolate(cam, imgA.shape[2:], mode='bilinear')
            IS_cam = IS_cam / (F.adaptive_max_pool2d(IS_cam, (1, 1)) + 1e-5)
            cam = cam / (F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5)
            cls_labels_bkg = torch.argmax(IS_cam, 1)
            cls_labels_bkg_cam = torch.argmax(cam, 1)
            # append preds
            # assert not torch.eq(cls_labels_bkg, 1).any() 
            # cls_labels_bkg[cls_labels_bkg < 1] = 0
            # cls_labels_bkg[cls_labels_bkg == 2] = 1
            preds.append(cls_labels_bkg[0].cpu().numpy().copy())
            preds_cam.append(cls_labels_bkg_cam[0].cpu().numpy().copy())
            # append labels
            img = cv2.imread(os.path.join('./dataset/BCD/change_label', img_name + '.png'), cv2.IMREAD_GRAYSCALE)
            gt_label = (img // 255)
            labels.append(gt_label.copy())

        confusion = calc_semantic_segmentation_confusion(preds, labels)
        confusion_cam = calc_semantic_segmentation_confusion(preds_cam, labels)
        gtj = confusion.sum(axis=1)
        gtj_cam = confusion_cam.sum(axis=1)
        resj = confusion.sum(axis=0)
        resj_cam = confusion_cam.sum(axis=0)
        gtjresj = np.diag(confusion)
        gtjresj_cam = np.diag(confusion_cam)
        denominator = gtj + resj - gtjresj
        denominator_cam = gtj_cam + resj_cam - gtjresj_cam
        iou = gtjresj / denominator
        iou_cam = gtjresj_cam / denominator_cam
        print({'iou': iou, 'miou': np.nanmean(iou)})
        print({'iou_cam': iou_cam, 'miou_cam': np.nanmean(iou_cam)})

    model.train()

    return np.nanmean(iou)


def setup_seed(seed):
    print("random seed is set to", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--CAM_root", default='dataset/BCD', type=str,
                        help="Dataset floder. Please enter the folder names for T1,T2 images in the IMG_FOLDER_NAME_A "
                             "and IMG_FOLDER_NAME_B sections of the dataloader")
    parser.add_argument("--SAM_A", default="./dataset/BCD/train/A1/", type=str)
    parser.add_argument("--SAM_B", default="./dataset/BCD/train/B1/", type=str,
                        help="Remove the unchanged pixel pairs in the predicted mask, and only use SAM for the "
                             "changed pixel pairs")

    # Dataset
    parser.add_argument("--train_list", default="dataset/BCD/train5.txt", type=str)
    parser.add_argument("--val_list", default="dataset/BCD/amn_train_change.txt", type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_epoches", default=10, type=int)
    parser.add_argument("--network", default="net.repvggb1g2_SIPE", type=str)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=1e-4, type=float)
    parser.add_argument("--session_name", default="exp_rep", type=str)
    parser.add_argument("--print_freq", default=100, type=int)
    parser.add_argument("--tf_freq", default=500, type=int)
    parser.add_argument("--val_freq", default=470, type=int)
    parser.add_argument("--seed", default=15, type=int)
    args = parser.parse_args()

    setup_seed(args.seed)
    os.makedirs(args.session_name, exist_ok=True)
    os.makedirs(os.path.join(args.session_name, 'runs'), exist_ok=True)
    os.makedirs(os.path.join(args.session_name, 'ckpt'), exist_ok=True)
    pyutils.Logger(os.path.join(args.session_name, args.session_name + '.log'))
    tblogger = SummaryWriter(os.path.join(args.session_name, 'runs'))

    model = getattr(importlib.import_module(args.network), 'Net')(backbone_file=f'pth/RepVGG-B1g2-train.pth')
    train_dataset = dataloader.VOC12ClassificationDataset(args.train_list, CAM_root=args.CAM_root,
                                                          resize_long=(190, 320), hor_flip=True,
                                                          crop_size=256, crop_method="random")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    max_step = (len(train_dataset) // args.batch_size) * args.max_epoches

    val_dataset = dataloader.VOC12ClassificationDataset(args.val_list, CAM_root=args.CAM_root, crop_size=256)
    val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                 pin_memory=True, drop_last=True)

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 10 * args.lr, 'weight_decay': args.wt_dec}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter()
    timer = pyutils.Timer()

    bestiou = 0
    for ep in range(args.max_epoches):

        print('Epoch %d/%d' % (ep + 1, args.max_epoches))

        for step, pack in enumerate(train_data_loader):
            imgA = pack['imgA']
            imgB = pack['imgB']
            imgA = imgA.cuda()
            imgB = imgB.cuda()
            n, c, h, w = imgA.shape
            label = pack['label'].cuda(non_blocking=True)

            valid_mask = pack['valid_mask'].cuda()
            valid_mask[:, 1:] = valid_mask[:, 1:] * label.unsqueeze(-1).unsqueeze(-1)
            valid_mask_lowres = F.interpolate(valid_mask, size=(h // 16, w // 16), mode='nearest')

            outputs = model.forward(imgA, imgB, valid_mask_lowres)
            score = outputs['score']
            norm_cam = outputs['cam']
            IS_cam = outputs['IS_cam']
            IS_cam = IS_cam / (F.adaptive_max_pool2d(IS_cam, (1, 1)) + 1e-5)

            
            lossCLS = F.multilabel_soft_margin_loss(score.squeeze(-1).squeeze(-1), label)
            # loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            # lossCLS = nn.BCEWithLogitsLoss()(score.squeeze(-1).squeeze(-1), label)
            lossGSC = torch.mean(torch.abs(norm_cam[:, 1, :, :] - IS_cam[:, 1, :, :]))
            losses = lossCLS + 1 * lossGSC
            # losses = lossCLS
            avg_meter.add({'lossCLS': lossCLS.item(), 'lossGSC': lossGSC.item()})
            # avg_meter.add({'lossCLS': lossCLS.item()})

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            if (optimizer.global_step - 1) % args.print_freq == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'lossCLS:%.4f' % (avg_meter.pop('lossCLS')),
                      'lossGSC:%.4f' % (avg_meter.pop('lossGSC')),
                      'imps:%.1f' % ((step + 1) * args.batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)

                # tf record
                tblogger.add_scalar('lossCLS', lossCLS, optimizer.global_step)
                # tblogger.add_scalar('lossGSC', lossGSC, optimizer.global_step)
                tblogger.add_scalar('lr', optimizer.param_groups[0]['lr'], optimizer.global_step)

            if (optimizer.global_step - 1) % args.tf_freq == 0:
                # visualization
                img_8 = visualization.convert_to_tf(imgB[0])
                norm_cam = F.interpolate(norm_cam, img_8.shape[1:], mode='bilinear')[0].detach().cpu().numpy()
                IS_cam = F.interpolate(IS_cam, img_8.shape[1:], mode='bilinear')[0].detach().cpu().numpy()
                CAM = visualization.generate_vis(norm_cam, None, img_8,
                                                 func_label2color=visualization.VOClabel2colormap, threshold=None,
                                                 norm=False)
                IS_CAM = visualization.generate_vis(IS_cam, None, img_8,
                                                    func_label2color=visualization.VOClabel2colormap, threshold=None,
                                                    norm=False)

                # tf record
                tblogger.add_images('CAM', CAM, optimizer.global_step)
                tblogger.add_images('IS_CAM', IS_CAM, optimizer.global_step)

            if (optimizer.global_step - 1) % args.val_freq == 0 and optimizer.global_step > 10:
                miou = validate(model, val_data_loader)
                # torch.save({'net': model.module.state_dict()},
                #            os.path.join(args.session_name, 'ckpt', 'iter_' + str(optimizer.global_step) + '.pth'))
                if miou > bestiou:
                    bestiou = miou
                    torch.save({'net': model.module.state_dict()}, os.path.join(args.session_name, 'ckpt', 'best.pth'))

        else:
            timer.reset_stage()

    torch.save({'net': model.module.state_dict()}, os.path.join(args.session_name, 'ckpt', 'final.pth'))
    torch.cuda.empty_cache()


if __name__ == '__main__':
    train()
