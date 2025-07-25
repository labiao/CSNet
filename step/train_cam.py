import torch
import torch.nn as nn
from torch.backends import cudnn

cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F

import importlib

import dataloader
from misc import pyutils, torchutils


def validate(model, data_loader):
    print('validating ... ', flush=True, end='')

    val_loss_meter = pyutils.AverageMeter('loss1', 'loss2')

    model.eval()
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for pack in data_loader:
            imgA = pack['imgA']
            imgB = pack['imgB']

            label = pack['label'].cuda(non_blocking=True)
            # print(img.shape)
            # exit(-1)
            x = model(imgA, imgB)
            loss = F.multilabel_soft_margin_loss(x, label)

            val_loss_meter.add({'loss': loss.item()})

    model.train()

    print('loss: %.4f' % (val_loss_meter.pop('loss')))

    return


def run(args):
    model = getattr(importlib.import_module(args.cam_network), 'Net')()

    print('111')
    train_dataset = dataloader.VOC12ClassificationDataset(args.train_list, CAM_root=args.CAM_root,
                                                          hor_flip=True,
                                                          crop_size=224, crop_method="random")
    train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    max_step = (len(train_dataset) // args.cam_batch_size) * args.cam_num_epoches

    val_dataset = dataloader.VOC12ClassificationDataset(args.val_list, CAM_root=args.CAM_root,
                                                        crop_size=224)
    val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[1], 'lr': 10 * args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
    ], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)

    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()

    for ep in range(args.cam_num_epoches):

        print('Epoch %d/%d' % (ep + 1, args.cam_num_epoches))

        for step, pack in enumerate(train_data_loader):

            imgA = pack['imgA']
            imgB = pack['imgB']
            imgA = imgA.cuda()
            imgB = imgB.cuda()
            label = pack['label'].cuda(non_blocking=True)
            x = model(imgA, imgB)

            optimizer.zero_grad()

            loss = F.multilabel_soft_margin_loss(x, label)

            loss.backward()
            avg_meter.add({'loss': loss.item()})

            optimizer.step()
            if (optimizer.global_step - 1) % 100 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f' % (avg_meter.pop('loss')),
                      'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)

        validate(model, val_data_loader)
        timer.reset_stage()

    torch.save(model.module.state_dict(), args.cam_weights_name + '.pth')
    torch.cuda.empty_cache()
