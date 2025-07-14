from torch.utils.data import DataLoader
from misc import pyutils, imutils
from segment_anything import sam_model_registry
from chainercv.evaluations import calc_semantic_segmentation_confusion
from tqdm import tqdm
from misc.lossutils import *

import os
import sys
import cv2
from torch.backends import cudnn
import os.path as osp
import importlib

import dataloader

cudnn.enabled = True


def save_feature(train_list, CAM_root, save_dir):
    # save feature from resnet50 at 'result/sam_feature/'
    sam_checkpoint = './pth/sam_vit_h_4b8939.pth'
    device = "cuda"
    model_type = "vit_h"
    sys.path.append("..")
    sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam_model.to(device=device)

    dataset = dataloader.VOC12ClassificationDataset(train_list, CAM_root=CAM_root)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    tensor_feature = {}
    name2id = dict()
    print(len(data_loader))
    with torch.no_grad():
        for i, pack in tqdm(enumerate(data_loader)):
            imgA = pack['imgA'].cuda()
            imgB = pack['imgB'].cuda()
            x = (imgB - imgA)  # 16, 3, 224, 224
            sam_imput = F.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)
            sam_embedding = sam_model.image_encoder(sam_imput)
            sam_embedding = F.interpolate(x, size=(16, 16), mode='bilinear', align_corners=False)
            name2id[pack['name'][0]] = i  # 文件名到i的映射
            tensor_feature[i] = sam_imput[0].cpu()

    os.makedirs(save_dir, exist_ok=True)
    torch.save(tensor_feature, osp.join(save_dir, 'tensor_feature' + str(i) + '.pt'))
    # np.save(osp.join(save_dir, 'name2id.npy'), name2id)


def run(args):
    # model = getattr(importlib.import_module(args.amn_network), 'Net')()
    model = getattr(importlib.import_module(args.amn_network), 'Net')(
    backbone_file=f"/mnt/sdb1/zhengdaoyuan/PycharmProjects/AMN/weights/RepVGG-B1g2-train.pth")

    train_dataset = dataloader.VOC12SegmentationDataset(args.amn_list,
                                                        label_dir=args.SAMlabel,
                                                        CAM_root=args.CAM_root,
                                                        hor_flip=True,
                                                        crop_size=args.amn_crop_size,
                                                        crop_method="random",
                                                        rescale=(0.5, 1.5)
                                                        )

    train_data_loader = DataLoader(train_dataset, batch_size=args.amn_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    val_dataset = dataloader.VOC12SegmentationDataset(args.infer_list,
                                                      label_dir=args.ir_label_out_dir,
                                                      CAM_root=args.CAM_root,
                                                      crop_size=None,
                                                      crop_method="none",
                                                      )

    val_data_loader = DataLoader(val_dataset, batch_size=1,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    param_groups = model.trainable_parameters()

    optimizer = torch.optim.Adam(
        params=[
            {
                'params': param_groups[0],
                'lr': 5e-05,
                'weight_decay': 1.0e-4,
            },
            {
                'params': param_groups[1],
                'lr': 1e-03,
                'weight_decay': 1.0e-4,
            },
        ],
    )

    total_epochs = args.amn_num_epoches

    model = torch.nn.DataParallel(model).cuda()

    model.train()

    avg_meter = pyutils.AverageMeter()
    best_mIoU = 0.0
    for ep in range(total_epochs):
        loader_iter = iter(train_data_loader)

        pbar = tqdm(
            range(1, len(train_data_loader) + 1),
            total=len(train_data_loader),
            dynamic_ncols=True,
        )
        for iteration, _ in enumerate(pbar):
            optimizer.zero_grad()
            try:
                pack = next(loader_iter)
            except:
                loader_iter = iter(train_data_loader)
                pack = next(loader_iter)

            imgA = pack['imgA'].cuda(non_blocking=True)
            imgB = pack['imgB'].cuda(non_blocking=True)
            label_amn = pack['label'].long().cuda(non_blocking=True)
            label_cls = pack['label_cls'].cuda(non_blocking=True)

            # print(label_amn.shape)
            logit = model(imgA, imgB, label_cls)

            B, C, H, W = logit.shape
            label_amn = resize_labels(label_amn.cpu(), size=logit.shape[-2:]).cuda()

            # AMN原先是从irlabel读取的，我这里改成了samlabel
            label_amn[label_amn == 255] = 1
            label_amn[label_amn > 1] = 255

            label_ = label_amn.clone()
            label_[label_amn == 255] = 0

            given_labels = torch.full(size=(B, C, H, W), fill_value=args.eps / (C - 1)).cuda()
            # print(given_labels.shape, torch.unique(given_labels))
            # torch.Size([16, 2, 16, 16]) tensor([0.4000], device='cuda:0')
            given_labels.scatter_(dim=1, index=torch.unsqueeze(label_, dim=1), value=1 - args.eps)

            # print(given_labels.shape, torch.unique(given_labels))
            # torch.Size([16, 2, 16, 16]) tensor([0.4000, 0.6000], device='cuda:0')

            loss_pcl = balanced_cross_entropy(logit, label_amn, given_labels)

            loss = loss_pcl

            loss.backward()

            optimizer.step()

            avg_meter.add({'loss': loss.item()})

            pbar.set_description(f"[{ep + 1}/{total_epochs}] "
                                 f"PCL: [{avg_meter.pop('loss'):.4f}]")

        with torch.no_grad():
            model.eval()
            # dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)
            labels = []
            preds = []

            for i, pack in enumerate(tqdm(val_data_loader)):
                img_name = pack['name'][0]
                imgA = pack['imgA']
                imgB = pack['imgB']
                label_cls = pack['label_cls'][0]

                imgA = imgA.cuda()
                imgB = imgB.cuda()

                logit = model(imgA, imgB, pack['label_cls'].cuda())

                size = imgA.shape[-2:]
                strided_up_size = imutils.get_strided_up_size(size, 16)

                valid_cat = torch.nonzero(label_cls)[:, 0]
                keys = np.pad(valid_cat + 1, (1, 0), mode='constant')
                logit_up = F.interpolate(logit, strided_up_size, mode='bilinear', align_corners=False)
                logit_up = logit_up[0, :, :size[0], :size[1]]

                logit_up = F.softmax(logit_up, dim=0)[keys].cpu().numpy()

                cls_labels = np.argmax(logit_up, axis=0)
                cls_labels = keys[cls_labels]

                preds.append(cls_labels.copy())

                # gt_label = dataset.get_example_by_keys(i, (1,))[0]
                # print(os.path.join('/mnt/sdb1/zhengdaoyuan/PycharmProjects/AMN/Dataset/vai/SegmentationClassAug', img_name + '.png'))
                img = cv2.imread(
                    os.path.join('dataset/BCD/change_label', img_name + '.png'),
                    cv2.IMREAD_GRAYSCALE)

                gt_label = img // 255

                labels.append(gt_label.copy())

            confusion = calc_semantic_segmentation_confusion(preds, labels)

            gtj = confusion.sum(axis=1)
            resj = confusion.sum(axis=0)
            gtjresj = np.diag(confusion)
            denominator = gtj + resj - gtjresj
            iou = gtjresj / denominator

            print(f'[{ep + 1}/{total_epochs}] miou: {np.nanmean(iou):.4f}')

            if np.nanmean(iou) > best_mIoU:
                best_mIoU = np.nanmean(iou)
                torch.save(model.module.state_dict(), args.amn_weights_name + '.pth')
                print(best_mIoU)

            model.train()

    # torch.save(model.module.state_dict(), args.amn_weights_name + '.pth')
    torch.cuda.empty_cache()


# def run(args):
#     # save_feature(args.train_list, args.CAM_root, save_dir=os.path.join(args.work_space, 'sam_feature'))
#     run0(args)
