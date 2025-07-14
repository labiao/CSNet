import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from .repvggb1g2 import create_RepVGG_B1g2


class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP)
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        for i, rate in enumerate(rates):
            self.add_module(
                "c{}".format(i),
                nn.Conv2d(in_ch, out_ch, 3, 1, padding=rate, dilation=rate, bias=True),
            )

        for m in self.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return sum([stage(x) for stage in self.children()])


class Net(nn.Module):

    def __init__(self, backbone_file, deploy=False, pretrained=True):
        super(Net, self).__init__()

        self.repvgg = create_RepVGG_B1g2(deploy)

        if pretrained:
            checkpoint = torch.load(backbone_file)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}  # strip the names
            self.repvgg.load_state_dict(ckpt)

        self.stage0, self.stage1, self.stage2, self.stage3, \
            self.stage4 = self.repvgg.stage0, self.repvgg.stage1, \
            self.repvgg.stage2, self.repvgg.stage3, self.repvgg.stage4

        for n, m in self.stage4.named_modules():
            if ('rbr_dense' in n or 'rbr_reparam' in n) and isinstance(m, nn.Conv2d):
                m.dilation, m.padding, m.stride = (1, 1), (1, 1), (1, 1)
                # print(m.dilation, m.padding, m.stride, m.kernel_size)  # (1, 1) (1, 1) (2, 2)
                print('change dilation, padding, stride of ', n)
            elif 'rbr_1x1' in n and isinstance(m, nn.Conv2d):
                m.stride = (1, 1)
                print('change stride of ', n)

        self.conv = nn.Conv2d(2048, 2048, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(2048)
        self.relu = nn.ReLU(inplace=True)
        self.stage5 = nn.Sequential(self.conv, self.bn, self.relu)

        astrous_rates = [6, 12, 18, 24]

        self.label_enc = nn.Linear(1, 2048)

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            _ASPP(in_ch=2048, out_ch=2, rates=astrous_rates)
        )
        self.backbone = nn.ModuleList([self.stage0, self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.stage5, self.classifier, self.label_enc])

    def forward(self, imgA, imgB, label_cls):
        # y相当于权重作用于深层次特征了
        y = self.label_enc(label_cls).unsqueeze(-1).unsqueeze(-1)

        x1 = self.stage0(imgA)
        x2 = self.stage0(imgB)
        x1 = self.stage1(x1)
        x2 = self.stage1(x2)
        x1 = self.stage2(x1)
        x2 = self.stage2(x2)
        x1 = self.stage3(x1)
        x2 = self.stage3(x2)
        x1 = self.stage4(x1)
        x2 = self.stage4(x2)
        # torch.Size([16, 64, 128, 128])
        # torch.Size([16, 128, 64, 64])
        # torch.Size([16, 256, 32, 32])
        # torch.Size([16, 512, 16, 16])
        # torch.Size([16, 2048, 16, 16])

        # feature differencing
        x = torch.abs(x1 - x2)
        x = self.stage5(x)

        x = x * y
        # sa_output = None
        logit = self.classifier(x)

        return logit

    # def train(self, mode=True):
    #     for p in self.resnet50.conv1.parameters():
    #         p.requires_grad = False
    #     for p in self.resnet50.bn1.parameters():
    #         p.requires_grad = False

    def trainable_parameters(self):

        return list(self.backbone.parameters()), list(self.newly_added.parameters())


class CAM(Net):

    def __init__(self):
        super(CAM, self).__init__()

    def forward(self, x, y):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        B, C, H, W = x.shape

        y = self.label_enc(y)
        y = y.unsqueeze(-1).unsqueeze(-1)
        x = x * y

        logit = self.classifier(x)

        logit = (logit[0] + logit[1].flip(-1)) / 2

        return logit


class CAM2(Net):

    def __init__(self, backbone_file, deploy=False, pretrained=True):
        super(CAM2, self).__init__(backbone_file, deploy=deploy, pretrained=pretrained)

    def forward(self, imgA, imgB, label_cls):

        y = self.label_enc(label_cls).unsqueeze(-1).unsqueeze(-1)
        x1 = self.stage0(imgA)
        x2 = self.stage0(imgB)
        x1 = self.stage1(x1)
        x2 = self.stage1(x2)
        x1 = self.stage2(x1)
        x2 = self.stage2(x2)
        x1 = self.stage3(x1)
        x2 = self.stage3(x2)
        x1 = self.stage4(x1)
        x2 = self.stage4(x2)

        # feature differencing
        x = torch.abs(x1 - x2)
        x = self.stage5(x)
        x = x * y

        # logit = self.pam(x)
        logit = self.classifier(x)

        logit = (logit[0] + logit[1].flip(-1)) / 2

        return logit
