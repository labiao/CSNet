import torch
import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils

from repvggb1g2 import create_RepVGG_B1g2


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
                # print('change dilation, padding, stride of ', n)
            elif 'rbr_1x1' in n and isinstance(m, nn.Conv2d):
                m.stride = (1, 1)
                print('change stride of ', n)

        self.classifier = nn.Conv2d(2048, 1, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage0, self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])

    def forward(self, x):

        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        # print(x.shape) # [16, 256, 32, 32]
        x = self.stage3(x)
        # print(x.shape) # [16, 512, 16, 16]
        x = self.stage4(x)
        # print(x.shape) # [16, 2048, 16, 16]
        x = torchutils.gap2d(x, keepdims=True)
        x = self.classifier(x)
        x = x.view(-1, 1)

        return x

    def trainable_parameters(self):

        return list(self.backbone.parameters()), list(self.newly_added.parameters())


class Net_CAM(Net):

    def __init__(self, backbone_file, deploy=False, pretrained=True):
        super(Net_CAM, self).__init__(backbone_file, deploy=deploy, pretrained=pretrained)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)

        x = self.stage3(x)
        feature = self.stage4(x)

        x = torchutils.gap2d(feature, keepdims=True)
        x = self.classifier(x)
        x = x.view(-1, 1)  # 1个类别

        cams = F.conv2d(feature, self.classifier.weight)
        cams = F.relu(cams)

        return x, cams, feature


class CAM(Net):

    def __init__(self):
        super(CAM, self).__init__()

    def forward(self, x):
        x = self.stage1(x)

        x = self.stage2(x)

        x = self.stage3(x)

        x = self.stage4(x)

        x = F.conv2d(x, self.classifier.weight)
        x = F.relu(x)

        x = x[0] + x[1].flip(-1)

        return x


class Net_Feature(Net):

    def __init__(self, backbone_file, deploy=False, pretrained=True):
        super(Net_Feature, self).__init__(backbone_file, deploy=deploy, pretrained=pretrained)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)

        x = self.stage3(x)
        feature = self.stage4(x)

        return feature


class CAM2(Net):

    def __init__(self, backbone_file, deploy=False, pretrained=True):
        super(CAM2, self).__init__(backbone_file, deploy=deploy, pretrained=pretrained)

    def forward(self, x, separate=False):
        x = self.stage0(x)

        x = self.stage1(x)

        x = self.stage2(x)

        x = self.stage3(x)

        x = self.stage4(x)

        x = F.conv2d(x, self.classifier.weight)
        if separate:
            return x
        x = F.relu(x)

        x = x[0] + x[1].flip(-1)

        return x
