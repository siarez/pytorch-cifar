'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from torch.nn import Conv2d as Conv2dNormal
from torch.nn import MaxPool2d as MaxPool2dNormal
from torch.nn import BatchNorm2d as BatchNorm2dNormal
from .spatial_modules import SpatialConv2d, SpatialMaxpool2d, SpatialBatchNorm2d
import numpy as np

cfg = {
    'VGG_tiny': [32, 'M', 64, 'M', 128, 128, 'M'],
    'VGG_mini': [32, 'M', 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

Conv2d = None
MaxPool2d = None
BatchNorm2d = None

def img_show(img, idx=0):
    from matplotlib import pyplot as plt
    img = img[idx, 0:3].clone().detach().cpu()
    mean = torch.Tensor([0.4914, 0.4822, 0.4465]).unsqueeze(-1).unsqueeze(-1)
    std = torch.Tensor([0.2023, 0.1994, 0.2010]).unsqueeze(-1).unsqueeze(-1)
    img = img * std + mean
    plt.imshow(np.rot90(img.permute(2, 1, 0).numpy(), k=-1) )
    plt.show()

class SpatialVGG(nn.Module):
    def __init__(self, vgg_name, normal=False, sparsity=0.0):
        super(SpatialVGG, self).__init__()
        global Conv2d
        global MaxPool2d
        global BatchNorm2d
        if normal:
            Conv2d = Conv2dNormal
            MaxPool2d = MaxPool2dNormal
            BatchNorm2d = BatchNorm2dNormal
        else:
            Conv2d = SpatialConv2d
            MaxPool2d = SpatialMaxpool2d
            BatchNorm2d = SpatialBatchNorm2d
            Conv2d.sparsity = sparsity
        self.features = self._make_layers(cfg[vgg_name])
        if vgg_name == 'VGG_mini':
            self.classifier = nn.Linear(256 + 5, 10)
        elif vgg_name == 'VGG_tiny':
            self.classifier = nn.Linear(2128, 10)
        else:
            self.classifier = nn.Linear(512 + 5, 10)
        self.counter = 0


    def forward(self, x):
        self.counter += 1
        if self.counter > 1000:
            img_show(x)
            self.counter = 0
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [Conv2d(in_channels, x, kernel_size=3, padding=1)]
                in_channels = x
        layers += [nn.BatchNorm2d(in_channels+5), nn.AvgPool2d(kernel_size=1, stride=1)]  # This last batch norm should be normal
        return nn.Sequential(*layers)


def test():
    net = SpatialVGG('VGG11')
    print('Num of parameters: ', sum(p.numel() for p in net.parameters() if p.requires_grad))
    x = torch.randn(2,3+5,32,32)
    y = net(x)
    print(y.size())

# test()
