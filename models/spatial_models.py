'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from torch.nn import Conv2d as Conv2dNormal
from torch.nn import MaxPool2d as MaxPool2dNormal
from torch.nn import BatchNorm2d as BatchNorm2dNormal
from .spatial_modules import SpatialConv2d, SpatialMaxpool2d, SpatialBatchNorm2d
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal

Conv2d = None
MaxPool2d = None
BatchNorm2d = None


def img_show(img, ax, idx=0):
    img = img[idx, 0:3].clone().detach().cpu()
    mean = torch.Tensor([0.4914, 0.4822, 0.4465]).unsqueeze(-1).unsqueeze(-1)
    std = torch.Tensor([0.2023, 0.1994, 0.2010]).unsqueeze(-1).unsqueeze(-1)
    img = img * std + mean
    ax.imshow(np.rot90(img.permute(2, 1, 0).numpy(), k=-1) )


def plot_shapes(window_aggregated_shapes, ax, idx=0):
    centers = window_aggregated_shapes[idx, 0:2].view(2, -1).permute(1, 0).clone().detach().cpu().numpy()
    covs = window_aggregated_shapes[idx, 2:5].view(3, -1).clone().detach().cpu().numpy()
    cov_mats = np.zeros((centers.shape[0], 2, 2))
    cov_mats[:, 0, 0] = covs[0, :]
    cov_mats[:, 1, 1] = covs[1, :]
    cov_mats[:, 0, 1] = covs[2, :]
    cov_mats[:, 1, 0] = covs[2, :]
    plot_min_x, plot_max_x = -1, 33
    plot_min_y, plot_max_y = -1, 33
    x, y = np.mgrid[plot_min_x:plot_max_x:.05, plot_min_y:plot_max_y:.05]
    pos = np.dstack((x, y))
    z = np.zeros(pos.shape[0:2])
    for center, cov_mat in zip(centers, cov_mats):
        z += multivariate_normal(center, cov_mat).pdf(pos) * (cov_mat[0, 0]*cov_mat[1, 1])**2 / centers.shape[0]
    ax.contourf(-y, -x, z)
    ax.scatter(-centers[:, 1], -centers[:, 0], c='red', s=2)



class SpatialModel1(nn.Module):
    def __init__(self, normal=False, sparsity=0.0):
        super(SpatialModel1, self).__init__()
        global Conv2d
        global MaxPool2d
        global BatchNorm2d
        self.normal = normal
        if normal:
            Conv2d = Conv2dNormal
            MaxPool2d = MaxPool2dNormal
            BatchNorm2d = BatchNorm2dNormal
        else:
            Conv2d = SpatialConv2d
            MaxPool2d = SpatialMaxpool2d
            BatchNorm2d = SpatialBatchNorm2d
            Conv2d.sparsity = sparsity

        self.conv1 = Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = Conv2d(128, 128, kernel_size=3, padding=1)
        self.mp1 = MaxPool2d(kernel_size=2, stride=2)
        self.mp2 = MaxPool2d(kernel_size=2, stride=2)
        self.mp3 = MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Linear(2128, 10)
        self.batch_count = 0


    def forward(self, x):
        self.batch_count += 1
        c1 = self.conv1(x)
        mp1 = self.mp1(c1)
        c2 = self.conv2(mp1)
        mp2 = self.mp2(c2)
        c3 = self.conv3(mp2)
        c4 = self.conv4(c3)
        mp3 = self.mp3(c4)
        mp3_flat = mp3.view(mp3.size(0), -1)
        out = self.classifier(mp3_flat)
        if self.batch_count > 20:
            # Creates two subplots and unpacks the output array immediately
            f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=False)
            img_show(x, ax1)
            plot_shapes(mp1[:, -5:, ...], ax2)
            plot_shapes(mp2[:, -5:, ...], ax3)
            plot_shapes(mp3[:, -5:, ...], ax4)
            plt.show()
            self.batch_count = 0
        return out


def test():
    net = SpatialModel1()
    print('Num of parameters: ', sum(p.numel() for p in net.parameters() if p.requires_grad))
    x = torch.randn(2,3+5,32,32)
    y = net(x)
    print(y.size())

# test()
