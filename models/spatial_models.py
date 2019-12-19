'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from torch.nn import Conv2d as Conv2dNormal
from torch.nn import MaxPool2d as MaxPool2dNormal
from torch.nn import BatchNorm2d as BatchNorm2dNormal
from .spatial_modules import SpatialConv2d, SpatialMaxpool2d, SpatialBatchNorm2d, SpatialMaxpool2d_2, SpatialMaxpool2d_3
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

Conv2d = None
MaxPool2d = None
BatchNorm2d = None


def img_show(img, ax, idx=0):
    img = img[idx, 0:3].clone().detach().cpu()
    mean = torch.Tensor([0.4914, 0.4822, 0.4465]).unsqueeze(-1).unsqueeze(-1)
    std = torch.Tensor([0.2023, 0.1994, 0.2010]).unsqueeze(-1).unsqueeze(-1)
    img = img * std + mean
    ax.imshow(img.permute(1, 2, 0).numpy(), interpolation='none', extent=(0, 32, 32, 0))
    minor_grid_interval = img.shape[1] / 32
    major_grid_interval = img.shape[1] / 8
    minor_grid_locations = np.arange(minor_grid_interval, img.shape[1], minor_grid_interval)
    major_grid_locations = np.arange(major_grid_interval, img.shape[1], major_grid_interval)
    ax.set_yticks(minor_grid_locations, minor=True)
    ax.set_xticks(minor_grid_locations, minor=True)
    ax.yaxis.grid(True, which='minor', linestyle=':')
    ax.xaxis.grid(True, which='minor', linestyle=':')
    ax.set_yticks(major_grid_locations, minor=False)
    ax.set_xticks(major_grid_locations, minor=False)
    ax.yaxis.grid(True, which='major', linestyle='-')
    ax.xaxis.grid(True, which='major', linestyle='-')


def confidence_ellipse(center, cov, ax, n_std=1.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse

    Parameters
    ----------
    center, cov : array_like, shape (n, )
        center and cov matrix. Note: first dimension is y and the 2nd is x

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = center[0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = center[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_y, scale_x) \
        .translate(mean_y, mean_x)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def plot_shapes(window_aggregated_shapes, ax, idx=0, size=32):
    centers = window_aggregated_shapes[idx, -5:-3].view(2, -1).permute(1, 0).clone().detach().cpu().numpy()
    covs = window_aggregated_shapes[idx, -3:].view(3, -1).clone().detach().cpu().numpy()
    intensities = torch.norm(window_aggregated_shapes[:, :-5, ...], p=1, dim=1)[idx].view(-1).detach().cpu().numpy()
    intensities /= intensities.max()
    cov_mats = np.zeros((centers.shape[0], 2, 2))
    cov_mats[:, 0, 0] = covs[0, :]
    cov_mats[:, 1, 1] = covs[1, :]
    cov_mats[:, 0, 1] = covs[2, :]
    cov_mats[:, 1, 0] = covs[2, :]
    plot_min_x, plot_max_x = 0, 32
    plot_min_y, plot_max_y = 0, 32
    ax.axis(xmin=plot_min_x, xmax=plot_max_x, ymin=plot_min_y, ymax=plot_max_y)
    ax.invert_yaxis()
    x, y = np.mgrid[plot_min_x:plot_max_x:.05, plot_min_y:plot_max_y:.05]
    pos = np.dstack((x, y))
    z = np.zeros(pos.shape[0:2])
    for center, cov_mat, intensity in zip(centers, cov_mats, intensities):
        # z += multivariate_normal(center, cov_mat).pdf(pos) * (cov_mat[0, 0]+cov_mat[1, 1]) / 2 / centers.shape[0]
        confidence_ellipse(center, cov_mat, ax, edgecolor=(intensity, 0.0, 1 - intensity))
        # z += multivariate_normal(center, cov_mat).pdf(pos) / centers.shape[0]
        pass
    minor_grid_interval = size / window_aggregated_shapes.shape[-1]
    major_grid_interval = size * 2 / window_aggregated_shapes.shape[-1]
    minor_grid_locations = np.arange(minor_grid_interval, size, minor_grid_interval)
    major_grid_locations = np.arange(major_grid_interval, size, major_grid_interval)
    ax.set_yticks(minor_grid_locations, minor=True)
    ax.set_xticks(minor_grid_locations, minor=True)
    ax.yaxis.grid(True, which='minor', linestyle=':')
    ax.xaxis.grid(True, which='minor', linestyle=':')
    ax.set_yticks(major_grid_locations, minor=False)
    ax.set_xticks(major_grid_locations, minor=False)
    ax.yaxis.grid(True, which='major', linestyle='-')
    ax.xaxis.grid(True, which='major', linestyle='-')
    # ax.contourf(y, x, z)
    ax.scatter(centers[:, 1], centers[:, 0], c='red', s=2)


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
            MaxPool2d = SpatialMaxpool2d_2
            # MaxPool2d = SpatialMaxpool2d
            BatchNorm2d = SpatialBatchNorm2d
            Conv2d.sparsity = sparsity

        self.conv1 = Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = Conv2d(128, 128, kernel_size=3, padding=1)

        padding = 0 if normal else (0, 1, 0, 1)
        self.mp1 = MaxPool2d(kernel_size=2, stride=2, padding=padding)
        self.mp2 = MaxPool2d(kernel_size=2, stride=2, padding=padding)
        self.mp3 = MaxPool2d(kernel_size=2, stride=2, padding=padding)
        # self.mp1 = MaxPool2d(kernel_size=3, stride=2, padding=(1, 2, 1, 2))
        # self.mp2 = MaxPool2d(kernel_size=3, stride=2, padding=(1, 2, 1, 2))
        # self.mp3 = MaxPool2d(kernel_size=3, stride=2, padding=(1, 2, 1, 2))
        # self.mp1 = MaxPool2d(kernel_size=3, stride=2, padding=(0, 2, 0, 2))
        # self.mp2 = MaxPool2d(kernel_size=3, stride=2, padding=(0, 2, 0, 2))
        # self.mp3 = MaxPool2d(kernel_size=3, stride=2, padding=(0, 2, 0, 2))
        # self.mp1 = MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.mp2 = MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.mp3 = MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.classifier = nn.Linear(2128, 10)
        # self.classifier = nn.Linear(3325, 10)
        # self.classifier = nn.Linear(3200, 10)
        self.classifier = nn.Linear(1968, 100) if normal else nn.Linear(2048, 100)
        # self.classifier = nn.Linear(1152, 10)
        self.batch_count = 0
        self.test_img_interval = 1000000


    def forward(self, x):

        if self.batch_count > self.test_img_interval:
            # Creating a dummy input to inspect shape pooling
            # x[:, :-5, :, :] = torch.zeros_like(x[:, :-5, :, :]) - 1 + torch.randn_like(x[:, :-5, :, :])/20
            # ones = torch.ones(x.shape[0], x.shape[1] - 5, 16, 4) + torch.randn((x.shape[0], x.shape[1] - 5, 16, 4))/20
            # x[:, :-5, 12:28, 12:16] = ones
            pass
        c1 = self.conv1(x)
        mp1 = self.mp1(c1)
        c2 = self.conv2(mp1)
        mp2 = self.mp2(c2)
        c3 = self.conv3(mp2)
        c4 = self.conv4(c3)
        mp3 = self.mp3(c4)
        mp3_flat = mp3[:, :-5, ...].view(mp3.size(0), -1)
        out = self.classifier(mp3_flat)
        if self.batch_count > self.test_img_interval:
            # Creates two subplots and unpacks the output array immediately
            f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, sharey=False)
            ax1.set_aspect(1)
            ax2.set_aspect(1)
            ax3.set_aspect(1)
            ax4.set_aspect(1)
            ax5.set_aspect(1)
            ax6.set_aspect(1)
            img_show(x, ax1)
            plot_shapes(mp1, ax2)
            plot_shapes(mp2, ax3)
            plot_shapes(mp3, ax4)
            plot_shapes(self.mp1(mp1), ax5)
            plot_shapes(self.mp1(self.mp1(mp1)), ax6)
            plt.show()
            self.batch_count = 0
            self.test_img_interval = 300
        self.batch_count += 1
        return out


def test():
    net = SpatialModel1()
    print('Num of parameters: ', sum(p.numel() for p in net.parameters() if p.requires_grad))
    x = torch.randn(2, 3+5, 32, 32)
    y = net(x)
    print(y.size())

# test()
