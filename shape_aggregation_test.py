'''VGG11/13/16/19 in Pytorch.'''
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import torch
from torch.nn import functional, MaxPool2d, MaxUnpool2d
from torch.nn.functional import fold, unfold

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
    plot_min_x, plot_max_x = 0, size
    plot_min_y, plot_max_y = 0, size
    ax.axis(xmin=plot_min_x, xmax=plot_max_x, ymin=plot_min_y, ymax=plot_max_y)
    ax.invert_yaxis()
    x, y = np.mgrid[plot_min_x:plot_max_x:.05, plot_min_y:plot_max_y:.05]
    pos = np.dstack((x, y))
    z = np.zeros(pos.shape[0:2])
    for center, cov_mat, intensity in zip(centers, cov_mats, intensities):
        # z += multivariate_normal(center, cov_mat).pdf(pos) * (cov_mat[0, 0]+cov_mat[1, 1]) / 2 / centers.shape[0]
        confidence_ellipse(center, cov_mat, ax, edgecolor=(intensity, 0.0, 1 - intensity))
        # z += multivariate_normal(center, cov_mat).pdf(pos) / centers.shape[0]
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


class SpatialMaxpool2d(torch.nn.Module):
    def __init__(self, kernel_size, stride=2, padding=0, dilation=1, groups=1, bias=False):
        super(SpatialMaxpool2d, self).__init__()
        self.k, self.stride = kernel_size, stride
        self.max_pool = MaxPool2d(kernel_size, stride=stride, padding=0, dilation=1, return_indices=True, ceil_mode=False)
        self.max_unpool = MaxUnpool2d(kernel_size, stride=stride, padding=0)

    def forward(self, x):
        in_channels, h, w = x.shape[1], x.shape[2], x.shape[3]
        pooled_features, idx = self.max_pool(x[:, :-5, ...])
        num_of_win = pooled_features.shape[2] * pooled_features.shape[3]
        # ********** Shape aggregation **********
        max_unpooled = self.max_unpool(pooled_features, idx, output_size=(h, w))
        # The division a heuristic taken from the Attention paper. The idea is to avoid the extremes of the softmax. It should
        # be experimented with. I think it has to do with random walks.
        shape_weights = torch.norm(max_unpooled, p=1, dim=1, keepdim=True) / torch.tensor(in_channels - 5, requires_grad=False).type(torch.FloatTensor).sqrt()
        shape_weights = torch.norm(x, p=1, dim=1, keepdim=True) / torch.tensor(in_channels - 5, requires_grad=False).type(torch.FloatTensor).sqrt()
        # shape_weights = max_unpooled.sum(dim=1, keepdim=True) / torch.tensor(in_channels - 5, requires_grad=False).type(
        #     torch.FloatTensor)  # .sqrt()
        # window_shape_query = functional.softmax(unfold(shape_weights, (self.k, self.k), padding=0, stride=self.stride),
        #                                         dim=1).unsqueeze(1)
        shape_windows = unfold(shape_weights, (self.k, self.k), padding=0, stride=self.stride)
        window_shape_query = (shape_windows / torch.norm(shape_windows, p=1, dim=1, keepdim=True)).unsqueeze(1)
        # window_shape_query = window_shape_query / (window_shape_query + 1)
        # window_shape_query = unfold(shape_weights, (self.k, self.k), padding=0, stride=self.stride).unsqueeze(1)
        # Computing window means
        window_means = torch.sum(unfold(x[:, -5:-3, ...], (self.k, self.k), stride=self.stride).view(-1, 2, self.k*self.k, num_of_win)
                                 * window_shape_query, dim=2)
        # Part_1 is contribution of variances of ellipses. Part_2 is the contribution of how far the mean of each ellipse is
        # from the mean of the window.
        window_var_part_1 = torch.sum(
            unfold(x[:, -3:-1, ...], (self.k, self.k), stride=self.stride).view(-1, 2, self.k*self.k, num_of_win) * window_shape_query, dim=2)
        window_var_part_2 = ((((unfold(x[:, -5:-3, ...], (self.k, self.k), stride=self.stride).view(-1, 2, self.k*self.k, num_of_win) - window_means.
                               unsqueeze(2))) ** 2) * window_shape_query).sum(dim=2)
        window_var = window_var_part_1 + window_var_part_2
        # Part_1 is contribution of covariances of ellipses. Part_2 is the contribution of how far the mean of each ellipse is
        # from the mean of the window.
        window_covar_part_1 = torch.sum(
            unfold(x[:, -1:, ...], (self.k, self.k), stride=self.stride).view(-1, 1, self.k * self.k, num_of_win) * window_shape_query, dim=2)
        window_covar_part_2 = (((unfold(x[:, -5:-4, ...], (self.k, self.k), stride=self.stride).view(-1, 1, self.k * self.k, num_of_win) - window_means[:,0:1,:].unsqueeze(2)) *
                                (unfold(x[:, -4:-3, ...], (self.k, self.k), stride=self.stride).view(-1, 1, self.k * self.k, num_of_win) - window_means[:,1:2,:].unsqueeze(2))) *
                               (window_shape_query**2)).sum(dim=2)
        window_covar = window_covar_part_1 + window_covar_part_2
        window_aggregated_shapes = torch.cat([window_means, window_var, window_covar], dim=1)
        window_aggregated_shapes = fold(window_aggregated_shapes, (pooled_features.shape[2], pooled_features.shape[3]), (1, 1))
        # plot_shapes(window_aggregated_shapes, idx=0)
        output = torch.cat([pooled_features, window_aggregated_shapes], dim=1)
        return output

shape_pool = SpatialMaxpool2d(2, 2)

h, w = 16, 16
batch = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

shape_map_center = torch.stack(torch.meshgrid(torch.arange(0.5, h, step=1), torch.arange(0.5, w, step=1))).unsqueeze(
    0).repeat(batch, 1, 1, 1)
shape_map_var = torch.ones((batch, 2, h, w)) / 16  # 4 is a hyper parameter determining the diameter of pixels.
shape_map_cov = torch.zeros((batch, 1, h, w))
shape_map = torch.cat([shape_map_center, shape_map_var, shape_map_cov], dim=1).to(device)
# shape_map[0, 0, 1, 1] += -0.5
# shape_map[0, 1, 1, 1] += +0.5
# shape_map[0, 2, 1, 0] += +0.2
# shape_map[0, 2, 0, 0] += +0.1
# shape_map[0, 3, 0, 0] += +0.1

while True:
    features = (torch.randn(batch, 4, h, w)/100 + 1).to(device)
    # features[:, 0, 0, 0] = 10
    # features[:, 1, 1, 1] = 10
    # features[:, 2, 1, 0] = 10
    # features[:, 3, 0, 1] = 10
    feature_shape = torch.cat([features, shape_map[:features.shape[0], ...]], dim=1)
    output1 = shape_pool(feature_shape)
    output2 = shape_pool(output1)
    output3 = shape_pool(output2)
    output4 = shape_pool(output3)
    # Creates two subplots and unpacks the output array immediately
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
    ax1.set_aspect(1)
    ax2.set_aspect(1)
    plot_shapes(feature_shape, ax1, size=h)
    plot_shapes(output3, ax2, size=h)
    plt.show()
    print('')