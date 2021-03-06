import torch
from torch.nn import functional, Conv2d, Parameter, MaxPool2d, MaxUnpool2d, BatchNorm2d
from torch.nn.functional import fold, unfold


class SpatialConv2d(torch.nn.Module):
    """
    For now it only accepts square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, shape_passthrough=False):
        super(SpatialConv2d, self).__init__()
        self.in_channels, self.out_channels, self.k, self.pad = in_channels, out_channels, kernel_size, padding
        self.shape_passthrough = shape_passthrough
        self.conv = Conv2d(in_channels, out_channels, kernel_size)
        self.conv2 = Conv2d(out_channels, 1, kernel_size=1)
        # self.conv3 = Conv2d(out_channels, out_channels, kernel_size=1)
        self.pad2d = torch.nn.ReplicationPad2d(padding)  # Replication padding makes the most sense to me
        # moved batchnorm here so I can do it before the Relu, like the original architecture
        self.bn2d = BatchNorm2d(out_channels, affine=False)
        # todo: normalize. This can be tricky.
        self.shapes_kernel = Parameter(torch.rand((1, out_channels, kernel_size * kernel_size * 5, 1)))
        # `shapes_kernel_weight` is basically part of the shape distance function. Sharing it across channels makes sense, if
        # we want all channels use the same distance function. I think this is a good way of doing it, otherwise I think we are
        # over parameterizing.
        # self.shapes_kernel_weight = Parameter(torch.rand((1, 1, kernel_size * kernel_size * 5, 1)) / torch.tensor(kernel_size * kernel_size * 5.).sqrt())
        self.win_center_idx = kernel_size * kernel_size // 2
        self.query_scale_factor = Parameter(torch.tensor(self.out_channels).type(torch.FloatTensor).sqrt(), requires_grad=False)
        
    def forward(self, x):
        # conv_out = self.bn2d(x[:, :2, ...].repeat(1, self.out_channels//2, 1, 1))  # todo: testing
        # conv_out += torch.randn_like(conv_out) / 100
        # conv_out = functional.relu(conv_out)
        if not self.shape_passthrough:
            x = self.pad2d(x)
            conv_out = self.bn2d(self.conv(x[:, :-5, ...]))
            num_of_win = conv_out.shape[2] * conv_out.shape[3]
            # Only compute new shapes if `self.shape_passthrough` is False
            # ********** Compute shape concordance **********
            # Need to `clone()` because more than one element of the unfolded tensor may refer to a single memory location.
            # This would make subtraction of the local window centers no work properly when writing back to `shape_windows`
            shape_windows = unfold(x[:, -5:, ...], self.k).view(-1, 5, self.k * self.k, num_of_win).clone()
            shape_windows[:, -5:-3, :, :] = shape_windows[:, -5:-3, :, :] - shape_windows[:, -5:-3,
                                                                            self.win_center_idx:self.win_center_idx + 1, :]
            shape_difference = torch.abs(shape_windows.view(-1, 5 * self.k * self.k, num_of_win).unsqueeze(
                1) - self.shapes_kernel)**2  # experiment: l1 norm, could be other norms
            # could use `torch.dist` or `torch.nn.functional.pairwise_distance` if I didn't have `shapes_kernel_weight`
            # shape_distance_weighted = torch.sum(shape_difference * self.shapes_kernel_weight, dim=2).permute(1, 0, 2)
            # shape_distance_weighted = torch.mean(shape_difference, dim=2)
            shape_distance_weighted = torch.softmax(-torch.sum(shape_difference, dim=2), dim=1)
            shape_distance_weighted = fold(shape_distance_weighted, (x.shape[2]-self.k+1, x.shape[3]-self.k+1), (1, 1))
            # shape_attended_features = functional.relu(conv_out / (shape_distance_weighted + 1))  # +1 is to avoid division by zero.

            # shape_attended_features = functional.relu(conv_out)  # +1 is to avoid division by zero.
            shape_attended_features = functional.relu(conv_out * shape_distance_weighted)  # +1 is to avoid division by zero.
            # shape_attended_features = functional.relu(conv_out - shape_distance_weighted )  # +1 is to avoid division by zero.
            # Experiment: We could also use an exponential to modulate the conv with `shape_distance_weighted`. This
            # punishes shape mismatch more aggressively. It could be a good idea if the shapes end up having very low
            # variance(looking all similar).
            # out = functional.relu(conv_out) * torch.exp(-shape_distance_weighted)

            # ********** Aggregating shapes in the convolution window **********
            # The mean can be a weighted/attantion mean which uses `shape_attended_features` to create a query. Here the query
            # (aka weights) is calculated using a convolution operation. The weights then are unfolded into windows and a
            # softmax ensures that the weights in each window sums to one. Softmax can be replaced with other normalization as
            # long as the weight are possitive and sum to one.
            # Method 1:
            shape_query = self.conv2(shape_attended_features)  # Experiment: add relu after conv
            # This is a heuristic taken from the Attention paper. The idea is to avoid the extremes of the softmax. It should
            # be experimented with. I think it has to do with random walks.
            shape_query = shape_query  # / self.query_scale_factor
            shape_query_unfolded = functional.softmax(unfold(shape_query, (self.k, self.k), padding=1), dim=1).unsqueeze(1)

            # Method 2: A different method for calculating the query
            # shape_query = torch.norm(shape_attended_features, p=1, dim=1, keepdim=True)
            # shape_query = unfold(shape_query, (self.k, self.k), padding=1)
            # shape_query_unfolded = (shape_query / torch.norm(shape_query, p=1, dim=1, keepdim=True)).unsqueeze(1)

            # This line computes the weighted average of means for each window
            window_means = torch.sum(unfold(x[:, -5:-3, ...], (self.k, self.k)).
                view(-1, 2, self.k*self.k, num_of_win) * shape_query_unfolded, dim=2)

            # Part_1 is contribution of variances of ellipses. Part_2 is the contribution of how far the mean of each ellipse is
            # from the mean of the window.
            window_var_part_1 = torch.sum(
                unfold(x[:, -3:-1, ...], (self.k, self.k)).view(-1, 2, self.k*self.k, num_of_win) * shape_query_unfolded, dim=2)
            # window_var_part_2 = (((unfold(self.pad2d(x[:, -5:-3, ...]), (self.k, self.k)).view(-1, 2, self.k*self.k,num_of_win) - window_means.unsqueeze(2)) ** 2)
            #                      * shape_query_unfolded).sum(dim=2)  # This is most likely not correct, because of weight calculation and number also look wrong
            window_var_part_2 = (((unfold(x[:, -5:-3, ...], (self.k, self.k)).view(-1, 2, self.k * self.k,
                                                                           num_of_win) - window_means.unsqueeze(2)) ** 2) * shape_query_unfolded).sum(dim=2)

            window_var = window_var_part_1 + window_var_part_2
            # Part_1 is contribution of covariances of ellipses. Part_2 is the contribution of how far the mean of each ellipse is
            # from the mean of the window.
            window_covar_part_1 = torch.sum(
                unfold(x[:, -1:, ...], (self.k, self.k)).view(-1, 1, self.k*self.k, num_of_win) * shape_query_unfolded, dim=2)
            # window_covar_part_2 = \
            #     (((unfold(self.pad2d(x[:, -5:-4, ...]), (self.k, self.k)).view(-1, 1, self.k*self.k, num_of_win) - window_means[:,0:1,:].unsqueeze(2)) *
            #     (unfold(self.pad2d(x[:, -4:-3, ...]), (self.k, self.k)).view(-1, 1, self.k*self.k,num_of_win) - window_means[:,1:2,:].unsqueeze(2))) *
            #    shape_query_unfolded).sum(dim=2)  # Following the change in variance calculation, I replaced this as well
            window_covar_part_2 = \
                (((unfold(x[:, -5:-4, ...], (self.k, self.k)).view(-1, 1, self.k*self.k, num_of_win) - window_means[:,0:1,:].unsqueeze(2)) *
                (unfold(x[:, -4:-3, ...], (self.k, self.k)).view(-1, 1, self.k*self.k,num_of_win) - window_means[:,1:2,:].unsqueeze(2))) *
                 (shape_query_unfolded)).sum(dim=2)
            window_covar = window_covar_part_1 + window_covar_part_2
            window_aggregated_shapes = torch.cat([window_means, window_var, window_covar], dim=1)
            window_aggregated_shapes = fold(window_aggregated_shapes, (x.shape[2]-self.k+1, x.shape[3]-self.k+1), (1, 1))
            output = torch.cat([shape_attended_features, window_aggregated_shapes], dim=1)
        else:
            conv_out = functional.leaky_relu(self.bn2d(self.conv(self.pad2d(x[:, :-5, ...]))))
            output = torch.cat([conv_out, x[:, -5:, ...]], dim=1)

        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_channels, self.out_channels, self.kernel_size
        )


class SpatialMaxpool2d(torch.nn.Module):
    def __init__(self, kernel_size, stride=2, padding=0, dilation=1, groups=1, bias=False):
        super(SpatialMaxpool2d, self).__init__()
        self.k, self.stride = kernel_size, stride
        self.max_pool = MaxPool2d(kernel_size, stride=stride, padding=0, dilation=1, return_indices=True, ceil_mode=False)
        self.max_unpool = MaxUnpool2d(kernel_size, stride=stride, padding=0)
        self.pad2d = torch.nn.ReplicationPad2d(padding)

    def forward(self, x):
        x = self.pad2d(x)
        in_channels, h, w = x.shape[1], x.shape[2], x.shape[3]
        pooled_features, idx = self.max_pool(x[:, :-5, ...])
        num_of_win = pooled_features.shape[2] * pooled_features.shape[3]
        # ********** Shape aggregation **********
        max_unpooled = self.max_unpool(pooled_features, idx, output_size=(h, w))
        # The division a heuristic taken from the Attention paper. The idea is to avoid the extremes of the softmax. It should
        # be experimented with. I think it has to do with random walks.
        shape_weights = torch.norm(max_unpooled, p=1, dim=1, keepdim=True) / torch.tensor(in_channels - 5, requires_grad=False).type(torch.FloatTensor).sqrt()
        # shape_weights = torch.norm(x, p=2, dim=1, keepdim=True) / torch.tensor(in_channels - 5, requires_grad=False).type(torch.FloatTensor).sqrt()

        # shape_weights = max_unpooled.sum(dim=1, keepdim=True) / torch.tensor(in_channels - 5, requires_grad=False).type(
        #     torch.FloatTensor)  # .sqrt()
        # window_shape_query = functional.softmax(unfold(shape_weights, (self.k, self.k), padding=0, stride=self.stride),
        #                                         dim=1).unsqueeze(1)
        shape_windows = unfold(shape_weights, (self.k, self.k), padding=0, stride=self.stride)
        window_shape_query = (shape_windows / torch.norm(shape_windows, p=1, dim=1, keepdim=True)).unsqueeze(1)
        # window_shape_query = unfold(shape_weights, (self.k, self.k), padding=0, stride=self.stride).unsqueeze(1)
        # Computing window means
        window_means = torch.sum(unfold(x[:, -5:-3, ...], (self.k, self.k), stride=self.stride).view(-1, 2, self.k*self.k, num_of_win)
                                 * window_shape_query, dim=2)
        # Part_1 is contribution of variances of ellipses. Part_2 is the contribution of how far the mean of each ellipse is
        # from the mean of the window.
        window_var_part_1 = torch.sum(
            unfold(x[:, -3:-1, ...], (self.k, self.k), stride=self.stride).view(-1, 2, self.k*self.k, num_of_win) * window_shape_query, dim=2)
        window_var_part_2 = (((unfold(x[:, -5:-3, ...], (self.k, self.k), stride=self.stride).view(-1, 2, self.k*self.k, num_of_win) - window_means.
                               unsqueeze(2)) ** 2) * window_shape_query).sum(dim=2)
        window_var = window_var_part_1 + window_var_part_2
        # Part_1 is contribution of covariances of ellipses. Part_2 is the contribution of how far the mean of each ellipse is
        # from the mean of the window.
        window_covar_part_1 = torch.sum(
            unfold(x[:, -1:, ...], (self.k, self.k), stride=self.stride).view(-1, 1, self.k * self.k, num_of_win) * window_shape_query, dim=2)
        window_covar_part_2 = (((unfold(x[:, -5:-4, ...], (self.k, self.k), stride=self.stride).view(-1, 1, self.k * self.k, num_of_win) - window_means[:,0:1,:].unsqueeze(2)) *
                                (unfold(x[:, -4:-3, ...], (self.k, self.k), stride=self.stride).view(-1, 1, self.k * self.k, num_of_win) - window_means[:,1:2,:].unsqueeze(2))) *
                               window_shape_query).sum(dim=2)
        window_covar = window_covar_part_1 + window_covar_part_2
        window_aggregated_shapes = torch.cat([window_means, window_var, window_covar], dim=1)
        window_aggregated_shapes = fold(window_aggregated_shapes, (pooled_features.shape[2], pooled_features.shape[3]), (1, 1))
        # plot_shapes(window_aggregated_shapes, idx=0)
        output = torch.cat([pooled_features, window_aggregated_shapes], dim=1)
        return output

    def extra_repr(self):
        return 'kernel={}, stride={}'.format(self.k, self.stride)


class SpatialBatchNorm2d(torch.nn.Module):
    """Does batchnorm on feature channels but doesn't touch shape channels"""
    def __init__(self, channels):
        super(SpatialBatchNorm2d, self).__init__()
        self.ch = channels
        self.bn2d = BatchNorm2d(channels)

    def forward(self, x):
        x_bn = self.bn2d(x[:, :-5, ...])
        return torch.cat([x_bn, x[:, -5:, ...]], dim=1)

    def extra_repr(self):
        return 'channels={}'.format(self.ch)
