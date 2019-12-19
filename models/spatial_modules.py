import torch
from torch.nn import functional, Conv2d, Parameter, MaxPool2d, MaxUnpool2d, BatchNorm2d, Softmax2d, utils
from torch.nn.functional import fold, unfold


class SpatialConv2d(torch.nn.Module):
    """
    For now it only accepts square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, shape_passthrough=True, cat_distance=False):
        super(SpatialConv2d, self).__init__()
        self.in_channels, self.out_channels, self.k, self.pad = in_channels, out_channels, kernel_size, padding
        self.shape_passthrough = shape_passthrough
        self.cat_distance = cat_distance
        self.conv = Conv2d(in_channels + (out_channels if cat_distance else 0), out_channels, kernel_size)
        self.conv2 = Conv2d(out_channels, 1, kernel_size=1)
        # self.conv3_shape_mux = Conv2d(out_channels, out_channels, kernel_size=1, bias=False, )
        self.pad2d = torch.nn.ReplicationPad2d(padding)  # Replication padding makes the most sense to me
        # moved batchnorm here so I can do it before the Relu, like the original architecture
        self.bn2d = BatchNorm2d(out_channels, affine=False)
        self.bn2d2 = BatchNorm2d(out_channels, affine=False)
        # Initializing shape kernels
        kernel_init_coef = 1.0  # / torch.sqrt(torch.tensor(kernel_size * kernel_size * 5.0))
        self.shapes_kernel = torch.zeros((1, out_channels, 5, kernel_size * kernel_size, 1))
        # I'm initializing the centers so they are not completely out of order
        kernel_centers_init = torch.stack(torch.meshgrid(torch.arange(-(kernel_size // 2), 1 + (kernel_size // 2), step=1.),
                                   torch.arange(-(kernel_size // 2), 1 + (kernel_size // 2), step=1.))) \
                                    .view(1, 1, 2, kernel_size * kernel_size, 1)
        # initializing variance to values around one.
        kernel_vars_init = torch.ones((1, out_channels, 2, kernel_size * kernel_size, 1)) + (torch.rand((1, out_channels, 2, kernel_size * kernel_size, 1)) / 10)
        kernel_covars_init = torch.rand((1, out_channels, 1, kernel_size * kernel_size, 1)) - 0.5
        self.shapes_kernel[:, :, 0:2, :, :] = self.shapes_kernel[:, :, 0:2, :, :] + kernel_centers_init
        self.shapes_kernel[:, :, 2:4, :, :] = self.shapes_kernel[:, :, 2:4, :, :] + kernel_vars_init
        self.shapes_kernel[:, :, 4:, :, :] = self.shapes_kernel[:, :, 4:, :, :] + kernel_covars_init
        self.shapes_kernel = Parameter(self.shapes_kernel * kernel_init_coef)
        # self.shapes_kernel = Parameter((torch.rand((1, out_channels, 5, kernel_size * kernel_size, 1))) * kernel_init_coef)
        # `shapes_kernel_weight` is basically part of the shape distance function. Sharing it across channels makes sense, if
        # we want all channels use the same distance function. I think this is a good way of doing it, otherwise I think we are
        # over parameterizing.
        # self.shapes_kernel_weight = Parameter(torch.rand((1, 1, kernel_size * kernel_size * 5, 1)) / torch.tensor(kernel_size * kernel_size * 5.).sqrt())
        self.shapes_channel_weight = Parameter(torch.rand((1, self.out_channels, 1)))
        self.win_center_idx = kernel_size * kernel_size // 2
        self.query_scale_factor = Parameter(torch.tensor(self.out_channels).type(torch.FloatTensor).sqrt(), requires_grad=False)
        self.sm2d = Softmax2d()
        self.concordance_coef = Parameter(torch.tensor(1.0))
        self.concordance_coef2 = Parameter(torch.tensor(2.0))
        self.shape_difference_logging = None
        self.register_buffer('shape_distance_weighted', None)
        self.register_buffer('shape_distance_weighted_muxed', None)
        self.register_buffer('conv_batchnorm', None)
        # utils.clip_grad_value_(self.parameters(), 0.1)
        
    def forward(self, x):
        if not self.shape_passthrough:
            if not self.cat_distance:
                x_padded = self.pad2d(x)
                conv_out = self.conv(x_padded[:, :-5, ...])
                num_of_win = conv_out.shape[2] * conv_out.shape[3]
                # Only compute new shapes if `self.shape_passthrough` is False
                # ********** Compute shape concordance **********
                # Need to `clone()` because more than one element of the unfolded tensor may refer to a single memory location.
                # This would make subtraction of the local window centers to work properly when writing back to `shape_windows`
                shape_windows = unfold(x_padded[:, -5:, ...], self.k).view(-1, 5, self.k * self.k, num_of_win).clone()
                shape_windows[:, -5:-3, :, :] = shape_windows[:, -5:-3, :, :] - shape_windows[:, -5:-3, self.win_center_idx:self.win_center_idx + 1, :]
                shapes_delta = (shape_windows.unsqueeze(1) - self.shapes_kernel).view(-1, self.out_channels, 5*(self.k**2), num_of_win)
                # shapes_delta = (torch.exp((shape_windows.unsqueeze(1) - self.shapes_kernel)**2)).view(-1, self.out_channels, 5*self.k**2, num_of_win)
                # shape_difference = shapes_delta.sum(dim=2)
                shape_difference = torch.norm(shapes_delta, p=1, dim=2) # / self.query_scale_factor  # experiment: l1 norm, could be other norms
                # shape_difference_soft = torch.softmax(-shape_difference)  # bigger difference results in smaller multiplier
                # shape_difference_soft = shape_difference_soft*self.shapes_channel_weight # learnable scale factor
                self.shape_distance_weighted = fold(shape_difference, (x_padded.shape[2] - self.k + 1, x_padded.shape[3] - self.k + 1), (1, 1))
                # shape_distance_weighted = self.sm2d(-shape_distance_weighted * self.concordance_coef2)
                # shape_distance_weighted = self.bn2d3(shape_distance_weighted)
                # self.shape_distance_weighted_muxed = functional.relu(self.bn2d2(self.conv3_shape_mux(self.shape_distance_weighted)))
                self.shape_distance_weighted_muxed = functional.relu(self.bn2d2(self.shape_distance_weighted))
                self.conv_batchnorm = self.bn2d(conv_out)
                shape_attended_features = functional.relu(self.conv_batchnorm - self.shape_distance_weighted_muxed)
                # shape_attended_features = functional.leaky_relu(self.bn2d(conv_out)) * shape_distance_weighted
                # shape_attended_features = functional.leaky_relu(self.bn2d(conv_out) + functional.leaky_relu(self.bn2d2(self.conv3(conv_out))) * self.shape_distance_weighted)


                # shape_attended_features = functional.relu(self.bn2d(conv_out - shape_distance_weighted * self.concordance_coef))
                # Experiment: We could also use an exponential to modulate the conv with `shape_distance_weighted`. This
                # punishes shape mismatch more aggressively. It could be a good idea if the shapes end up having very low
                # variance(looking all similar).
                # out = functional.relu(conv_out) * torch.exp(-shape_distance_weighted)

                # ********** Aggregating shapes in the convolution window **********
                # The mean can be a weighted/attantion mean which uses `shape_attended_features` to create a query. Here the query
                # (aka weights) is calculated using a convolution operation. The weights then are unfolded into windows and a
                # softmax ensures that the weights in each window sums to one. Softmax can be replaced with other normalization as
                # long as the weight are possitive and sum to one.
                # # Method 1:
                # shape_query = self.conv2(shape_attended_features)  # Experiment: add relu after conv
                # # This is a heuristic taken from the Attention paper. The idea is to avoid the extremes of the softmax. It should
                # # be experimented with. I think it has to do with random walks.
                # shape_query = shape_query  # / self.query_scale_factor
                # shape_query_unfolded = functional.softmax(unfold(shape_query, (self.k, self.k), padding=1), dim=1).unsqueeze(1)
                #
                # # Method 2: A different method for calculating the query
                # # shape_query = torch.norm(shape_attended_features, p=1, dim=1, keepdim=True)
                # # shape_query = unfold(shape_query, (self.k, self.k), padding=1)
                # # shape_query_unfolded = (shape_query / torch.norm(shape_query, p=1, dim=1, keepdim=True)).unsqueeze(1)
                #
                # # This line computes the weighted average of means for each window
                # window_means = torch.sum(unfold(x_padded[:, -5:-3, ...], (self.k, self.k)).
                #     view(-1, 2, self.k*self.k, num_of_win) * shape_query_unfolded, dim=2)
                #
                # # Part_1 is contribution of variances of ellipses. Part_2 is the contribution of how far the mean of each ellipse is
                # # from the mean of the window.
                # window_var_part_1 = torch.sum(
                #     unfold(x_padded[:, -3:-1, ...], (self.k, self.k)).view(-1, 2, self.k*self.k, num_of_win) * shape_query_unfolded, dim=2)
                # window_var_part_2 = (((unfold(x_padded[:, -5:-3, ...], (self.k, self.k)).view(-1, 2, self.k * self.k,
                #                                                                num_of_win) - window_means.unsqueeze(2)) ** 2) * shape_query_unfolded).sum(dim=2)
                # window_var = window_var_part_1 + window_var_part_2
                # # Part_1 is contribution of covariances of ellipses. Part_2 is the contribution of how far the mean of each ellipse is
                # # from the mean of the window.
                # window_covar_part_1 = torch.sum(
                #     unfold(x_padded[:, -1:, ...], (self.k, self.k)).view(-1, 1, self.k*self.k, num_of_win) * shape_query_unfolded, dim=2)
                # window_covar_part_2 = \
                #     (((unfold(x_padded[:, -5:-4, ...], (self.k, self.k)).view(-1, 1, self.k*self.k, num_of_win) - window_means[:,0:1,:].unsqueeze(2)) *
                #     (unfold(x_padded[:, -4:-3, ...], (self.k, self.k)).view(-1, 1, self.k*self.k,num_of_win) - window_means[:,1:2,:].unsqueeze(2))) *
                #      (shape_query_unfolded)).sum(dim=2)
                # window_covar = window_covar_part_1 + window_covar_part_2
                # window_aggregated_shapes = torch.cat([window_means, window_var, window_covar], dim=1)
                # window_aggregated_shapes = fold(window_aggregated_shapes, (x_padded.shape[2]-self.k+1, x_padded.shape[3]-self.k+1), (1, 1))
                # output = torch.cat([shape_attended_features, window_aggregated_shapes], dim=1)

                output = torch.cat([shape_attended_features, x[:, -5:, ...]], dim=1)
            else:
                """Here the idea is to append the distances to the input of the convolution, instead of subtracting them from its output.
                The hope is for the convolution to learn to use the distance information. It has not worked very well so far."""
                x_padded = self.pad2d(x)
                num_of_win = x.shape[2] * x.shape[3]
                # Only compute new shapes if `self.shape_passthrough` is False
                # ********** Compute shape concordance **********
                # Need to `clone()` because more than one element of the unfolded tensor may refer to a single memory location.
                # This would make subtraction of the local window centers to work properly when writing back to `shape_windows`
                shape_windows = unfold(x_padded[:, -5:, ...], self.k).view(-1, 5, self.k * self.k, num_of_win).clone()
                shape_windows[:, -5:-3, :, :] = shape_windows[:, -5:-3, :, :] - shape_windows[:, -5:-3, self.win_center_idx:self.win_center_idx + 1,
                                                                                :]
                shapes_delta = (shape_windows.unsqueeze(1) - self.shapes_kernel).view(-1, self.out_channels, 5 * (self.k ** 2), num_of_win)
                # shapes_delta = (torch.exp((shape_windows.unsqueeze(1) - self.shapes_kernel)**2)).view(-1, self.out_channels, 5*self.k**2, num_of_win)
                # shape_difference = shapes_delta.sum(dim=2)
                shape_difference = torch.norm(shapes_delta, p=1, dim=2)  # / self.query_scale_factor  # experiment: l1 norm, could be other norms
                # shape_difference_soft = torch.softmax(-shape_difference)  # bigger difference results in smaller multiplier
                # shape_difference_soft = shape_difference_soft*self.shapes_channel_weight # learnable scale factor
                self.shape_distance_weighted = fold(shape_difference, (x_padded.shape[2] - self.k + 1, x_padded.shape[3] - self.k + 1), (1, 1))
                # shape_distance_weighted = self.sm2d(-shape_distance_weighted * self.concordance_coef2)
                # shape_distance_weighted = self.bn2d3(shape_distance_weighted)
                # shape_distance_weighted = functional.relu(self.bn2d2(self.conv3_shape_mux(self.shape_distance_weighted)))
                shape_distance_weighted = functional.relu(self.bn2d2(self.shape_distance_weighted))
                # shape_attended_features = functional.relu(self.bn2d(shape_distance_weighted))
                # conv_out = self.conv(torch.cat([x_padded[:, :-5, ...], torch.zeros_like(self.pad2d(shape_distance_weighted))], dim=1))
                conv_out = self.conv(torch.cat([x_padded[:, :-5, ...], self.pad2d(shape_distance_weighted)], dim=1))
                shape_attended_features = functional.relu(self.bn2d(conv_out))

                output = torch.cat([shape_attended_features, x[:, -5:, ...]], dim=1)
        else:
            conv_out = functional.relu(self.bn2d(self.conv(self.pad2d(x[:, :-5, ...]))))
            output = torch.cat([conv_out, x[:, -5:, ...]], dim=1)

        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_channels, self.out_channels, self.kernel_size
        )


class SpatialMaxpool2d_3(torch.nn.Module):
    def __init__(self, kernel_size, stride=2, padding=0, dilation=1, groups=1, bias=False):
        super(SpatialMaxpool2d_3, self).__init__()
        self.k, self.stride = kernel_size, stride
        self.pad2d = torch.nn.ReplicationPad2d(padding)

    def forward(self, x):
        column_weights = torch.norm(x[:, :-5, ...], p=1, dim=1)
        unfold(x, (self.k, self.k), padding=0)
        pass


class SpatialMaxpool2d_2(torch.nn.Module):
    def __init__(self, kernel_size, stride=2, padding=0, dilation=1, groups=1, bias=False):
        super(SpatialMaxpool2d_2, self).__init__()
        self.k, self.stride = kernel_size, stride
        self.max_pool = MaxPool2d(kernel_size, stride=stride, padding=0, dilation=1, return_indices=True, ceil_mode=False)
        self.max_pool2 = MaxPool2d(2, stride=2, padding=0, dilation=1, return_indices=True, ceil_mode=False)
        self.max_unpool = MaxUnpool2d(kernel_size, stride=stride, padding=0)
        self.pad2d = torch.nn.ReplicationPad2d(padding)

    def forward(self, x):
        x = self.pad2d(x)
        x_h, x_w = x.shape[2], x.shape[3]
        num_of_win2 = (x_h - self.k + 1) * (x_w - self.k + 1)
        in_channels, h, w = x.shape[1], x.shape[2], x.shape[3]
        pooled_features, idx = self.max_pool(x[:, :-5, ...])
        num_of_win = pooled_features.shape[2] * pooled_features.shape[3]
        # ********** Shape aggregation **********
        max_unpooled = self.max_unpool(pooled_features, idx, output_size=(h, w)).detach()
        # The division a heuristic taken from the Attention paper. The idea is to avoid the extremes of the softmax. It should
        # be experimented with. I think it has to do with random walks.
        shape_weights = torch.norm(max_unpooled, p=1, dim=1, keepdim=True) / torch.tensor(in_channels - 5, requires_grad=False).type(torch.FloatTensor).sqrt()
        shape_stride = 1
        shape_windows_weights = unfold(shape_weights, (self.k, self.k), padding=0, stride=shape_stride)
        window_shape_query = (shape_windows_weights / (torch.norm(shape_windows_weights, p=1, dim=1, keepdim=True)+0.001)).unsqueeze(1)
        # window_shape_query = unfold(shape_weights, (self.k, self.k), padding=0, stride=self.stride).unsqueeze(1)
        # Computing window means
        ellipse_centers = unfold(x[:, -5:-3, ...], (self.k, self.k), stride=shape_stride)
        window_means = torch.sum(ellipse_centers.view(-1, 2, self.k*self.k, num_of_win2) * window_shape_query, dim=2)
        # Part_1 is contribution of variances of ellipses. Part_2 is the contribution of how far the mean of each ellipse is
        # from the mean of the window.
        window_var_part_1 = torch.sum(
            unfold(x[:, -3:-1, ...], (self.k, self.k), stride=shape_stride).view(-1, 2, self.k*self.k, num_of_win2) * window_shape_query, dim=2)
        window_var_part_2 = (((unfold(x[:, -5:-3, ...], (self.k, self.k), stride=shape_stride).view(-1, 2, self.k*self.k, num_of_win2) - window_means.
                               unsqueeze(2)) ** 2) * window_shape_query).sum(dim=2)
        window_var = window_var_part_1 + window_var_part_2
        # Part_1 is contribution of covariances of ellipses. Part_2 is the contribution of how far the mean of each ellipse is
        # from the mean of the window.
        window_covar_part_1 = torch.sum(
            unfold(x[:, -1:, ...], (self.k, self.k), stride=shape_stride).view(-1, 1, self.k * self.k, num_of_win2) * window_shape_query, dim=2)
        window_covar_part_2 = (((unfold(x[:, -5:-4, ...], (self.k, self.k), stride=shape_stride).view(-1, 1, self.k * self.k, num_of_win2) - window_means[:,0:1,:].unsqueeze(2)) *
                                (unfold(x[:, -4:-3, ...], (self.k, self.k), stride=shape_stride).view(-1, 1, self.k * self.k, num_of_win2) - window_means[:,1:2,:].unsqueeze(2))) *
                               window_shape_query).sum(dim=2)
        window_covar = window_covar_part_1 + window_covar_part_2
        window_aggregated_shapes = torch.cat([window_means, window_var, window_covar], dim=1)
        window_aggregated_shapes = fold(window_aggregated_shapes, (x_h - self.k + 1, x_h - self.k + 1), (1, 1))
        # whether to do max pool overlapping neighbourhoods or do regular pooling
        if False:
            # todo `dim=` are wrong, they are place holders
            neghbourhood_weights = fold(shape_windows_weights.sum(dim=1, keepdim=True), (x_h - self.k + 1, x_w - self.k + 1), (1, 1))
            _, heavy_neighbourhoods_idx = self.max_pool2(neghbourhood_weights)
            # gather indices from
            heavy_neighbourhoods_idx = heavy_neighbourhoods_idx.expand(heavy_neighbourhoods_idx.shape[0], 5, heavy_neighbourhoods_idx.shape[2],
                                            heavy_neighbourhoods_idx.shape[3])
            flattened_window_shapes = window_aggregated_shapes.flatten(start_dim=2)
            pooled_shapes = flattened_window_shapes.gather(dim=2, index=heavy_neighbourhoods_idx.flatten(start_dim=2)).view_as(heavy_neighbourhoods_idx)
        else:
            pooled_shapes = window_aggregated_shapes[:, :, ::2, ::2]
        # output = torch.cat([pooled_features[:, :, :-1, :-1], pooled_shapes], dim=1)
        output = torch.cat([pooled_features[:, :, :pooled_shapes.shape[2], :pooled_shapes.shape[3]], pooled_shapes], dim=1)
        return output

    def extra_repr(self):
        return 'kernel={}, stride={}'.format(self.k, self.stride)


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
        window_shape_query = (shape_windows / (torch.norm(shape_windows, p=1, dim=1, keepdim=True) + 0.001)).unsqueeze(1)
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
