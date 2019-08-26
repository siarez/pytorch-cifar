import torch
from torch import nn
from torch.autograd import Function


# Inherit from Function
class LinearFunctionCustom(Function):
    @staticmethod
    def forward(ctx, input, weight, b_weights, bias=None):
        ctx.save_for_backward(input, weight, b_weights, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # import pydevd
        # pydevd.settrace(suspend=True, trace_only_current_thread=True)
        input, weight, b_weights, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(b_weights)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, torch.zeros_like(b_weights), grad_bias

class LinearCustom(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(LinearCustom, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        # The same initialization as describe in the torch docs
        self.k = torch.sqrt(1/torch.tensor(input_features, dtype=torch.float32))
        # self.weight_bw = torch.randn((output_features, input_features), requires_grad=False) * self.k
        self.weight_bw = (torch.zeros((output_features, input_features), requires_grad=False).uniform_() > 0.7).float()  # random binary
        # self.weight_bw = torch.randint(0, output_features, size=(output_features*2,))  # for the "fast" version. aka gather indices
        self.weight.data = torch.randn((output_features, input_features), requires_grad=True) * self.k
        if bias is not None:
            self.bias.data.uniform_(-1.0*self.k, self.k)

    def forward(self, input):
        return LinearFunctionCustom.apply(input, self.weight, self.weight_bw, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Conv2DFunctionCustom(Function):
    @staticmethod
    def forward(ctx, input, weight, b_weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
        ctx.save_for_backward(input, weight, b_weights, bias)
        ctx.stride, ctx.padding, ctx.dilation, ctx.groups = stride, padding, dilation, groups
        output = torch.nn.functional.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # import pydevd
        # pydevd.settrace(suspend=True, trace_only_current_thread=True)
        input, weight, b_weights, bias = ctx.saved_tensors
        stride, padding, dilation, groups = ctx.stride, ctx.padding, ctx.dilation, ctx.groups
        grad_input = grad_weight = grad_bias = grad_stride = grad_padding = grad_dilation = grad_groups = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, b_weights, grad_output, stride, padding, dilation, groups)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups)

        return grad_input, grad_weight, torch.zeros_like(b_weights), grad_bias, grad_stride, grad_padding, grad_dilation, grad_groups


class Conv2DCustom(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(Conv2DCustom, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
        # The backward weights are created such that a kernel either sends gradients to a previous channel or it doesn't
        # That is, for a kernel(aka output channel) either all backwards weights of an input channel are 0 to all are 1.
        self.weight_bw = (torch.zeros((out_channels, in_channels//groups, 1), requires_grad=False).uniform_() > 0.5).float()  # random binary
        self.weight_bw = self.weight_bw.expand(-1, -1, kernel_size**2).view(out_channels, in_channels//groups, kernel_size, kernel_size)

    def forward(self, x):
        return Conv2DFunctionCustom.apply(x, self.weight, self.weight_bw, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_channels, self.out_channels, self.kernel_size
        )
