import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional
import torch.nn as nn
from itertools import repeat
from typing import Tuple, Union


# <editor-fold desc="Type helpers">
def _n_tuple(n):
    def parse(x):
        if isinstance(x, Tuple):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _n_tuple(1)
_pair = _n_tuple(2)
_triple = _n_tuple(3)
_quadruple = _n_tuple(4)

_one_or_more = Union[int, Tuple[int, ...]]
_one_or_two = Union[int, Tuple[int, int]]


# </editor-fold>

class Conv2dLocal(nn.Module):

    def __init__(self, in_height: int, in_width: int, in_channels: int, out_channels: int,
                 kernel_size: _one_or_two, stride: _one_or_two = 1, padding: _one_or_two = 0,
                 bias: bool = True, dilation: _one_or_two = 1):
        super(Conv2dLocal, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        self.in_height = in_height
        self.in_width = in_width
        self.out_height = (in_height + 2 * self.padding[0] -
                           self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        self.out_width = (in_height + 2 * self.padding[1] -
                          self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        self.weight = Parameter(torch.Tensor(
            self.out_height, self.out_width,
            out_channels, in_channels, *self.kernel_size))

        if bias:
            self.bias = Parameter(torch.Tensor(
                out_channels, self.out_height, self.out_width))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stddev = 1 / np.sqrt(n)
        self.weight.data.uniform_(-stddev, stddev)
        if self.bias is not None:
            self.bias.data.uniform_(-stddev, stddev)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, input_tensor: torch.Tensor):
        return conv2d_local(input_tensor, self.weight, self.bias, padding=self.padding, stride=self.stride,
                            dilation=self.dilation)


def conv2d_local(input_tensor: torch.Tensor, weight: Parameter, bias: Parameter = None,
                 padding: _one_or_two = 0, stride: _one_or_two = 1, dilation: _one_or_two = 1):
    if input_tensor.dim() != 4:
        raise NotImplementedError("Input Error: Only 4D input Tensors supported (got {}D)".format(input_tensor.dim()))
    if weight.dim() != 6:
        # outH x outW x outC x inC x kH x kW
        raise NotImplementedError("Input Error: Only 6D weight Tensors supported (got {}D)".format(weight.dim()))

    outH, outW, outC, inC, kH, kW = weight.size()
    kernel_size = (kH, kW)

    # N x [inC * kH * kW] x [outH * outW]
    cols = functional.unfold(input_tensor, kernel_size, dilation=dilation, padding=padding, stride=stride)
    cols = cols.view(cols.size()[0], cols.size()[1], cols.size()[2], 1).permute(0, 2, 3, 1)

    out = torch.matmul(cols, weight.view(outH * outW, outC, inC * kH * kW).permute(0, 2, 1))
    out = out.view(cols.size(0), outH, outW, outC).permute(0, 3, 1, 2)

    if bias is not None:
        out = out + bias.expand_as(out)
    return out
