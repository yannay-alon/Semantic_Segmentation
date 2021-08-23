import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional
import torch.nn as nn
from itertools import repeat
from typing import Tuple, Union


# <editor-fold desc="Type helpers">
def _n_tuple(n):
    def parse(x, name: str = ""):
        if isinstance(x, Tuple):
            if len(x) != n:
                raise ValueError(f"Argument {name} must be of length {n}")
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

    def __init__(self, in_height: int, in_width: int, channels: int,
                 kernel_size: _one_or_two, stride: _one_or_two = 1, padding: _one_or_two = 0):
        # REMARK: only implemented for stride = 1

        super(Conv2dLocal, self).__init__()

        self.channels = channels
        self.kernel_size = _pair(kernel_size, "kernel_size")
        self.stride = _pair(stride, "stride")
        self.padding = _pair(padding, "padding")

        self.in_height = in_height
        self.in_width = in_width

        self.out_height = in_height + 2 * self.padding[0] - self.kernel_size[0] + 1
        self.out_width = in_width + 2 * self.padding[1] - self.kernel_size[1] + 1

        self.weights = Parameter(torch.Tensor(
            self.out_height * self.out_width,
            self.kernel_size[0] * self.kernel_size[1]
        ))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.channels
        for k in self.kernel_size:
            n *= k
        stddev = 1 / np.sqrt(n)
        self.weights.data.uniform_(-stddev, stddev)

    def forward(self, input_tensor: torch.Tensor):
        kernel_height, kernel_width = self.kernel_size

        unfolded_data = nn.Unfold(kernel_size=self.kernel_size, padding=self.padding)(input_tensor)
        unfolded_data = unfolded_data.permute(0, 2, 1) \
            .reshape(1, self.channels, self.out_height * self.out_width, kernel_height * kernel_width)

        output = (unfolded_data * self.weights).sum(-1) \
            .view(1, self.channels, self.out_height, self.out_width)

        return output[:, :, :self.in_height, :self.in_width]
