import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch import nn
from torch.nn import functional
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


_pair = _n_tuple(2)

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

        self.color_weight = Parameter(torch.zeros(1, dtype=torch.float))
        self.position_weight = Parameter(torch.zeros(1, dtype=torch.float))

        height_pos = torch.arange(self.kernel_size[0]).reshape(1, -1)
        width_pos = torch.arange(self.kernel_size[1]).reshape(1, -1)
        height_distance = (height_pos - self.kernel_size[0] // 2) ** 2
        width_distance = (width_pos - self.kernel_size[1] // 2) ** 2
        position_distances_tensor = (height_distance + width_distance.T).reshape(-1)

        self.register_buffer("position_distances", position_distances_tensor)

        self.reset_parameters()

    def reset_parameters(self):
        self.position_weight.data = torch.tensor(0.01, dtype=torch.float)
        self.color_weight.data = torch.tensor(0.01, dtype=torch.float)

    def forward(self, input_tensor: torch.Tensor, color_distance_tensor: torch.Tensor = 0):
        batch_size = input_tensor.size()[0]
        kernel_height, kernel_width = self.kernel_size

        top_padding, bottom_padding = kernel_height // 2, (kernel_height - 1) // 2
        left_padding, right_padding = kernel_width // 2, (kernel_width - 1) // 2

        padded = functional.pad(input_tensor,
                                pad=(top_padding, bottom_padding, left_padding, right_padding),
                                mode="constant", value=0)

        unfolded_data = nn.Unfold(kernel_size=self.kernel_size)(padded)
        unfolded_data = unfolded_data.permute(0, 2, 1) \
            .reshape(batch_size, self.channels, kernel_height * kernel_width, -1).permute(0, 1, 3, 2)

        weights = self.position_distances * self.position_weight + color_distance_tensor * self.color_weight

        output = (unfolded_data * weights).sum(-1) \
            .view(batch_size, self.channels, self.in_height, self.in_width)

        return output * input_tensor
