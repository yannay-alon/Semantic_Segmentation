from Model import BaseModel
from LocalConvolution import Conv2dLocal
import torch
from torch import Tensor
import torch.nn as nn
from typing import List
from PIL import Image


class BlockMinPooling(nn.Module):
    # REMARK: Only implemented for 1D kernel and stride=1
    def __init__(self, kernel_size: int, stride: int, dim: int = -1):
        super(BlockMinPooling, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.dim = dim

    def forward(self, input_tensor: Tensor) -> Tensor:
        new_shape = (*input_tensor.shape[:self.dim], self.kernel_size, -1, *input_tensor.shape[self.dim + 1:])
        tensors = torch.reshape(input_tensor, new_shape)
        return torch.min(tensors, dim=self.dim).values


class AdvancedModel(BaseModel, nn.Module):
    def __init__(self, image_height: int = 512, image_width: int = 512,
                 num_labels: int = 21, num_mixture_filters: int = 5):
        super(AdvancedModel, self).__init__()
        self.layers = nn.Sequential(
            # 1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),

            # 2
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),

            # 4
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 5
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),

            # 6
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 7
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),

            # 8
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1),
                      dilation=(2, 2), padding=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1),
                      dilation=(2, 2), padding=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1),
                      dilation=(2, 2), padding=(2, 2)),
            nn.ReLU(),

            # 9
            nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=(7, 7), stride=(1, 1),
                      dilation=(4, 4), padding=(12, 12)),
            nn.ReLU(),

            # 10
            nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(),

            # 11
            nn.Conv2d(in_channels=4096, out_channels=num_labels, kernel_size=(1, 1), stride=(1, 1)),
            nn.UpsamplingBilinear2d(size=(image_height, image_width)),
            nn.Sigmoid(),

            # 12 - Local convolutional
            Conv2dLocal(in_height=image_height, in_width=image_width, in_channels=num_labels, out_channels=num_labels,
                        kernel_size=(50, 50), stride=(1, 1), padding=(25, 25)),
            nn.Linear(in_features=num_labels, out_features=num_labels),

            # 13
            nn.Conv3d(in_channels=num_labels, out_channels=num_labels * num_mixture_filters,
                      kernel_size=(9, 9, num_labels), stride=(1, 1), padding=(4, 4, num_labels // 2)),
            nn.Linear(in_features=num_labels * num_mixture_filters, out_features=num_labels * num_mixture_filters),

            # 14 - Block min pooling
            BlockMinPooling(kernel_size=num_mixture_filters, stride=1),

            # 15 - Sum pooling (same as average with divisor=1)
            nn.AvgPool2d(kernel_size=1, stride=1, divisor_override=1),
            nn.Softmax()
        )

    def fit(self, images: List[Image.Image], labeled_images: List[Image.Image]):
        pass

    def predict(self, image: Image.Image):
        pass


def test():
    tensor = torch.tensor([
        [
            [0, 1, -1, -2, 1, 2],
            [2, 3, -3, -4, 3, 4],
            [4, 5, -5, -6, 5, 6],
            [-4, -5, 5, 6, -5, -6]
        ],
        [
            [6, 7, -7, -8, 7, 8],
            [8, 9, -9, -10, 9, 10],
            [10, 11, -11, -12, 11, 12],
            [-10, -11, 11, 12, -11, -12]
        ],
        [
            [12, 13, -13, -14, 13, 14],
            [14, 15, -15, -16, 15, 16],
            [16, 17, -17, -18, 17, 18],
            [-16, -17, 17, 18, -17, -18]
        ]
    ])

    print(f"Tensor shape: {tensor}")


if __name__ == '__main__':
    test()
