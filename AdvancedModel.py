from Model import BaseModel
import torch
import torch.nn.functional as functional
import torch.nn as nn
from typing import List
from PIL import Image


class AdvancedModel(BaseModel, nn.Module):
    def __init__(self, num_labels: int = 21, num_mixture_filters: int = 5):
        super(AdvancedModel, self).__init__()
        self.layers = nn.Sequential(
            # 1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),

            # 2
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),

            # 4
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 5
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),

            # 6
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 7
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),

            # 8
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(5, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(5, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(5, 5), stride=(1, 1)),
            nn.ReLU(),

            # 9
            nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=(25, 25), stride=(1, 1)),
            nn.ReLU(),

            # 10
            nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(),

            # 11
            nn.Conv2d(in_channels=4096, out_channels=num_labels, kernel_size=(1, 1), stride=(1, 1)),
            nn.Sigmoid(),

            # 12 - Local convolutional

            # 13
            nn.Conv2d(in_channels=num_labels, out_channels=num_labels * num_mixture_filters,
                      kernel_size=(9, 9), stride=(1, 1)),
            nn.Linear(in_features=num_labels * num_mixture_filters, out_features=num_labels * num_mixture_filters),

            # 14 - block min pooling

            # 15
            nn.AvgPool2d(kernel_size=1, stride=1, divisor_override=1),
            nn.Softmax()
        )

    def fit(self, images: List[Image.Image], labeled_images: List[Image.Image]):
        pass

    def predict(self, image: Image.Image):
        pass
