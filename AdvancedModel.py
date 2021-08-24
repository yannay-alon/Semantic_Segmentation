import time
import numpy as np
from torchvision.models import vgg16
from Model import BaseModel
from LocalConvolution import Conv2dLocal
import torch
from torch.optim import Adam as optimizationFunction
from torch import Tensor
import torch.nn as nn
from typing import List, Dict
from PIL import Image
from collections import namedtuple

REPORT_CONSTANTS = namedtuple("REPORT", ["Loss", "Time"])
REPORT = REPORT_CONSTANTS(Loss=True, Time=True)


class BlockMinPooling(nn.Module):
    # REMARK: Only implemented for 1D kernel and stride=1
    def __init__(self, kernel_size: int, stride: int, dim: int = 1):
        super(BlockMinPooling, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.dim = dim

    def forward(self, input_tensor: Tensor) -> Tensor:
        new_shape = (*input_tensor.shape[:self.dim], self.kernel_size, -1, *input_tensor.shape[self.dim + 1:])
        tensors = torch.reshape(input_tensor, new_shape)
        return torch.min(tensors, dim=self.dim).values


class DPNModel(BaseModel, nn.Module):
    def __init__(self, image_height: int, image_width: int,
                 num_labels: int = 21, num_mixture_filters: int = 5, palette=None):
        super(DPNModel, self).__init__()

        # <editor-fold desc="Layers">

        self.unary_terms_layers = nn.Sequential(
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
        )

        # 12 - Local convolutional
        self.local_convolution_layer = Conv2dLocal(in_height=image_height, in_width=image_width, channels=num_labels,
                                                   kernel_size=(50, 50), stride=(1, 1), padding=(25, 25))
        self.local_activation_function = nn.Linear(in_features=num_labels, out_features=num_labels)

        # 13
        self.global_convolution_layer = nn.Conv2d(in_channels=num_labels, out_channels=num_labels * num_mixture_filters,
                                                  kernel_size=(9, 9), stride=(1, 1), padding=(4, 4))
        self.global_activation_function = nn.Linear(in_features=num_labels * num_mixture_filters,
                                                    out_features=num_labels * num_mixture_filters)

        self.pooling = nn.Sequential(
            # 14 - Block min pooling
            BlockMinPooling(kernel_size=num_mixture_filters, stride=1, dim=1),

            # 15 - Sum pooling (same as average with divisor=1)
            nn.AvgPool2d(kernel_size=1, stride=1, divisor_override=1),
            nn.Softmax(dim=1)
        )

        # </editor-fold>

        self.reset_parameters()

        self.color_palette = palette

        weights = torch.tensor([1] + [10] * 20, dtype=torch.float)
        self.loss_function = nn.CrossEntropyLoss(weight=weights, ignore_index=255)
        self.use_cuda = False

    def reset_parameters(self):
        def random_parameters(model: nn.Module):
            if isinstance(model, (nn.Conv2d, nn.Linear)):
                model.weight.data.normal_(0.0, 0.1)
                model.bias.data.fill_(0)

        # self.apply(random_parameters)

        pretrained_vgg = vgg16(pretrained=True)
        pretrained_vgg.train()
        relevant_vgg_parameters_indices = [index for index, layer in enumerate(pretrained_vgg.features)
                                           if isinstance(layer, nn.Conv2d)]
        relevant_dpn_parameters_indices = [index for index, layer in enumerate(self.unary_terms_layers)
                                           if isinstance(layer, nn.Conv2d)]

        for p_i, v_i in zip(relevant_dpn_parameters_indices, relevant_vgg_parameters_indices):
            self.unary_terms_layers[p_i].weight.data = pretrained_vgg.features[v_i].weight.data
            self.unary_terms_layers[p_i].bias.data = pretrained_vgg.features[v_i].bias.data
            # self.unary_terms_layers[p_i].requires_grad_(False)

    def forward(self, input_tensor: Tensor) -> Tensor:
        if self.use_cuda:
            input_tensor = input_tensor.to(device="cuda")
        intermediate_output = self.unary_terms_layers(input_tensor)

        intermediate_output = self.local_convolution_layer(intermediate_output).permute(0, 2, 3, 1)
        intermediate_output = self.local_activation_function(intermediate_output).permute(0, 3, 1, 2)

        intermediate_output = self.global_convolution_layer(intermediate_output).permute(0, 2, 3, 1)
        intermediate_output = self.global_activation_function(intermediate_output).permute(0, 3, 1, 2)

        output = self.pooling(intermediate_output)

        return output

    def fit(self, images: List[Image.Image], labeled_images: List[Image.Image],
            epochs: int = 10, learning_rate: float = 1e-2, batch_size: int = 5):

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.cuda()
        print(f"Using cuda: {self.use_cuda}")

        optimizer = optimizationFunction(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):

            if REPORT.Time:
                start_epoch_time = time.time()
                print(f"START EPOCH: {epoch}")

            if REPORT.Loss:
                epoch_loss = 0

            for index, (image, labeled_image) in enumerate(zip(images, labeled_images)):
                prediction = self.predict(image)

                labeled_tensor = torch.tensor(np.array(labeled_image), dtype=torch.long,
                                              device="cuda" if self.use_cuda else "cpu")
                labeled_tensor = torch.unsqueeze(labeled_tensor, 0)

                loss = self.loss_function(prediction, labeled_tensor) / batch_size
                loss.backward()

                if REPORT.Loss:
                    epoch_loss += loss.item()

                if (index + 1) % batch_size == 0:
                    optimizer.step()
                    self.zero_grad()

            if REPORT.Loss:
                print(f"\tLoss: {epoch_loss}")

            if REPORT.Time:
                end_epoch_time = time.time()
                print(f"EPOCH TIME: {end_epoch_time - start_epoch_time}")

        self.cpu()
        self.use_cuda = False

    def predict(self, image: Image.Image) -> Tensor:
        tensor = torch.tensor(np.array(image), dtype=torch.float)
        tensor = tensor.permute(2, 0, 1)
        tensor = torch.unsqueeze(tensor, 0)

        output = self.forward(tensor)

        return output

    def prediction_to_image(self, prediction: Tensor) -> Image.Image:
        prediction = torch.squeeze(prediction, 0)
        labeled = torch.argmax(prediction, dim=0).type(torch.uint8)

        image = Image.fromarray(labeled.numpy(), mode="P")
        image.putpalette(self.color_palette)
        return image


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
