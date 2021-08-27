import time
import numpy as np
from torchvision.models import vgg16
from Model import BaseModel
from LocalConvolution import Conv2dLocal
import torch
from torch.optim import Adam as optimizationFunction
from torch import Tensor
from torchvision import transforms
from torch import nn
from torch.nn import functional
from typing import List, Union, Tuple
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

        self.preprocess = transforms.Compose([
            transforms.Resize(size=(image_height, image_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        default_activation_function = nn.LeakyReLU()

        # <editor-fold desc="Layers">

        # (repetitions, out_channels, kernel, dilation, padding)
        architecture = [(2, 64, 3, 1, 1), "M", (2, 128, 3, 1, 1), "M", (3, 256, 3, 1, 1), "M",
                        (3, 512, 3, 1, 1), (3, 512, 3, 2, 2),
                        (1, 4096, 7, 4, 12), (1, num_labels, 1, 1, 0)]

        layers = []
        in_channels = 3
        for step in architecture:
            if isinstance(step, tuple):
                repetitions, out_channels, kernel, dilation, padding = step
                for _ in range(repetitions):
                    layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel, kernel), dilation=(dilation, dilation),
                                         padding=(padding, padding), stride=(1, 1)),
                               default_activation_function]
                    in_channels = out_channels
            elif isinstance(step, str):
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        self.unary_terms_layers = nn.Sequential(
            *layers,
            nn.UpsamplingBilinear2d(size=(image_height, image_width)),
            nn.Sigmoid(),
        )

        # 12
        self.color_distance_calculator = DPNModel._color_distance_wrapper(channels=3, kernel_size=(50, 50))
        self.local_convolution_layer = Conv2dLocal(in_height=image_height, in_width=image_width, channels=num_labels,
                                                   kernel_size=(50, 50), padding=(25, 25))
        self.local_activation_function = nn.Linear(in_features=num_labels, out_features=num_labels)

        # 13
        self.global_convolution_layer = nn.Conv2d(in_channels=num_labels, out_channels=num_labels * num_mixture_filters,
                                                  kernel_size=(9, 9), padding=(4, 4))
        self.global_activation_function = nn.Linear(in_features=num_labels * num_mixture_filters,
                                                    out_features=num_labels * num_mixture_filters)

        # 14 - Block min block_pooling
        self.block_pooling = BlockMinPooling(kernel_size=num_mixture_filters, stride=1, dim=1)

        # </editor-fold>

        self.reset_parameters()

        self.num_labels = num_labels
        self.image_size = (image_height, image_width)
        self.color_palette = palette

        self.loss_function = nn.NLLLoss(ignore_index=255)
        self.use_cuda = False

        self.count = 0
        self.save_halfway = False

    def reset_parameters(self):
        def random_parameters(model: nn.Module):
            if isinstance(model, (nn.Conv2d, nn.Linear)):
                model.weight.data.normal_(0, 0.01)
                model.bias.data.fill_(0.01)

        self.apply(random_parameters)

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
        unary_output = self.unary_terms_layers(input_tensor)

        if self.save_halfway:
            halfway_prediction = self.prediction_to_image(unary_output)
            halfway_prediction.save(f"Images/halfway_prediction_{self.count}.png")
            self.count += 1

        # return unary_output

        color_weight_tensor = self.color_distance_calculator(input_tensor)
        if self.use_cuda:
            color_weight_tensor = color_weight_tensor.to(device="cuda")

        intermediate_output = self.local_convolution_layer(unary_output, color_weight_tensor).permute(0, 2, 3, 1)
        intermediate_output = self.local_activation_function(intermediate_output).permute(0, 3, 1, 2)

        intermediate_output = self.global_convolution_layer(intermediate_output).permute(0, 2, 3, 1)
        intermediate_output = self.global_activation_function(intermediate_output).permute(0, 3, 1, 2)

        smoothness_output = self.block_pooling(intermediate_output)

        output = torch.log(unary_output) - smoothness_output

        return nn.Softmax(dim=1)(output)

    def fit(self, images: List[Image.Image], labeled_images: List[Image.Image],
            epochs: int = 50, learning_rate: float = 1e-4, batch_size: int = 1):

        labeled_images = [image.resize(self.image_size) for image in labeled_images]
        self.loss_function = nn.NLLLoss(weight=self._calc_weights(labeled_images), ignore_index=255)

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.cuda()
        print(f"Using cuda: {self.use_cuda}")

        optimizer = optimizationFunction(self.parameters(), lr=learning_rate)

        if REPORT.Time:
            start_time = time.time()

        for epoch in range(epochs):

            if REPORT.Time:
                start_epoch_time = time.time()
                print(f"START EPOCH: {epoch}")

            if REPORT.Loss:
                epoch_loss = 0

            start_index = 0
            while start_index < len(images):
                end_index = start_index + batch_size

                images_tensors = []
                labels_tensors = []
                batch_images = images[start_index: end_index]
                batch_labels = labeled_images[start_index: end_index]
                for image, labeled_image in zip(batch_images, batch_labels):
                    images_tensors.append(self.preprocess(image))
                    labels_tensors.append(torch.tensor(np.array(labeled_image), dtype=torch.long))

                stacked_images = torch.stack(images_tensors)
                stacked_labels = torch.stack(labels_tensors)

                stacked_images = stacked_images.to(device="cuda" if self.use_cuda else "cpu")
                stacked_labels = stacked_labels.to(device="cuda" if self.use_cuda else "cpu")

                predictions = self.forward(stacked_images)

                loss = self.loss_function(torch.log(predictions), stacked_labels) / batch_size

                loss.backward()

                optimizer.step()
                self.zero_grad()

                if REPORT.Loss:
                    epoch_loss += loss.item() * batch_size

                start_index = end_index

            if REPORT.Loss:
                print(f"\tLoss: {epoch_loss:.3f}")

            if REPORT.Time:
                end_epoch_time = time.time()
                print(f"EPOCH TIME: {end_epoch_time - start_epoch_time:.3f} sec")

        if REPORT.Time:
            end_time = time.time()
            print(f"TOTAL TIME: {end_time - start_time:.3f} sec")

        self.cpu()
        self.use_cuda = False

    def predict(self, image: Image.Image) -> Tensor:
        tensor = self.preprocess(image)
        tensor = torch.unsqueeze(tensor, 0)

        output = self.forward(tensor)

        return output

    def prediction_to_image(self, prediction: Tensor) -> Image.Image:
        prediction = torch.squeeze(prediction, 0)
        labeled = torch.argmax(prediction, dim=0).type(torch.uint8)

        image = Image.fromarray(labeled.numpy(), mode="P")
        image.putpalette(self.color_palette)
        return image

    def _calc_weights(self, labeled_images: List[Image.Image]) -> Tensor:
        weights = torch.zeros(self.num_labels)
        for image in labeled_images:
            values, counts = np.unique(np.array(image), return_counts=True)
            for value, count in zip(values, counts):
                if value < self.num_labels:
                    weights[value] += count

        mask = weights > 0
        weights[mask] = 1 / weights[mask]
        # weights[~mask] = 1

        return weights

    def calc_intersection_over_union(self, prediction: Union[torch.LongTensor, Image.Image],
                                     target: Union[torch.LongTensor, Image.Image],
                                     include_background: bool = False) -> torch.LongTensor:

        if isinstance(prediction, Image.Image):
            prediction = torch.tensor(np.array(prediction.resize(self.image_size)), dtype=torch.long)
        if isinstance(target, Image.Image):
            target = torch.tensor(np.array(target.resize(self.image_size)), dtype=torch.long)

        start = 0 if include_background else 1
        intersection_over_union_tensor = torch.zeros(self.num_labels, dtype=torch.long)
        for label in range(start, self.num_labels):
            prediction_indicator = prediction == label
            target_indicator = target == label

            intersection = prediction_indicator[target_indicator].sum().item()
            union = prediction_indicator.sum().item() + target_indicator.sum().item() - intersection

            if intersection == 0:
                intersection_over_union_tensor[label] = 0
            elif union == 0:
                intersection_over_union_tensor[label] = -1
            else:
                intersection_over_union_tensor[label] = intersection / union

        return intersection_over_union_tensor

    @staticmethod
    def _calc_color_distance(image_tensor: Tensor, channels: int,
                             kernel_size: Tuple[int, int]) -> Tensor:
        k1, k2 = kernel_size

        top_padding, bottom_padding = k1 // 2, (k1 - 1) // 2
        left_padding, right_padding = k2 // 2, (k2 - 1) // 2

        padded = functional.pad(image_tensor,
                                pad=(top_padding, bottom_padding, left_padding, right_padding),
                                mode="constant", value=0)
        unfolded = torch.nn.Unfold(kernel_size=kernel_size)(padded) \
            .reshape(1, channels, k1 * k2, -1).permute(0, 1, 3, 2)
        distances = torch.pow(image_tensor.view(1, channels, -1, 1) - unfolded, 2).sum(dim=1)

        return distances

    @staticmethod
    def _color_distance_wrapper(channels: int, kernel_size: Tuple[int, int]):
        def wrapped_function(image_tensor: Tensor):
            return DPNModel._calc_color_distance(image_tensor, channels=channels,
                                                 kernel_size=kernel_size)

        return wrapped_function


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
