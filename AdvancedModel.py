from Model import BaseModel
from LocalConvolution import Conv2dLocal

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional
from torch.optim import Adam as optimizationFunction
from torchvision import transforms
from torchvision.models import vgg16

import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import namedtuple
from typing import List, Union, Tuple, Optional, Callable

REPORT_CONSTANTS = namedtuple("REPORT", ["Loss", "Time"])
REPORT = REPORT_CONSTANTS(Loss=True, Time=True)

SAVE_CHECKPOINT = False


class BlockMinPooling(nn.Module):
    def __init__(self, kernel_size: int, dim: int = 1):
        super(BlockMinPooling, self).__init__()

        self.kernel_size = kernel_size
        self.dim = dim

    def forward(self, input_tensor: Tensor) -> Tensor:
        new_shape = (*input_tensor.shape[:self.dim], self.kernel_size, -1, *input_tensor.shape[self.dim + 1:])
        tensors = torch.reshape(input_tensor, new_shape)
        return torch.min(tensors, dim=self.dim).values


class LinearActivation(nn.Module):
    def __init__(self, num_channels: int, height: int, width: int, bias: bool = True):
        super(LinearActivation, self).__init__()
        self.weight = nn.Parameter(torch.empty(1, num_channels, height, width))
        if bias:
            self.bias = nn.Parameter(torch.empty(1, num_channels, height, width))
        else:
            self.bias = None

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            nn.init.xavier_normal_(self.bias.data)

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.weight * input_tensor
        if self.bias is not None:
            return output + self.bias
        return output


class DPNModel(BaseModel, nn.Module):
    NUM_PHASES: int = 4

    def __init__(self, image_height: int, image_width: int,
                 num_labels: int = 21, num_mixture_filters: int = 5, palette=None):
        super(DPNModel, self).__init__()

        self.preprocess = self._preprocess(image_height, image_width)

        activation_function = nn.ReLU()

        # <editor-fold desc="Model Layers">

        # <editor-fold desc="Unary Term Architecture">
        # The architecture of the first 11 groups
        # Each group consist of a convolution or pooling, decided by the first letter (C or M)
        #
        # A convolution is described by: ("C", repetitions, out_channels, kernel, dilation, padding, stride)
        #   repetitions: Number of repeated layers
        #   out_channels: The number of channels after this group
        #   kernel: The size of the kernel
        #   dilation: The dilation step
        #   padding: The padding for the layer
        #   stride: The stride of the kernel
        #
        # A pooling is described by: ("M", kernel, stride)
        #   kernel: The size of the kernel
        #   stride: The stride of the kernel

        # Original architecture
        architecture = [("C", 2, 64, 3, 1, 1, 1), ("M", 2, 2),
                        ("C", 2, 128, 3, 1, 1, 1), ("M", 2, 2),
                        ("C", 3, 256, 3, 1, 1, 1), ("M", 2, 2),
                        ("C", 3, 512, 3, 1, 1, 1),
                        ("C", 3, 512, 3, 2, 2, 1),
                        ("C", 1, 4096, 7, 4, 12, 1),
                        ("C", 1, 4096, 1, 1, 0, 1),
                        ("C", 1, num_labels, 1, 1, 0, 1)]

        architecture = [("C", 2, 64, 7, 1, 3, 1), ("M", 2, 2),
                        ("C", 2, 128, 5, 1, 2, 1), ("M", 2, 2),
                        ("C", 3, 256, 5, 1, 2, 1),  # ("M", 2, 2),
                        ("C", 3, 512, 3, 1, 1, 1),
                        ("C", 3, 512, 3, 2, 2, 1),
                        ("C", 1, 4096, 7, 4, 12, 1),
                        ("C", 1, 4096, 1, 1, 0, 1),
                        ("C", 1, num_labels, 1, 1, 0, 1)]

        layers = []
        in_channels = 3  # Initial number of channels (for RGB images)
        for step in architecture:
            letter, *arguments = step
            if letter == "C":
                repetitions, out_channels, kernel, dilation, padding, stride = arguments
                for index in range(repetitions):
                    # Each convolution layer followed by an activation function
                    layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel, kernel), dilation=(dilation, dilation),
                                         padding=(padding, padding), stride=(stride, stride))]
                    if index < repetitions - 1:
                        layers += [activation_function]
                    in_channels = out_channels
            elif letter == "M":
                kernel, stride = arguments
                layers += [nn.MaxPool2d(kernel_size=(kernel, kernel), stride=(stride, stride))]

        # Groups 1 - 11
        self.unary_terms_layers = nn.Sequential(
            *layers,
            nn.UpsamplingBilinear2d(size=(image_height, image_width)),  # Up-sample to the original size
            # nn.Sigmoid()
        )

        # </editor-fold>

        # Group 12 - Local convolution
        self.color_distance_calculator = DPNModel._color_distance(channels=3, kernel_size=(50, 50))
        self.local_convolution_layer = Conv2dLocal(in_height=image_height, in_width=image_width,
                                                   channels=num_labels, kernel_size=(50, 50))
        self.local_activation_function = LinearActivation(num_labels, image_height, image_width)
        # self.local_activation_function = nn.Linear(in_features=num_labels, out_features=num_labels)

        # Group 13
        temporal_channels = num_labels * num_mixture_filters
        self.global_convolution_layer = nn.Conv2d(in_channels=num_labels, out_channels=temporal_channels,
                                                  kernel_size=(9, 9), padding=(4, 4))
        self.global_activation_function = LinearActivation(temporal_channels, image_height, image_width)
        # self.global_activation_function = nn.Linear(in_features=num_labels * num_mixture_filters,
        #                                             out_features=num_labels * num_mixture_filters)

        # Group 14 - Block min block_pooling
        self.block_pooling = BlockMinPooling(kernel_size=num_mixture_filters, dim=1)

        self.final_activation_function = nn.LogSoftmax(dim=1)

        # </editor-fold>

        # Initialize the parameters of the model
        self.reset_parameters(use_vgg=False)

        self.num_labels = num_labels
        self.num_mixture_filters = num_mixture_filters
        self.image_size = (image_height, image_width)
        self.color_palette = palette

        self.loss_function = nn.NLLLoss(ignore_index=255)
        self.use_cuda = False

    def reset_parameters(self, use_vgg: bool = True):
        """
        Reset the parameters of the model

        :param use_vgg: Whether or not use the pretrained vgg model
        """

        def random_parameters(model: nn.Module):
            if isinstance(model, (nn.Conv2d, nn.Linear)):
                model.weight.data.normal_(0, 1 / torch.numel(model.weight.data) ** 0.5)
                model.bias.data.normal_(0, 1 / torch.numel(model.bias.data) ** 0.5)

        self.apply(random_parameters)  # Initialize the parameters randomly

        self.local_convolution_layer.reset_parameters()
        self.global_activation_function.reset_parameters()

        if use_vgg:
            # Use the pretrained VGG-16 model for the first
            pretrained_vgg = vgg16(pretrained=True)
            pretrained_vgg.train()

            relevant_vgg_parameters_indices = [index for index, layer in enumerate(pretrained_vgg.features)
                                               if isinstance(layer, nn.Conv2d)]
            relevant_dpn_parameters_indices = [index for index, layer in enumerate(self.unary_terms_layers)
                                               if isinstance(layer, nn.Conv2d)]

            for p_i, v_i in zip(relevant_dpn_parameters_indices, relevant_vgg_parameters_indices):
                self.unary_terms_layers[p_i].weight.data = pretrained_vgg.features[v_i].weight.data
                self.unary_terms_layers[p_i].bias.data = pretrained_vgg.features[v_i].bias.data

    def forward(self, input_tensor: Tensor, phase: int = NUM_PHASES) -> Tensor:
        """
        Calculate the forward pass through the model layers

        :param input_tensor: The input for the pass
        :param phase: For training purposes - incremental learning. phase should be between 1 and 4 (included)
        :return: The result of the pass
        """

        # Phase 1: Unary terms
        unary_output = self.unary_terms_layers(input_tensor)

        if phase == 1:
            return self.final_activation_function(unary_output)

        # unary_output = functional.softmax(unary_output, dim=1)

        # Calculate the color distance tensor
        color_distance_tensor = self.color_distance_calculator(input_tensor)

        # Phase 2: Adding triple penalty
        # intermediate_output = self.local_convolution_layer(unary_output, color_distance_tensor).permute(0, 2, 3, 1)
        # intermediate_output = self.local_activation_function(intermediate_output).permute(0, 3, 1, 2)

        intermediate_output = self.local_convolution_layer(unary_output, color_distance_tensor)
        intermediate_output = self.local_activation_function(intermediate_output)

        if phase == 2:
            return self.final_activation_function(intermediate_output)

        # Phase 3: Adding local labels context
        # intermediate_output = self.global_convolution_layer(intermediate_output).permute(0, 2, 3, 1)
        # intermediate_output = self.global_activation_function(intermediate_output).permute(0, 3, 1, 2)

        intermediate_output = self.global_convolution_layer(intermediate_output)
        intermediate_output = self.global_activation_function(intermediate_output)

        smoothness_output = self.block_pooling(intermediate_output)

        if phase == 3:
            return self.final_activation_function(smoothness_output)

        # Phase 4: Combining both unary and smoothness terms
        # output = torch.log(unary_output) - smoothness_output
        output = functional.log_softmax(unary_output, dim=1) - smoothness_output

        return self.final_activation_function(output)

    def fit(self, images: List[Image.Image], labeled_images: List[Image.Image],
            epochs: int = 80, learning_rate: float = 1e-5, batch_size: int = 1, incremental: bool = True,
            output_weight_path: Optional[str] = None, saving_cycle: int = 10, input_weight_path: Optional[str] = None):
        """
        Train the model on the given images

        :param images: The RGB images for the training
        :param labeled_images: The annotated images as ground-truth
        :param epochs: The number of epochs to train
        :param learning_rate: The rate for the optimizer
        :param batch_size: The size of each batch in the training process
        :param incremental: Whether or not to train the model using incremental learning
        :param output_weight_path: The path where the weights will be saved
        :param saving_cycle: Number of epochs before saving the model weights
                (relevant only if output_weight_path is not None)
        :param input_weight_path:
        """

        # Resize the annotated images and calculate the weight for the classes
        labeled_images = [image.resize(self.image_size) for image in labeled_images]
        self.loss_function = nn.NLLLoss(weight=self._calc_weights(labeled_images), ignore_index=255)

        def fit_single_phase(phase: int = self.NUM_PHASES):
            """
            Fit for a single phase\n
            (training only for phase=4 is the same as non incremental learning)

            :param phase: The phase of the training
            """
            if REPORT.Loss:
                loss_list = []
                if output_weight_path is not None:
                    best_loss = float('inf')
                    saving_counter = saving_cycle

            if REPORT.Time:
                if incremental:
                    print(f"START PHASE {phase}")
                start_time = time.time()

            for epoch in range(epochs):

                if REPORT.Loss:
                    epoch_loss = 0

                if REPORT.Time:
                    start_epoch_time = time.time()
                    print(f"\tSTART EPOCH: {epoch}")

                for batch_index, (image_tensor, labels_tensor) in enumerate(zip(images_tensors, labeled_tensors)):
                    prediction = self.forward(image_tensor, phase=phase)

                    loss = self.loss_function(prediction, labels_tensor) / batch_size
                    loss.backward()

                    if (batch_index + 1) % batch_size == 0:
                        optimizer.step()
                        self.zero_grad()

                    if REPORT.Loss:
                        epoch_loss += loss.item() * batch_size

                if REPORT.Loss:
                    loss_list.append(epoch_loss)
                    print(f"\t\tLoss: {epoch_loss:.3f}")

                    if SAVE_CHECKPOINT and output_weight_path is not None:
                        if epoch_loss < best_loss:
                            best_loss = epoch_loss
                            if saving_counter <= 0:
                                print(f"\t\tSaving epoch {epoch} at phase {phase}")
                                torch.save({"epoch": epoch, "phase": phase, "loss": epoch_loss,
                                            "model_state_dict": self.state_dict(),
                                            "optimizer_state_dict": optimizer.state_dict(),
                                            }, f"checkpoint_{output_weight_path}")
                                saving_counter = saving_cycle
                        saving_counter -= 1

                if REPORT.Time:
                    end_epoch_time = time.time()
                    print(f"\t\tTIME: {end_epoch_time - start_epoch_time:.3f} sec")

            if REPORT.Time:
                end_time = time.time()
                print(f"\tTOTAL PHASE TIME: {end_time - start_time:.3f} sec")

            if REPORT.Loss:
                plt.plot(loss_list)
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(f"Loss v.s. Epochs in phase {phase}")
                plt.savefig(f"Loss_vs_Epochs_phase_{phase}.png")
                plt.cla()

        # Move the model to the GPU if possible
        self.use_cuda = torch.cuda.is_available()
        device = "cpu"
        if self.use_cuda:
            self.cuda()
            device = "cuda"
        print(f"Using cuda: {self.use_cuda}")

        images_tensors = [self.preprocess(image).to(device=device) for image in images]
        labeled_tensors = [torch.unsqueeze(torch.tensor(np.array(image), dtype=torch.long, device=device), dim=0)
                           for image in labeled_images]

        optimizer = optimizationFunction(self.parameters(), lr=learning_rate)
        if input_weight_path is not None:
            weights = torch.load(input_weight_path)
            try:
                optimizer.load_state_dict(weights["optimizer_state_dict"])
            except ValueError:
                print("Could not load state_dict for the optimizer")
            try:
                self.load_state_dict(weights["model_state_dict"], strict=False)
            except ValueError:
                print("Could not load state_dict for the model")

        if incremental:
            for phase_index in range(self.NUM_PHASES):
                # Freeze the trained layers
                if phase_index == 0:
                    pass
                if phase_index == 1:
                    self.unary_terms_layers.requires_grad_(False)
                if phase_index == 2:
                    self.local_convolution_layer.requires_grad_(False)
                    self.local_activation_function.requires_grad_(False)
                if phase_index == 3:
                    self.unary_terms_layers.requires_grad_(True)
                    self.local_convolution_layer.requires_grad_(True)
                    self.local_activation_function.requires_grad_(True)

                fit_single_phase(phase=phase_index + 1)
        else:
            fit_single_phase()

        # Remove the model from the GPU
        self.cpu()
        self.use_cuda = False

        # Save the model weights (if requested)
        if output_weight_path is not None:
            print(f"\tSaving weights at {output_weight_path}")
            torch.save({"model_state_dict": self.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()
                        }, output_weight_path)
            print("\tSaved!")

    def predict(self, image: Image.Image, phase: Optional[int] = None) -> Tensor:
        """
        Get the prediction for the given image

        :param image: The image to predict the annotations for
        :param phase: Whether or not to use only part of the network for the prediction
        :return: A probability tensor, for each pixel there is a probability vector over the labels
        """

        if phase is None:
            phase = self.NUM_PHASES

        tensor = self.preprocess(image)

        output = self.forward(tensor, phase=phase)

        return output

    def prediction_to_image(self, prediction: Tensor) -> Image.Image:
        """
        Translate a probability tensor into an image

        :param prediction: The probability tensor (usually the output of predict or forward)
        :return: The image predicted as the most likely
        """
        prediction = torch.squeeze(prediction, 0)
        labeled = torch.argmax(prediction, dim=0).type(torch.uint8)

        image = Image.fromarray(labeled.numpy(), mode="P")
        image.putpalette(self.color_palette)
        return image

    def calc_mean_intersection_over_union(self, prediction: Union[torch.LongTensor, Image.Image],
                                          target: Union[torch.LongTensor, Image.Image],
                                          include_background: bool = False) -> float:
        """

        :param prediction:
        :param target:
        :param include_background:
        :return:
        """
        iou = self.calc_intersection_over_union(prediction, target, include_background)

        mask = iou > 0
        num_relevant_labels = torch.sum(mask)

        if num_relevant_labels == 0:
            return 0

        mean = torch.sum(iou[mask]) / num_relevant_labels
        return mean.item()

    def calc_intersection_over_union(self, prediction: Union[torch.LongTensor, Image.Image],
                                     target: Union[torch.LongTensor, Image.Image],
                                     include_background: bool = False) -> torch.LongTensor:
        """
        Calculate the IoU (Intersection over Union) for the given prediction and target

        :param prediction: The prediction (not as a probability tensor)
        :param target: The target image to compare the prediction
        :param include_background: Whether or not the background should be included in the calculations
        :return: The IoU for each class.<br>
                In case the class does not appear in either the prediction nor the target, the value will be negative
        """

        # Transform the prediction and the target to tensors if necessary
        if isinstance(prediction, Image.Image):
            prediction = torch.tensor(np.array(prediction.resize(self.image_size)), dtype=torch.long)
        if isinstance(target, Image.Image):
            target = torch.tensor(np.array(target.resize(self.image_size)), dtype=torch.long)

        prediction = torch.flatten(prediction)
        target = torch.flatten(target)

        start = 0 if include_background else 1
        intersection_over_union_tensor = torch.zeros(self.num_labels, dtype=torch.float)
        for label in range(start, self.num_labels):
            prediction_indicator = prediction == label
            target_indicator = target == label

            intersection = prediction_indicator[target_indicator].sum().item()
            union = prediction_indicator.sum().item() + target_indicator.sum().item() - intersection

            if union == 0:
                # The IoU is not defined (represents as a negative number)
                intersection_over_union_tensor[label] = -1
            else:
                intersection_over_union_tensor[label] = intersection / union

        return intersection_over_union_tensor

    @staticmethod
    def _preprocess(image_height: int, image_width: int) -> Callable[[Image.Image], Tensor]:
        transformation = transforms.Compose([
            transforms.Resize(size=(image_height, image_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        def wrapped_preprocess(image: Image.Image) -> Tensor:
            return torch.unsqueeze(transformation(image), dim=0)

        return wrapped_preprocess

    def _calc_weights(self, labeled_images: List[Image.Image]) -> Tensor:
        """
        Calculate the weights for the loss function, in order to deal with unbalanced classes

        :param labeled_images: A list of annotated images to calculate the bias over
        :return: A tensor of weights to counteract the bias
        """
        weights = torch.zeros(self.num_labels)
        for image in labeled_images:
            values, counts = np.unique(np.array(image), return_counts=True)
            for value, count in zip(values, counts):
                if value < self.num_labels:
                    weights[value] += count

        mask = weights > 0
        weights[mask] = 1 / weights[mask]

        return weights

    @staticmethod
    def _color_distance(kernel_size: Tuple[int, int], channels: int = 3) -> Callable[[Tensor], Tensor]:
        """
        Get a function to calculate the color distance tensor

        :param kernel_size: The size of the kernel's receptive field
        :param channels: The number of channels in the image
        :return: A function to calculate the color distance tensor
        """

        def wrapped_color_distance(image_tensor: Tensor) -> Tensor:
            """
            Calculate the color distance tensor

            :param image_tensor: The tensor to calculate over
            :return: The color distance of the given tensor
            """
            batch_size = image_tensor.size()[0]
            k1, k2 = kernel_size

            top_padding, bottom_padding = k1 // 2, (k1 - 1) // 2
            left_padding, right_padding = k2 // 2, (k2 - 1) // 2

            padded = functional.pad(image_tensor,
                                    pad=(top_padding, bottom_padding, left_padding, right_padding),
                                    mode="constant", value=0)
            unfolded = torch.nn.Unfold(kernel_size=kernel_size)(padded) \
                .reshape(batch_size, channels, k1 * k2, -1).permute(0, 1, 3, 2)
            distances = torch.pow(image_tensor.view(batch_size, channels, -1, 1) - unfolded, 2).sum(dim=1)

            return distances

        return wrapped_color_distance
