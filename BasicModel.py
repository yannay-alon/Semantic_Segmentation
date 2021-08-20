from typing import List, Dict

from Model import BaseModel
from PIL import Image
import numpy as np


def get_matching_pixels(colored_image: Image.Image, labeled_image: Image.Image) -> dict:
    """
    Get all pixels_class matching to each class in the image

    :param colored_image: image to be scraped
    :param labeled_image: same image - labeled
    :return: A dictionary with matching pixel for each class in the image
    """
    colored_image = np.array(colored_image)
    labeled_image = np.array(labeled_image)
    pixels_class = {}

    for pixel_index, pixel_color in np.ndenumerate(labeled_image):
        colored_pixel_value = colored_image[pixel_index].tolist()
        if pixel_color in pixels_class:
            pixels_class[pixel_color].append(colored_pixel_value)
        else:
            pixels_class[pixel_color] = [colored_pixel_value]

    return pixels_class


def get_total_matching_pixels(images: List[Image.Image], labeled_images: List[Image.Image]) -> dict:
    assert len(images) == len(labeled_images), "number of images is mismatched"
    total_pixel_class = {}
    for image, labeled_image in zip(images, labeled_images):
        pixel_class = get_matching_pixels(image, labeled_image)
        for label in pixel_class:
            if label in total_pixel_class:
                total_pixel_class[label].extend(pixel_class[label])
            else:
                total_pixel_class[label] = pixel_class[label]
    return total_pixel_class


def normal_pdf(x, mean, var):
    return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))


class MRFModel(BaseModel):
    def __init__(self, beta=1):
        self.beta = beta  # doubleton potential
        self.class_info = {}

    def fit(self, images: List[Image.Image], labeled_images: List[Image.Image]):
        # initial probabilities from training data
        # for each color

        class_pixels = get_total_matching_pixels(images, labeled_images)
        total_number_of_pixels = sum(len(number_pixels) for number_pixels in class_pixels.values())

        for label in class_pixels:
            pixels = np.array(class_pixels[label])
            class_mean = np.mean(pixels, 1)
            class_var = np.var(pixels, 1)
            class_freq = len(pixels)
            class_prior_prob = class_freq / total_number_of_pixels

            self.class_info[label] = (class_prior_prob, class_mean, class_var)

    def predict(self, image: Image.Image):
        image = np.array(image)
        predict_image = np.zeros(image.shape[:-1])
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                max_prob = 0
                best_class = 0

                for cls in self.class_info:
                    cls_posterior = self.class_info[cls][0]
                    for color in range(3):
                        value = image[i][j][color]
                        mean = self.class_info[cls][1][color]
                        var = self.class_info[cls][2][color]
                        likelihood = normal_pdf(value, mean, var)
                        cls_posterior *= likelihood

                    if cls_posterior > max_prob:
                        max_prob = cls_posterior
                        best_class = cls

                predict_image[i][j] = best_class

        return predict_image
