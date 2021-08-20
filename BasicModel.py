from typing import List, Dict
from Model import BaseModel
from PIL import Image
import random
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
    def __init__(self, beta=10000):
        self.beta = beta  # doubleton potential
        self.neighbors_indices = [[0, 1], [0, -1], [1, 0], [-1, 0]] # neighbors included in energy calculation
        self.class_info = {}

    def pixel_energy(self, i, j, image, interpretation):
        energy = 0
        for color in range(3):
            mean = self.class_info[interpretation[i, j]][1][color]
            var = self.class_info[interpretation[i, j]][2][color]
            value = image[i][j][color]
            energy += np.log(np.sqrt(2 * np.pi * var)) + ((value - mean) ** 2) / (2 * var)
        for a, b in self.neighbors_indices:
            a += j
            b += j
            if 0 <= a < image.shape[0] and 0 <= b < image.shape[1]:
                energy += self.beta * (-1 if interpretation[i][j] == interpretation[a][b] else 1)
        return energy

    def calculate_energy(self, image, interpretation):
        energy = 0
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                energy += self.pixel_energy(i, j, image, interpretation)
        return energy

    def delta_energy(self, image, interpretation, pixel, new_class):
        initial_energy = self.pixel_energy(*pixel, image, interpretation)
        interpretation[pixel[0]][pixel[1]] = new_class
        new_energy = self.pixel_energy(*pixel, image, interpretation)
        print("initial energy " + str(initial_energy))
        print("new_energy " + str(new_energy))
        return new_energy - initial_energy

    def simulated_annealing(self, image, interpretation, iterations, initial_temp=1000):
        current_energy = self.calculate_energy(image, interpretation)
        current_tmp = initial_temp
        iteration = 0
        while iteration < iterations:
            i = random.randint(0, image.shape[0] - 1)
            j = random.randint(0, image.shape[1] - 1)

            possible_classes = list(self.class_info.keys())
            possible_classes.remove(interpretation[i, j])
            new_class = random.choice(possible_classes)
            delta = self.delta_energy(image, interpretation, (i, j), new_class)

            if delta <= 0:
                interpretation[i][j] = new_class
                current_energy += delta
                print("CHANGED better")
                try:
                    print("class " + str(new_class))
                    print("neighbors " + str(interpretation[i+1][j]) + " " + str(interpretation[i-1][j]))
                    print("neighbors " + str(interpretation[i][j+1]) + " " + str(interpretation[i][j-1]))
                except IndexError:
                    pass
            else:
                if current_tmp == 0 or -delta / current_tmp < -600:
                    k = 0
                else:
                    k = np.exp(-delta / current_tmp)
                r = random.uniform(0, 1)
                if r < k:
                    print("CHANGED worse")
                    interpretation[i][j] = new_class
                    current_energy += delta

            current_tmp /= (1 + iteration)  # linear cooling - possible to try logarithmic cooling
            iteration += 1

        return interpretation

    def fit(self, images: List[Image.Image], labeled_images: List[Image.Image]):
        # initial probabilities from training data (prior)
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

    def naive_bayes_prediction(self, image: np.ndarray):
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

    def predict(self, image: Image.Image):
        image = np.array(image)
        initial_interpretation = self.naive_bayes_prediction(image)
        return self.simulated_annealing(image, initial_interpretation, iterations=100000)


