import torch
import io
import random
import tarfile
import numpy as np
from PIL import Image
from PIL.ImageOps import invert
from datetime import datetime
from typing import Union, List, Optional

from BasicModel import MRFModel
from AdvancedModel import DPNModel

VOC_TAR_PATH = "VOCtrainval_11-May-2012"


# <editor-fold desc="Read from VOC tar">
def get_images(tar: tarfile.TarFile, image_names: Union[List[str], str],
               annotations: Optional[str] = None) -> Union[List[Image.Image], Image.Image]:
    """
        Get the image(s) from the tar.\n
        The images return as a list if image_names is given as a list

        :param tar: The tar file which holds the image
        :param image_names: The name or list of names of the image(s)
        :param annotations: The type of annotations of the image (either 'class', 'object' or None for unannotated)
        :return: The image as an object or list of images
        """
    if annotations is None:
        image_directory = "VOCdevkit/VOC2012/JPEGImages"
        suffix = "jpg"
    elif annotations == "class":
        image_directory = "VOCdevkit/VOC2012/SegmentationClass"
        suffix = "png"
    elif annotations == "object":
        image_directory = "VOCdevkit/VOC2012/SegmentationObject"
        suffix = "png"
    else:
        raise ValueError("annotations must be 'class', 'objet' or None")

    if isinstance(image_names, str):
        image_data = tar.extractfile(f"{image_directory}/{image_names}.{suffix}").read()
        image = Image.open(io.BytesIO(image_data))
        return image

    images = []
    for image_name in image_names:
        if image_name == "":
            print("Empty")
        image_data = tar.extractfile(f"{image_directory}/{image_name}.{suffix}").read()
        images.append(Image.open(io.BytesIO(image_data)))
    return images


def get_dataset_paths(tar: tarfile.TarFile, dataset: str) -> List[str]:
    """
    Get the names of the images for the requested dataset

    :param tar: The tar object which holds the images
    :param dataset: Which dataset to get - one of [train, val]
    :return: A list of the requested images' names
    """

    directory = "VOCdevkit/VOC2012/ImageSets/Segmentation"
    file_path = f"{directory}/{dataset}.txt"
    paths = tar.extractfile(file_path).read().decode("utf-8").split("\n")[:-1]
    return paths


# </editor-fold>


def voc_images():
    tar = tarfile.open(f"{VOC_TAR_PATH}.tar")

    train_file_paths = get_dataset_paths(tar, "train")
    val_file_paths = get_dataset_paths(tar, "val")

    image_name = val_file_paths[:10]

    annotated_images = get_images(tar, image_name, "class")
    images = get_images(tar, image_name)

    images = [image.resize((64, 64)) for image in images]
    for index, image in enumerate(images):
        image.save(f"Images/true_{index}.jpg")


def mrf():
    tar = tarfile.open(f"{VOC_TAR_PATH}.tar")
    train_file_paths = get_dataset_paths(tar, "train")

    image_name = train_file_paths[:10]

    annotated_image = get_images(tar, image_name, "class")
    images = get_images(tar, image_name)

    palette = annotated_image[0].getpalette()

    MRF = MRFModel(palette)
    MRF.fit(images, annotated_image)

    for image in images:
        predicted = MRF.naive_bayes_prediction(np.array(image))

        predicted_image = Image.fromarray(predicted.astype(np.uint8), mode="P")
        predicted_image.putpalette(palette)
        predicted_image.show()


def dpn():
    Train = False
    load_weights_for_train = True
    Test = True
    test_on_val = True

    num_images_train = 50
    num_images_test = 50

    tar = tarfile.open(f"{VOC_TAR_PATH}.tar")
    train_file_paths = get_dataset_paths(tar, "train")
    validation_file_paths = get_dataset_paths(tar, "val")

    train_images_names = train_file_paths[:num_images_train]
    # random.shuffle(train_images_names)
    train_images = get_images(tar, train_images_names)
    train_annotated_images = get_images(tar, train_images_names, annotations="class")

    print(f"Number of training images: {len(train_images_names)}")

    val_images_names = validation_file_paths[:num_images_test]
    # random.shuffle(val_images_names)
    val_images = get_images(tar, val_images_names)
    val_annotated_images = get_images(tar, val_images_names, annotations="class")

    print(f"Number of validation images: {len(val_images_names)}")

    size = (64, 64)

    model = DPNModel(*size, num_labels=21, palette=train_annotated_images[0].getpalette())

    output_weights = "weights.pt"

    if Train:
        input_weights = None
        if load_weights_for_train:
            input_weights = "weights.pt"
        model.train()
        model.fit(train_images, train_annotated_images, epochs=40, batch_size=1, incremental=True,
                  output_weight_path=output_weights, input_weight_path=input_weights)
    elif load_weights_for_train:
        model.load_state_dict(torch.load(output_weights)["model_state_dict"], strict=False)

    if Test:
        model.eval()
        total_miou = 0
        with torch.no_grad():
            images_for_test = val_images if test_on_val else train_images
            annotated_for_test = val_annotated_images if test_on_val else train_annotated_images
            class_counter_tensor = torch.zeros(21)
            for index, (image, annotated_image) in enumerate(zip(images_for_test[:num_images_test],
                                                                 annotated_for_test[:num_images_test])):
                output = model.predict(image)
                predicted_image = model.prediction_to_image(output)
                image.resize(size).save(f"Images/true_{index}.jpg")
                predicted_image.save(f"Images/predicted_{index}.png")
                annotated_image.resize(size).save(f"Images/annotated_{index}.png")
                miou = model.calc_intersection_over_union(predicted_image, annotated_image,
                                                          include_background=False)
                mask = miou >= 0
                class_counter_tensor += mask
                total_miou += miou * mask
                # print(f"Image {index}: mIoU = {miou:.3f}")
            print(f"Average mIoU = {total_miou / class_counter_tensor}")


def test_transformation(transformation):
    test_on_val = True
    num_images = 50

    tar = tarfile.open(f"{VOC_TAR_PATH}.tar")
    train_file_paths = get_dataset_paths(tar, "train")
    validation_file_paths = get_dataset_paths(tar, "val")

    train_images_names = train_file_paths[:num_images]
    # random.shuffle(train_images_names)
    train_images = get_images(tar, train_images_names)
    train_annotated_images = get_images(tar, train_images_names, annotations="class")

    print(f"Number of training images: {len(train_images_names)}")

    val_images_names = validation_file_paths[:num_images]
    # random.shuffle(val_images_names)
    val_images = get_images(tar, val_images_names)
    val_annotated_images = get_images(tar, val_images_names, annotations="class")

    print(f"Number of validation images: {len(val_images_names)}")

    size = (64, 64)

    model = DPNModel(*size, num_labels=21, palette=train_annotated_images[0].getpalette())

    output_weights = "weights.pt"

    model.load_state_dict(torch.load(output_weights)["model_state_dict"], strict=False)

    model.eval()
    total_miou = 0
    with torch.no_grad():
        images_for_test = val_images if test_on_val else train_images
        annotated_for_test = val_annotated_images if test_on_val else train_annotated_images
        for index, (image, annotated_image) in enumerate(zip(images_for_test[:num_images],
                                                             annotated_for_test[:num_images])):
            if transformation == "gray":
                transformed_image = image.convert("LA").convert("RGB")
            elif transformation == "invert":
                transformed_image = invert(image)
            output = model.predict(transformed_image)
            predicted_image = model.prediction_to_image(output)
            transformed_image.resize(size).save(f"Images/true_{index}.jpg")
            predicted_image.save(f"Images/predicted_{index}.png")
            annotated_image.resize(size).save(f"Images/annotated_{index}.png")
            miou = model.calc_mean_intersection_over_union(predicted_image, annotated_image)

            print(f"Image {index}: mIoU = {miou:.3f}")
            total_miou += miou
        print(f"Average mIoU = {total_miou / num_images}")


def test_image():
    tar = tarfile.open(f"{VOC_TAR_PATH}.tar")
    train_file_paths = get_dataset_paths(tar, "train")
    validation_file_paths = get_dataset_paths(tar, "val")

    train_images_names = train_file_paths[:1]
    # random.shuffle(train_images_names)
    train_images = get_images(tar, train_images_names)
    train_annotated_images = get_images(tar, train_images_names, annotations="class")

    image = Image.open("img151136.png").convert("RGB")
    model = DPNModel(64, 64, num_labels=21, palette=train_annotated_images[0].getpalette())
    output_weights = "weights.pt"
    model.load_state_dict(torch.load(output_weights)["model_state_dict"], strict=False)
    model.eval()
    output = model.predict(image)
    prediction = model.prediction_to_image(output)
    prediction.save("Images/Test_Robin.png")


def main():
    # mrf()
    # dpn()

    test_transformation("gray")
    # test_transformation("invert")

    # voc_images()
    # test_image()


if __name__ == '__main__':
    print(f"Start time: {datetime.now():%H:%M:%S}")
    main()
    print(f"End time: {datetime.now():%H:%M:%S}")
    print("DONE!")
