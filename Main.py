from typing import List, Optional
from PIL import Image
import tarfile
import io
import re
import numpy as np
import BasicModel

VOC_TAR_PATH = "VOCtrainval_11-May-2012"


# <editor-fold desc="Read from VOC tar">
def get_image(tar: tarfile.TarFile, image_name: str, annotations: Optional[str] = None) -> Image.Image:
    """
    Get the image from the tar

    :param tar: The tar file which holds the image
    :param image_name: The name of the image
    :param annotations: The type of annotations of the image (either 'class', 'object' or None for unannotated)
    :return: The image as an object
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

    image_data = tar.extractfile(f"{image_directory}/{image_name}.{suffix}").read()
    image = Image.open(io.BytesIO(image_data))
    return image


def get_dataset_paths(tar: tarfile.TarFile, dataset: str) -> List[str]:
    """
    Get the names of the images for the requested dataset

    :param tar: The tar object which holds the images
    :param dataset: Which dataset to get - one of [train, val]
    :return: A list of the requested images' names
    """

    directory = "VOCdevkit/VOC2012/ImageSets/Segmentation"
    file_path = f"{directory}/{dataset}.txt"
    paths = tar.extractfile(file_path).read().decode("utf-8").split("\n")
    return paths


def get_annotations(tar: tarfile.TarFile, image_name: str) -> List[dict]:
    """
    Get the annotation for the required image

    :param tar: The tar object which holds the annotation
    :param image_name: The name of the image
    :return: A dictionary with the annotations data
    """
    annotation_path = f"VOCdevkit/VOC2012/Annotations/{image_name}.xml"
    annotation_data = tar.extractfile(annotation_path).read().decode("utf-8")
    objects_strings = re.findall(r"<object>(.*?)</object>", annotation_data, flags=re.DOTALL)

    annotations = []
    properties = [("name", str, "class"),
                  ("xmin", int, "x_min"), ("xmax", int, "x_max"),
                  ("ymin", int, "y_min"), ("ymax", int, "y_max")]

    for object_string in objects_strings:
        annotation = dict()

        for property_name, property_type, property_key in properties:
            regex_string = rf"<{property_name}>(.*?)</{property_name}>"
            property_value = property_type(re.findall(regex_string, object_string, flags=re.DOTALL)[0])
            annotation[property_key] = property_value

        annotations.append(annotation)
    return annotations


# </editor-fold>


def main():
    tar = tarfile.open(f"{VOC_TAR_PATH}.tar")

    train_file_paths = get_dataset_paths(tar, "train")
    val_file_paths = get_dataset_paths(tar, "val")

    image_name = train_file_paths[0]
    annotations = get_annotations(tar, image_name)

    annotated_image = get_image(tar, image_name, "class")
    image = get_image(tar, image_name)

    image.show()
    annotated_image.show()

    palette = annotated_image.getpalette()

    # --------------------------------------------------------------------------------- #

    MRF = BasicModel.MRFModel(palette)
    MRF.fit([image], [annotated_image])
    prediction = MRF.predict(image)

    predicted_image = Image.fromarray(prediction.astype(np.uint8), mode="P")
    predicted_image.putpalette(palette)
    predicted_image.show()


if __name__ == '__main__':
    main()
    print("DONE!")
