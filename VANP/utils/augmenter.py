#######################################################################################################
# This file is borrowed from COiLTRAiNE https://github.com/felipecode/coiltraine by Felipe Codevilla  #
# COiLTRAiNE itself is under MIT License                                                              #
#######################################################################################################

from PIL import Image
from typing import Optional

import numpy as np
import torch
from imgaug import augmenters as iaa


DEBUG = False


def get_augmenter(
    iteration: Optional[int] = 1, bsz: Optional[int] = 32, aug_type: str = "medium"
):
    if aug_type == "medium":
        return medium(iteration, bsz)
    elif aug_type == "soft":
        return soft(iteration, bsz)
    elif aug_type == "high":
        return high(iteration, bsz)
    elif aug_type == "medium_harder":
        return medium_harder(iteration, bsz)
    elif aug_type == "super_hard":
        return super_hard(iteration, bsz)
    elif aug_type == "custom":
        return custom(iteration, bsz)
    elif aug_type == "soft_harder":
        return soft_harder(iteration, bsz)
    elif aug_type == "segmentation":
        return seg_aug()
    else:
        raise ValueError(
            "Unknown augmentation, value should be one of"
            "'medium', 'high', 'medium_harder', 'super_hard', 'soft_harder', 'custom'"
        )


class Crop:
    def __init__(self, crop_size):
        self.top = crop_size[0]
        self.bottom = crop_size[1]

    def __call__(self, img):
        return Image.fromarray(img[self.top : -self.bottom])


class MaskPILToTensor:
    def __call__(self, img):
        return torch.from_numpy(np.array(img)).long()


def seg_aug():
    augmenter = iaa.Sequential(
        [
            iaa.Sometimes(0.3, iaa.GaussianBlur()),
            # blur images with a sigma between 0 and 1.5
            iaa.Sometimes(0.3, iaa.AdditiveGaussianNoise(per_channel=True)),
            # add gaussian noise to images
            iaa.Sometimes(
                0.1, iaa.CoarseDropout(size_percent=(0.08, 0.2), per_channel=True)
            ),
            # randomly remove up to X% of the pixels
            iaa.Sometimes(0.1, iaa.Dropout(per_channel=True)),
            # randomly remove up to X% of the pixels
            iaa.Sometimes(0.2, iaa.LinearContrast(per_channel=True)),
            # improve or worsen the contrast
        ],
        random_order=True,  # do all of the above in random order
    )

    return augmenter


def medium(image_iteration: int = 1, bsz: int = 32):
    iteration = image_iteration / (bsz * 1.5)
    frequency_factor = 0.05 + float(iteration) / 1000000.0
    color_factor = float(iteration) / 1000000.0
    dropout_factor = 0.198667 + (0.03856658 - 0.198667) / (
        1 + (iteration / 196416.6) ** 1.863486
    )

    blur_factor = 0.5 + (0.5 * iteration / 100000.0)

    add_factor = 10 + 10 * iteration / 150000.0

    multiply_factor_pos = 1 + (2.5 * iteration / 500000.0)
    multiply_factor_neg = 1 - (0.91 * iteration / 500000.0)

    contrast_factor_pos = 1 + (0.5 * iteration / 500000.0)
    contrast_factor_neg = 1 - (0.5 * iteration / 500000.0)

    if DEBUG:
        print(
            f"Augment Status: {frequency_factor = }, {color_factor = }, {dropout_factor = }, "
            f"{blur_factor = }, {add_factor = }, "
            f"{multiply_factor_pos = }, {multiply_factor_neg = }, "
            f"{contrast_factor_pos = }, {contrast_factor_neg = }"
        )

    augmenter = iaa.Sequential(
        [
            iaa.Sometimes(frequency_factor, iaa.GaussianBlur((0, blur_factor))),
            # blur images with a sigma between 0 and 1.5
            iaa.Sometimes(
                frequency_factor,
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, dropout_factor), per_channel=color_factor
                ),
            ),
            # add gaussian noise to images
            iaa.Sometimes(
                frequency_factor,
                iaa.CoarseDropout(
                    (0.0, dropout_factor),
                    size_percent=(0.08, 0.2),
                    per_channel=color_factor,
                ),
            ),
            # randomly remove up to X% of the pixels
            iaa.Sometimes(
                frequency_factor,
                iaa.Dropout((0.0, dropout_factor), per_channel=color_factor),
            ),
            # randomly remove up to X% of the pixels
            iaa.Sometimes(
                frequency_factor,
                iaa.Add((-add_factor, add_factor), per_channel=color_factor),
            ),
            # change brightness of images (by -X to Y of original value)
            iaa.Sometimes(
                frequency_factor,
                iaa.Multiply(
                    (multiply_factor_neg, multiply_factor_pos), per_channel=color_factor
                ),
            ),
            # change brightness of images (X-Y% of original value)
            iaa.Sometimes(
                frequency_factor,
                iaa.LinearContrast(
                    (contrast_factor_neg, contrast_factor_pos), per_channel=color_factor
                ),
            ),
            # improve or worsen the contrast
            iaa.Sometimes(frequency_factor, iaa.Grayscale((0.0, 1))),  # put grayscale
        ],
        random_order=True,  # do all of the above in random order
    )

    return augmenter


def soft(image_iteration: int = 1, bsz: int = 32):
    iteration = image_iteration / (bsz * 1.5)
    frequency_factor = 0.05 + float(iteration) / 1200000.0
    color_factor = float(iteration) / 1200000.0
    dropout_factor = 0.198667 + (0.03856658 - 0.198667) / (
        1 + (iteration / 196416.6) ** 1.863486
    )

    blur_factor = 0.5 + (0.5 * iteration / 120000.0)

    add_factor = 10 + 10 * iteration / 170000.0

    multiply_factor_pos = 1 + (2.5 * iteration / 800000.0)
    multiply_factor_neg = 1 - (0.91 * iteration / 800000.0)

    contrast_factor_pos = 1 + (0.5 * iteration / 800000.0)
    contrast_factor_neg = 1 - (0.5 * iteration / 800000.0)

    if DEBUG:
        print(
            f"Augment Status: {frequency_factor = }, {color_factor = }, {dropout_factor = }, "
            f"{blur_factor = }, {add_factor = }, "
            f"{multiply_factor_pos = }, {multiply_factor_neg = }, "
            f"{contrast_factor_pos = }, {contrast_factor_neg = }"
        )

    augmenter = iaa.Sequential(
        [
            iaa.Sometimes(frequency_factor, iaa.GaussianBlur((0, blur_factor))),
            # blur images with a sigma between 0 and 1.5
            iaa.Sometimes(
                frequency_factor,
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, dropout_factor), per_channel=color_factor
                ),
            ),
            # add gaussian noise to images
            iaa.Sometimes(
                frequency_factor,
                iaa.CoarseDropout(
                    (0.0, dropout_factor),
                    size_percent=(0.08, 0.2),
                    per_channel=color_factor,
                ),
            ),
            # randomly remove up to X% of the pixels
            iaa.Sometimes(
                frequency_factor,
                iaa.Dropout((0.0, dropout_factor), per_channel=color_factor),
            ),
            # randomly remove up to X% of the pixels
            iaa.Sometimes(
                frequency_factor,
                iaa.Add((-add_factor, add_factor), per_channel=color_factor),
            ),
            # change brightness of images (by -X to Y of original value)
            iaa.Sometimes(
                frequency_factor,
                iaa.Multiply(
                    (multiply_factor_neg, multiply_factor_pos), per_channel=color_factor
                ),
            ),
            # change brightness of images (X-Y% of original value)
            iaa.Sometimes(
                frequency_factor,
                iaa.LinearContrast(
                    (contrast_factor_neg, contrast_factor_pos), per_channel=color_factor
                ),
            ),
            # improve or worsen the contrast
            iaa.Sometimes(frequency_factor, iaa.Grayscale((0.0, 1))),  # put grayscale
        ],
        random_order=True,  # do all of the above in random order
    )

    return augmenter


def high(image_iteration: int = 1, bsz: int = 32):
    iteration = image_iteration / (bsz * 1.5)
    frequency_factor = 0.05 + float(iteration) / 800000.0
    color_factor = float(iteration) / 800000.0
    dropout_factor = 0.198667 + (0.03856658 - 0.198667) / (
        1 + (iteration / 196416.6) ** 1.863486
    )

    blur_factor = 0.5 + (0.5 * iteration / 80000.0)

    add_factor = 10 + 10 * iteration / 120000.0

    multiply_factor_pos = 1 + (2.5 * iteration / 350000.0)
    multiply_factor_neg = 1 - (0.91 * iteration / 400000.0)

    contrast_factor_pos = 1 + (0.5 * iteration / 350000.0)
    contrast_factor_neg = 1 - (0.5 * iteration / 400000.0)

    if DEBUG:
        print(
            f"Augment Status: {frequency_factor = }, {color_factor = }, {dropout_factor = }, "
            f"{blur_factor = }, {add_factor = }, "
            f"{multiply_factor_pos = }, {multiply_factor_neg = }, "
            f"{contrast_factor_pos = }, {contrast_factor_neg = }"
        )

    augmenter = iaa.Sequential(
        [
            iaa.Sometimes(frequency_factor, iaa.GaussianBlur((0, blur_factor))),
            # blur images with a sigma between 0 and 1.5
            iaa.Sometimes(
                frequency_factor,
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, dropout_factor), per_channel=color_factor
                ),
            ),
            # add gaussian noise to images
            iaa.Sometimes(
                frequency_factor,
                iaa.CoarseDropout(
                    (0.0, dropout_factor),
                    size_percent=(0.08, 0.2),
                    per_channel=color_factor,
                ),
            ),
            # randomly remove up to X% of the pixels
            iaa.Sometimes(
                frequency_factor,
                iaa.Dropout((0.0, dropout_factor), per_channel=color_factor),
            ),
            # randomly remove up to X% of the pixels
            iaa.Sometimes(
                frequency_factor,
                iaa.Add((-add_factor, add_factor), per_channel=color_factor),
            ),
            # change brightness of images (by -X to Y of original value)
            iaa.Sometimes(
                frequency_factor,
                iaa.Multiply(
                    (multiply_factor_neg, multiply_factor_pos), per_channel=color_factor
                ),
            ),
            # change brightness of images (X-Y% of original value)
            iaa.Sometimes(
                frequency_factor,
                iaa.LinearContrast(
                    (contrast_factor_neg, contrast_factor_pos), per_channel=color_factor
                ),
            ),
            # improve or worsen the contrast
            iaa.Sometimes(frequency_factor, iaa.Grayscale((0.0, 1))),  # put grayscale
        ],
        random_order=True,  # do all of the above in random order
    )

    return augmenter


def medium_harder(image_iteration: int = 1, bsz: int = 32):
    iteration = image_iteration / bsz
    frequency_factor = 0.05 + float(iteration) / 1000000.0
    color_factor = float(iteration) / 1000000.0
    dropout_factor = 0.198667 + (0.03856658 - 0.198667) / (
        1 + (iteration / 196416.6) ** 1.863486
    )

    blur_factor = 0.5 + (0.5 * iteration / 100000.0)

    add_factor = 10 + 10 * iteration / 150000.0

    multiply_factor_pos = 1 + (2.5 * iteration / 500000.0)
    multiply_factor_neg = 1 - (0.91 * iteration / 500000.0)

    contrast_factor_pos = 1 + (0.5 * iteration / 500000.0)
    contrast_factor_neg = 1 - (0.5 * iteration / 500000.0)

    if DEBUG:
        print(
            f"Augment Status: {frequency_factor = }, {color_factor = }, {dropout_factor = }, "
            f"{blur_factor = }, {add_factor = }, "
            f"{multiply_factor_pos = }, {multiply_factor_neg = }, "
            f"{contrast_factor_pos = }, {contrast_factor_neg = }"
        )

    augmenter = iaa.Sequential(
        [
            iaa.Sometimes(frequency_factor, iaa.GaussianBlur((0, blur_factor))),
            # blur images with a sigma between 0 and 1.5
            iaa.Sometimes(
                frequency_factor,
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, dropout_factor), per_channel=color_factor
                ),
            ),
            # add gaussian noise to images
            iaa.Sometimes(
                frequency_factor,
                iaa.CoarseDropout(
                    (0.0, dropout_factor),
                    size_percent=(0.08, 0.2),
                    per_channel=color_factor,
                ),
            ),
            # randomly remove up to X% of the pixels
            iaa.Sometimes(
                frequency_factor,
                iaa.Dropout((0.0, dropout_factor), per_channel=color_factor),
            ),
            # randomly remove up to X% of the pixels
            iaa.Sometimes(
                frequency_factor,
                iaa.Add((-add_factor, add_factor), per_channel=color_factor),
            ),
            # change brightness of images (by -X to Y of original value)
            iaa.Sometimes(
                frequency_factor,
                iaa.Multiply(
                    (multiply_factor_neg, multiply_factor_pos), per_channel=color_factor
                ),
            ),
            # change brightness of images (X-Y% of original value)
            iaa.Sometimes(
                frequency_factor,
                iaa.LinearContrast(
                    (contrast_factor_neg, contrast_factor_pos), per_channel=color_factor
                ),
            ),
            # improve or worsen the contrast
            iaa.Sometimes(frequency_factor, iaa.Grayscale((0.0, 1))),  # put grayscale
        ],
        random_order=True,  # do all of the above in random order
    )

    return augmenter


def super_hard(image_iteration: int = 1, bsz: int = 32):
    """
    modified
    """

    iteration = image_iteration / bsz
    frequency_factor = min(0.05 + float(iteration) / 50000.0, 1.0)
    color_factor = float(iteration) / 100000.0
    dropout_factor = 0.198667 + (0.03856658 - 0.198667) / (
        1 + (iteration / 196416.6) ** 1.863486
    )

    blur_factor = 0.5 + (0.5 * iteration / 100000.0)

    add_factor = 10 + 10 * iteration / 100000.0

    multiply_factor_pos = 1 + (2.5 * iteration / 200000.0)
    multiply_factor_neg = 1 - (0.91 * iteration / 500000.0)

    contrast_factor_pos = 1 + (0.5 * iteration / 500000.0)
    contrast_factor_neg = 1 - (0.5 * iteration / 500000.0)

    if DEBUG:
        print(
            f"Augment Status: {frequency_factor = }, {color_factor = }, {dropout_factor = }, "
            f"{blur_factor = }, {add_factor = }, "
            f"{multiply_factor_pos = }, {multiply_factor_neg = }, "
            f"{contrast_factor_pos = }, {contrast_factor_neg = }"
        )

    augmenter = iaa.Sequential(
        [
            iaa.Sometimes(frequency_factor, iaa.GaussianBlur((0, blur_factor))),
            # blur images with a sigma between 0 and 1.5
            iaa.Sometimes(
                frequency_factor,
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, dropout_factor), per_channel=color_factor
                ),
            ),
            # add gaussian noise to images
            iaa.Sometimes(
                frequency_factor,
                iaa.CoarseDropout(
                    (0.0, dropout_factor),
                    size_percent=(0.08, 0.2),
                    per_channel=color_factor,
                ),
            ),
            # randomly remove up to X% of the pixels
            iaa.Sometimes(
                frequency_factor,
                iaa.Dropout((0.0, dropout_factor), per_channel=color_factor),
            ),
            # randomly remove up to X% of the pixels
            iaa.Sometimes(
                frequency_factor,
                iaa.Add((-add_factor, add_factor), per_channel=color_factor),
            ),
            # change brightness of images (by -X to Y of original value)
            iaa.Sometimes(
                frequency_factor,
                iaa.Multiply(
                    (multiply_factor_neg, multiply_factor_pos), per_channel=color_factor
                ),
            ),
            # change brightness of images (X-Y% of original value)
            iaa.Sometimes(
                frequency_factor,
                iaa.LinearContrast(
                    (contrast_factor_neg, contrast_factor_pos), per_channel=color_factor
                ),
            ),
            # improve or worsen the contrast
            # iaa.Sometimes(frequency_factor, iaa.Grayscale((0.0, 1))),  # put grayscale
        ],
        random_order=True,  # do all of the above in random order
    )

    return augmenter


def custom(image_iteration: int = 1, bsz: int = 32):
    """
    modified
    """

    iteration = image_iteration / bsz
    frequency_factor = min(0.05 + float(iteration) / 50000.0, 1.0)
    color_factor = float(iteration) / 100000.0
    dropout_factor = 0.198667 + (0.03856658 - 0.198667) / (
        1 + (iteration / 196416.6) ** 1.863486
    )

    blur_factor = 0.5 + (0.5 * iteration / 20000.0)

    # add_factor = 10 + 10 * iteration / 100000.0

    # print (add_factor)

    # multiply_factor_pos = 1 + (2.5 * iteration / 300000.0)
    # multiply_factor_neg = 1 - (0.91 * iteration / 300000.0)
    #
    # contrast_factor_pos = 1 + (0.2 * iteration / 500000.0)
    # contrast_factor_neg = 1 - (0.5 * iteration / 500000.0)

    if DEBUG:
        print(
            f"Augment Status: {frequency_factor = }, {color_factor = }, {dropout_factor = }, "
            f"{blur_factor = }"
        )

    augmenter = iaa.Sequential(
        [
            iaa.Sometimes(frequency_factor, iaa.GaussianBlur((0, blur_factor))),
            # blur images with a sigma between 0 and 1.5
            iaa.Sometimes(
                frequency_factor,
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, dropout_factor), per_channel=color_factor
                ),
            ),
            # # add gaussian noise to images
            # iaa.Sometimes(frequency_factor, iaa.CoarseDropout((0.0, dropout_factor), size_percent=(
            #     0.08, 0.2), per_channel=color_factor)),
            # # randomly remove up to X% of the pixels
            iaa.Sometimes(
                frequency_factor,
                iaa.Dropout((0.0, dropout_factor), per_channel=color_factor),
            ),
            # randomly remove up to X% of the pixels
            iaa.Sometimes(frequency_factor, iaa.Add((-30, 30), per_channel=False)),
            # # change brightness of images (by -X to Y of original value)
            iaa.Sometimes(frequency_factor, iaa.Multiply((0.9, 1.3), per_channel=True)),
            # # change brightness of images (X-Y% of original value)
            # iaa.Sometimes(frequency_factor, iaa.LinearContrast((0.1,0.5),
            #                                                               per_channel=True)),
            # improve or worsen the contrast
            # iaa.Sometimes(frequency_factor, iaa.Grayscale((0.0, 1))),  # put grayscale
        ],
        random_order=True,  # do all of the above in random order
    )

    return augmenter


def soft_harder(image_iteration: int = 1, bsz: int = 32):
    iteration = image_iteration / bsz
    frequency_factor = 0.05 + float(iteration) / 1200000.0
    color_factor = float(iteration) / 1200000.0
    dropout_factor = 0.198667 + (0.03856658 - 0.198667) / (
        1 + (iteration / 196416.6) ** 1.863486
    )

    blur_factor = 0.5 + (0.5 * iteration / 120000.0)

    add_factor = 10 + 10 * iteration / 170000.0

    multiply_factor_pos = 1 + (2.5 * iteration / 800000.0)
    multiply_factor_neg = 1 - (0.91 * iteration / 800000.0)

    contrast_factor_pos = 1 + (0.5 * iteration / 800000.0)
    contrast_factor_neg = 1 - (0.5 * iteration / 800000.0)

    if DEBUG:
        print(
            f"Augment Status: {frequency_factor = }, {color_factor = }, {dropout_factor = }, "
            f"{blur_factor = }, {add_factor = }, "
            f"{multiply_factor_pos = }, {multiply_factor_neg = }, "
            f"{contrast_factor_pos = }, {contrast_factor_neg = }"
        )

    augmenter = iaa.Sequential(
        [
            iaa.Sometimes(frequency_factor, iaa.GaussianBlur((0, blur_factor))),
            # blur images with a sigma between 0 and 1.5
            iaa.Sometimes(
                frequency_factor,
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, dropout_factor), per_channel=color_factor
                ),
            ),
            # add gaussian noise to images
            iaa.Sometimes(
                frequency_factor,
                iaa.CoarseDropout(
                    (0.0, dropout_factor),
                    size_percent=(0.08, 0.2),
                    per_channel=color_factor,
                ),
            ),
            # randomly remove up to X% of the pixels
            iaa.Sometimes(
                frequency_factor,
                iaa.Dropout((0.0, dropout_factor), per_channel=color_factor),
            ),
            # randomly remove up to X% of the pixels
            iaa.Sometimes(
                frequency_factor,
                iaa.Add((-add_factor, add_factor), per_channel=color_factor),
            ),
            # change brightness of images (by -X to Y of original value)
            iaa.Sometimes(
                frequency_factor,
                iaa.Multiply(
                    (multiply_factor_neg, multiply_factor_pos), per_channel=color_factor
                ),
            ),
            # change brightness of images (X-Y% of original value)
            iaa.Sometimes(
                frequency_factor,
                iaa.LinearContrast(
                    (contrast_factor_neg, contrast_factor_pos), per_channel=color_factor
                ),
            ),
            # improve or worsen the contrast
            iaa.Sometimes(frequency_factor, iaa.Grayscale((0.0, 1))),  # put grayscale
        ],
        random_order=True,  # do all of the above in random order
    )

    return augmenter
