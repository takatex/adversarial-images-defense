# -*- coding: utf-8 -*-

import numpy as np
import cv2
from imgaug import augmenters as iaa

import torch
from torch import nn
from torch.autograd import Variable


class Augmenter(object):

    def __init__(self):
        self.seq = iaa.Sequential([
            iaa.SomeOf((2, 3), [
                iaa.Superpixels(p_replace=0.5, n_segments=100),
                iaa.GaussianBlur(0, 2.0),
                iaa.BilateralBlur(5, sigma_color=250, sigma_space=250),
                iaa.Sharpen(alpha=1),
                iaa.Emboss(alpha=1),
                iaa.AdditiveGaussianNoise(scale = 0.1 * 255),
                iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25),
                iaa.ContrastNormalization((0.5, 2.0))
            ], random_order=True)
        ])

    def get_aug_image(self, image):
        return self.seq.augment_image(image)


class PreprocessImage(Augmenter):

    def __init__(self):
        super().__init__()

    def normalize(self, image):
        image = np.float32(cv2.resize(image, (224, 224))) / 255
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image = np.ascontiguousarray(image[..., ::-1])
        image = image.transpose(2, 0, 1)  # Convert array to D,W,H

        for c, _ in enumerate(image):
            image[c] -= mean[c]
            image[c] /= std[c]

        return image

    def preprocess(self, image, aug):
        if aug:
            image = self.get_aug_image(image)

        image = self.normalize(image)
        image = torch.from_numpy(image).float()
        image.unsqueeze_(0)
        image = Variable(image, requires_grad=True)
        return image


def restore_image(image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    image = image.data.numpy()[0].copy()
    for c in range(3):
        image[c] *= std[c]
        image[c] += mean[c]
    image[image > 1] = 1
    image[image < 0] = 0
    image = np.round(image * 255)
    image = np.uint8(image).transpose(1, 2, 0)
    # Convert RBG to GBR
    image = image[..., ::-1]
    return image
