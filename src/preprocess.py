# -*- coding: utf-8 -*-

import numpy as np
import cv2
import torch
from torch import nn
from torch.autograd import Variable

import copy
import cv2
import numpy as np

import torch
from torch.autograd import Variable
from torchvision import models


def normalize(image):
    image = np.float32(cv2.resize(image, (224, 224))) / 255
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = np.ascontiguousarray(image[..., ::-1])
    image = image.transpose(2, 0, 1)  # Convert array to D,W,H

    for c, _ in enumerate(image):
        image[c] -= mean[c]
        image[c] /= std[c]

    return image

# def


def preprocess_image(image, filter_):
    if filter_:
        pass
    # normalize
    image = normalize(image)

    image = torch.from_numpy(image).float()
    image.unsqueeze_(0)
    im_as_var = Variable(image, requires_grad=True)
    return im_as_var


def restore_image(image_a):
    """
    Args:
        image (torch variable): Image to recreate

    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = copy.copy(image_a.data.numpy()[0])
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
