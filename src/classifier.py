# -*- coding: utf-8 -*-

import copy
import cv2
import numpy as np

import torch
from torch.autograd import Variable
from torchvision import models

from preprocess import preprocess_image


class Classifier(object):

    def __init__(self, filter_):
        self.filter_ = filter_
        print('reading vgg16 ...')
        self.model = models.vgg16(pretrained=True)
        # print('reading vgg19 ...')
        # self.model = models.vgg19(pretrained=True)
        self.model.eval()

    def forward(self, image):
        image = preprocess_image(image, self.filter_)
        image.grad = None
        out = self.model(image)

        return out, image

