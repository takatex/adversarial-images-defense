# -*- coding: utf-8 -*-

import copy
import cv2
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torchvision import models

from preprocess import PreprocessImage


class Classifier(PreprocessImage):

    def __init__(self):
        super().__init__()
        print('Loading vgg16 ...')
        self.model = models.vgg16(pretrained=True)
        self.model.eval()

    def forward(self, image, aug):
        image = self.preprocess(image, aug)
        out = self.model(image)

        return out, image

    def ensemble_classify(image):
        conf = np.zeros([1000])
        for _ in range(100):
            out, _ = self.forward(image, aug=True)
            _, label = out.data.max(1)
            label = label.numpy()[0]
            conf += nn.functional.softmax(out)[0].data.numpy()

        pred = np.where(conf == max(conf))[0][0]
        return pred
