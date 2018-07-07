import os
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable

from classifier import Classifier
from utils import *
from preprocess import *


class FastGradientSignTargeted(Classifier):

    def __init__(self, alpha, n_iter, aug, save, save_path):
        super().__init__()
        self.alpha = alpha
        self.n_iter = n_iter
        self.aug = aug
        self.labels = read_labels('../data/labels.json')
        self.save = save
        self.save_path = save_path

    def _classify(self, image):
        image = restore_image(image)
        out, _ = self.forward(image, self.aug)

        _, adv_class = out.data.max(1)
        adv_class_confidence = nn.functional.softmax(out, dim=1)[0][adv_class].data.numpy()[0]
        adv_class = adv_class.numpy()[0]

        if adv_class == self.target_class:
            if self.save:
                noise_image = self.image_org - image
                save_image(self.save_path, self.org_class, self.target_class, noise_image, 'adv_noise')
                save_image(self.save_path, self.org_class, self.target_class, image, 'adv')
            return 1, adv_class_confidence
        else:
            return 0, adv_class_confidence

    def generate(self, image, org_class, target_class):
        self.image_org = image
        self.org_class = org_class
        self.target_class = target_class

        criterion = nn.CrossEntropyLoss()
        target_class_var = Variable(torch.from_numpy(np.asarray([target_class])))

        for _ in range(self.n_iter):
            out, image = self.forward(image, self.aug)
            loss = criterion(out, target_class_var)
            loss.backward()

            noise = self.alpha * torch.sign(image.grad.data)
            image.data = image.data - noise

            flg, adv_class_confidence = self._classify(image)
            if flg:
                image = restore_image(image)
                break

            image = restore_image(image)

        return flg, image, adv_class_confidence
