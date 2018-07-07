import os
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable

from classifier import Classifier
from utils import *
from preprocess import *


class FastGradientSignTargeted(Classifier):

    def __init__(self, alpha, output_dir, n_iter, aug):
        super().__init__()
        self.alpha = alpha
        self.output_dir = output_dir
        self.n_iter = n_iter
        self.aug = aug
        self.labels = read_labels('../data/labels.json')

    def _classify(self, image):
        image = restore_image(image)
        out, _ = self.forward(image, self.aug)

        _, adv_class = out.data.max(1)
        adv_class_confidence = nn.functional.softmax(out)[0][adv_class].data.numpy()[0]
        adv_class = adv_class.numpy()[0]
        # print(f'org : {self.org_class}, target : {self.target_class}, out : {adv_class}')
        # print(adv_class_confidence)

        if adv_class == self.target_class:
            # print(f'Org {self.org_class}({self.labels[str(self.org_class)]})\n' \
            #       f'Adv {adv_class}({self.labels[str(adv_class)]}) ({adv_class_confidence})')
            noise_image = self.image_org - image
            save_image(self.output_dir, self.org_class, self.target_class, noise_image, 'adv_noise')
            save_image(self.output_dir, self.org_class, self.target_class, image, 'adv')
            return 1
        else:
            return 0

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

            flg = self._classify(image)
            if flg:
                break
            image = restore_image(image)

        return flg
