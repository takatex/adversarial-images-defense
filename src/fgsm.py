import os
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
# from torch.autograd.gradcheck import zero_gradients  # See processed_image.grad = None

from utils import *
from classifier import Classifier
from preprocess import *
from utils import *

class FastGradientSignTargeted(Classifier):
    def __init__(self, alpha, output_dir, filter_):
        super().__init__(filter_)
        self.alpha = alpha
        self.output_dir = output_dir
        self.labels = read_labels('../data/labels.json')

    def test(self, image):
        image = restore_image(image)
        # image = preprocess_image(image)
        out, _ = self.forward(image)
        # out = self.forward(image)

        _, adv_class = out.data.max(1)
        adv_class_confidence = nn.functional.softmax(out)[0][adv_class].data.numpy()[0]
        adv_class = adv_class.numpy()[0]
        print(f'org : {self.org_class}, target : {self.target_class}, now : {adv_class}')
        print(adv_class_confidence)
        # Check if the prediction is different than the original
        if adv_class == self.target_class:
            print(f'Org {self.org_class}({self.labels[str(self.org_class)]})\n' \
                  f'Adv {adv_class}({self.labels[str(adv_class)]}) ({adv_class_confidence})')
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

        # image = preprocess_image(image)
        for i in range(10):
            # print(f'Iteration : {i}')
            # image.grad = None
            out, image = self.forward(image)
            # out = self.forward(image)
            loss = criterion(out, target_class_var)
            loss.backward()

            noise = self.alpha * torch.sign(image.grad.data)
            image.data = image.data - noise

            flg = self.test(image)
            if flg:
                break
            image = restore_image(image)

        return 1
