# -*- coding: utf-8 -*-

import os
import cv2
import json

def read_labels(path):
    with open(path, "r") as f:
        return json.loads(f.read())

def save_image(save_dir, org_class, adv_class, image, memo=''):
    path = os.path.join(save_dir, f'org_{org_class}_adv_{adv_class}_{memo}.png')
    cv2.imwrite(path, image)
    print(f'Saved in {path}.')

