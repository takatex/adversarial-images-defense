# -*- coding: utf-8 -*-

import sys, os
import cv2
import json

def read_labels(path):
    with open(path, "r") as f:
        return json.loads(f.read())

def save_image(save_dir, org_class, adv_class, image, memo=''):
    path = os.path.join(save_dir, f'org_{org_class}_adv_{adv_class}_{memo}.png')
    cv2.imwrite(path, image)
    print(f'Saved in {path}.')


def show_progress(iter_, n_iter, count, acc):
    sys.stdout.write(f'\r[{iter_ : 5d} / {n_iter : 5d}] count: {count : 5d} acc : {acc : 3f}')
    sys.stdout.flush()
