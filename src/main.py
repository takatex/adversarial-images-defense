# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import cv2

from fgsm import FastGradientSignTargeted

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dir', type=str, default='../data/input_images')
parser.add_argument('-o', '--output_dir', type=str, default='../data/output_images')

opt = parser.parse_args()
os.makedirs(opt.output_dir, exist_ok=True)


def main():
    # Pick one of the examples
    # example_list = [['../input_images/apple.JPEG', 948],
    #                 ['../input_images/eel.JPEG', 390],
    #                 ['../input_images/bird.JPEG', 13]]
    # selected_example = example_index
    # Read image
    image_path = '../data/input_images/bird.JPEG'
    image = cv2.imread(image_path, 1)
    org_class = 13
    target_class = 839
    FGSM = FastGradientSignTargeted(alpha=0.01, output_dir=opt.output_dir, filter_=False)
    FGSM.generate(image, org_class, target_class)

    # target_example = 0  # Apple
    # (original_image, prep_img, org_class, _, pretrained_model) =\
    #     get_params(target_example)
    # target_class = 62  # Mud turtle
    #


if __name__ == '__main__':
    main()
