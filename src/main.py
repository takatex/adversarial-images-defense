# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import cv2

import torch
from fgsm import FastGradientSignTargeted

parser = argparse.ArgumentParser()
parser.add_argument('--n_iter', type=int, default=10,
                    help='number of iterations to generate adv images (default: 10)')
parser.add_argument('-i', '--input_dir', type=str, default='../data/input_images',
                    help='input images directory')
parser.add_argument('-o', '--output_dir', type=str, default='../data/output_images',
                    help='output images directory')

args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)


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

    FGSM = FastGradientSignTargeted(alpha=0.01, output_dir=args.output_dir, n_iter=args.n_iter, aug=False)
    flg = FGSM.generate(image, org_class, target_class)


if __name__ == '__main__':
    main()
