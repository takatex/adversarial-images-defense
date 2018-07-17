# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import cv2
import glob

import torch
from fgsm import FastGradientSignTargeted
from classifier import Classifier
from utils import show_progress


parser = argparse.ArgumentParser()
parser.add_argument('--n_iter_adv', type=int, default=10,
                    help='number of iterations to generate adv images (default: 10)')
parser.add_argument('--n_iter_aug', type=int, default=50,
                    help='number of iterations to generate aug images (default: 50)')
parser.add_argument('--test_size', type=int, default=1000,
                    help='number of test images (default: 1000)')
parser.add_argument('-i', '--input_dir', type=str, default='../data/inputs',
                    help='input images directory')
parser.add_argument('-o', '--output_dir', type=str, default='../data/outputs',
                    help='output images directory')
parser.add_argument('-s', '--save_adv_image', action='store_true', default=True,
                    help='save adv images')

args = parser.parse_args()

def main():
    ## make output dirs
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_adv_image:
        os.makedirs(os.path.join(args.output_dir, 'adv_images'), exist_ok=True)

    ## input image paths and labels
    dataset = np.loadtxt('../data/val.txt', dtype=str)
    ind = np.random.randint(0, len(dataset), args.test_size)
    dataset = dataset[ind]

    ## Adv model
    FGSM = FastGradientSignTargeted(alpha=0.01, n_iter=args.n_iter_adv, aug=False,
                                    save=args.save_adv_image,
                                    save_path=os.path.join(args.output_dir, 'adv_images'))

    ## classifier
    C = Classifier()
    outs = []
    correct = 0
    count = 0
    for i, (image_path, org_class) in enumerate(dataset):
        image = cv2.imread(os.path.join(args.input_dir, image_path), 1)
        org_class = int(org_class)
        target_class = np.random.randint(0, 1000)

        out_normal = C.ensemble_classify(image, args.n_iter_aug)

        flg, image, adv_class_confidence = FGSM.generate(image, org_class, target_class)

        if flg:
            out_adv = C.ensemble_classify(image, args.n_iter_aug)
            correct += int(out_adv == org_class)
            outs.append([out_normal, out_adv, org_class, target_class, adv_class_confidence])
            count += 1
        else:
            pass
        show_progress(i+1, args.test_size, count, (correct / count))

    np.savetxt(os.path.join(args.output_dir, 'log.txt'), np.array(outs))

if __name__ == '__main__':
    main()
