#!/usr/bin/env python


import json
import cPickle
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import argparse


def showImage(im, kps):


    classToColor = ['', 'red', 'yellow', 'blue', 'magenta']
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    fig = ax.imshow(im, aspect='equal')
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    thresh = 0.7

    line = [[13, 14], [14, 4], [4, 5], [5, 6], [14, 1], [1, 2], [2, 3], \
            [14, 10], [10, 11], [11, 12], [14, 7], [7, 8], [8, 9]]


    b = 'b'
    r = 'r'
    g = 'g'
    c = [b, b, b, r, r, r, b, b, b, r, r, r, g, g]
    c2 = [g, r, r, r, b, b, b, r, r, r, b, b, b]

    for i, kp in enumerate(kps):

        for j in range(14):
            if kp[j * 3 + 2] == 3:
                continue
            x, y, z = kp[j * 3 : (j + 1) * 3]
            ax.add_patch(
                    plt.Circle((x, y), 3,
                          fill=True,
                          color = c[j], 
                          linewidth=2.0)
                )

        for j, l in enumerate(line):
            i0 = l[0] - 1
            p0 = kp[i0 * 3 : (i0 + 1) * 3] 

            i1 = l[1] - 1
            p1 = kp[i1 * 3 : (i1 + 1) * 3]

            if p0[2] == 3 or p1[2] == 3:
                continue
            
            ax.add_patch(
                    plt.Arrow(p0[0], p0[1], 
                    float(p1[0]) - p0[0], float(p1[1]) - p0[1], 
                    color = c2[j])
                    )




def showBox(pred_name, type_name, dataset):
    with open(pred_name, 'r') as f:
        data = json.load(f)


    for i, line in enumerate(data):
        if i % 1000 == 0:
            print(i)
            imname = line['image_id']
            im = cv2.imread(os.path.join('data/aichal', dataset, imname + '.jpg'))
            kps = line['keypoint_annotations'].values()

            showImage(im, kps)
            plt.savefig('{}{}{}.png'.format(dataset.upper(), i, type_name.upper()), 
                    bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type', type=str, help='option: cmu, aichal, fuse, combine')
    parser.add_argument('--dataset', type=str, 
            help='test or testb, default test', default='test')
    args = parser.parse_args()


    if args.dataset not in ['test', 'testb']:
        print('dataset not in option')
        exit(0)
    

    if args.type == 'cmu':
        pred = 'pred_cmu_test_0.6.json'
    elif args.type == 'aichal':
        pred = 'pred_cmu_aichal_v2_test_0.6.json'
    elif args.type == 'fuse':
        if args.dataset == 'test':
            pred = 'pred_fuse_averageBody_aichal_test_0.6_0.6.json'
        elif args.dataset == 'testb':
            pred = 'pred_fuse_averageBody_aichal_testb_0.6_0.6.json'
    else:
        print('type not in option')
        exit(0)

    showBox(pred, args.type, args.dataset)


