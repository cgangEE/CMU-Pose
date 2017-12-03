#!/usr/bin/env python


import json
import cPickle
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import argparse

def compute_oks(kps, gt_boxes, gt_kps):

    delta = 2*np.array([0.01388152, 0.01515228, 0.01057665, 0.01417709, \
        0.01497891, 0.01402144, 0.03909642, 0.03686941, 0.01981803, \
        0.03843971, 0.03412318, 0.02415081, 0.01291456, 0.01236173])

    kps_count = len(kps)
    gt_count = len(gt_kps)
    oks = np.zeros((gt_count, kps_count))

    oks_full = np.zeros((gt_count, kps_count, 14))

    for i in range(gt_count):
        gt_kp = np.reshape(gt_kps[i], (14, 3))
        visible = gt_kp[:, 2] == 1
        gt_box = gt_boxes[i]
        scale = np.float32((gt_box[3]-gt_box[1])*(gt_box[2]-gt_box[0]))

        if np.sum(visible) == 0:
            oks[i, :] = 0
            continue
        for j in range(kps_count):
            kp = np.reshape(kps[j], (14, 3))

            dis = np.sum((gt_kp[visible, :2] - kp[visible, :2]) ** 2, axis=1)
            oks[i, j] = np.mean(np.exp(-dis/2/delta[visible]**2/(scale+1)))

            dis_full = np.zeros(14)
            idx = np.nonzero(visible)
            dis_full[idx] = np.sum(
                    (gt_kp[visible, :2] - kp[visible, :2]) ** 2, axis=1)
            oks_full[i,j] = np.exp(-dis_full/2/delta**2/(scale+1))

        
    idx = np.argmax(oks, axis=0)
    oks = np.max(oks, axis=0)
    oks_full = oks_full[idx, range(kps_count)]

    return oks, oks_full




def showImage(im, kps, gt_boxes, gt_kps):

    oks, oks_full = compute_oks(kps, gt_boxes, gt_kps)

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
            ax.text(x, y - 2, '{:3f}'.format(oks_full[i,j]), 
                    bbox=dict(facecolor='blue', alpha=0.2),
                    fontsize=8, color='white')

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




def showBox(pred_name, type_name):
    with open(pred_name, 'r') as f:
        data = json.load(f)

    with open('data/aichal/val.json', 'r') as fG:
        dataGt = json.load(fG)

    for i, line in enumerate(data):
        if i % 1000 == 0:
            print(i)
            imname = line['image_id']
            im = cv2.imread(os.path.join('data/aichal', 'val', imname + '.jpg'))
            kps = line['keypoint_annotations'].values()

            gt = dataGt[i]
            gt_boxes = []
            gt_kps = []
            for key in gt['keypoint_annotations']:
                gt_boxes.append(gt['human_annotations'][key])
                gt_kps.append(gt['keypoint_annotations'][key])

            showImage(im, kps, gt_boxes, gt_kps)
            plt.savefig('{}{}.png'.format(i, type_name.upper()), 
                    bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type', type=str, help='option: cmu, aichal, fuse, combine')
    args = parser.parse_args()

    if args.type == 'cmu':
        pred = 'pred_cmu_val_1.0.json'
    elif args.type == 'aichal':
        pred = 'pred_cmu_aichal_val_0.6.json'
    elif args.type == 'fuse':
        pred = 'pred_fuse_cmu_val0.8_aichal_val0.6.json'
    elif args.type == 'combine':
        pred = 'pred_fuse_combine_cmu_val0.8_aichal_val0.6.json'
    elif args.type == 'average':
        pred = 'pred_fuse_average_cmu_val0.8_aichal_val0.6.json'
    elif args.type == 'MPI':
        pred = 'pred_MPI_val_0.6.json'
    else:
        print('not in option')
        exit(0)

    showBox(pred, args.type)


