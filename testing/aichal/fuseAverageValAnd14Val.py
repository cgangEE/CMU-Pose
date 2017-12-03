#!/usr/bin/env python

import json
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt



def compute_dis(kps_cmu, kps_aichal):

    delta = 2*np.array([0.01388152, 0.01515228, 0.01057665, 0.01417709, \
        0.01497891, 0.01402144, 0.03909642, 0.03686941, 0.01981803, \
        0.03843971, 0.03412318, 0.02415081, 0.01291456, 0.01236173])


    kps_count = len(kps_cmu)
    gt_count = len(kps_aichal)
    oks = np.zeros((gt_count, kps_count))


    for i, key in enumerate(kps_aichal):
        kp_aichal = np.reshape(kps_aichal[key], (14, 3))

        for j, key in enumerate(kps_cmu):
            kp_cmu = np.reshape(kps_cmu[key], (14, 3))

            dis = np.sum((kp_cmu[:, :2] - kp_aichal[:, :2]) ** 2, axis=1)
            visible =  np.logical_and(kp_cmu[:, 2] != 3, kp_aichal[:, 2] != 3)
            oks[i, j] = np.mean(dis[visible])
        
    if len(oks) == 0:
        return np.array([]), kps_cmu
    idx = np.argmin(oks, axis=0)
    oks = np.min(oks, axis=0)



    for j, key in enumerate(kps_cmu):
        kp_aichal = np.reshape(kps_aichal.values()[idx[j]], (14, 3))
        kp_cmu = np.reshape(kps_cmu[key], (14, 3))
        invisible = kp_cmu[:, 2] == 3
        kp_cmu[invisible] = kp_aichal[invisible]
        
        visible =  np.logical_and(kp_cmu[:, 2] != 3, kp_aichal[:, 2] != 3)
        kp_cmu[visible] = np.mean(np.array([kp_cmu[visible], 
                kp_aichal[visible]]), axis=0)
        kps_cmu[key] = list(np.reshape(kp_cmu, 42))
    

    return oks, kps_cmu



def fuse(cmu_idx, aichal_idx):

    cmu_name = 'pred_cmu_val_{}.json'.format(cmu_idx)
    aichal_name = 'pred_cmu_aichal_v2_40_val_{}.json'.format(aichal_idx)

    with open(cmu_name, 'r') as f:
        cmu = json.load(f)

    with open(aichal_name, 'r') as f:
        aichal = json.load(f)

    pred = []


    for i, cmu_ann in enumerate(cmu):
        kps_cmu = cmu_ann['keypoint_annotations']
        kps_aichal = aichal[i]['keypoint_annotations']

        dis, kps_cmu = compute_dis(kps_cmu, kps_aichal)
        idx = np.where(dis < 1e4)
        
        
        kps = np.array(kps_cmu.values())[idx]

        pred_ann = {}
        pred_ann['image_id'] = cmu_ann['image_id']
        pred_kps = {}

        for j, kp in enumerate(kps):
            kp_name = 'human{}'.format(j)
            pred_kps[kp_name] = list(kp)

        pred_ann['keypoint_annotations'] = pred_kps
        pred.append(pred_ann)

    with open('pred_fuse_average_cmu_val{}_aichal_val_40_{}.json'.format(cmu_idx, aichal_idx)
            , 'w') as f:
        json.dump(pred, f)

if __name__ == '__main__':
    for cmu_idx in [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]:
        for aichal_idx in [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]:
            fuse(cmu_idx, aichal_idx)
    
