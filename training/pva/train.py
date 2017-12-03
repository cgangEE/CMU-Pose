#!/usr/bin/env python

import os, cv2
#os.environ['GLOG_minloglevel'] = '1'
import sys
import numpy as np
sys.path.insert(0, '/home/cgangee/code/Realtime_Multi-Person_Pose_Estimation/caffe_train/python')

import caffe


def trainModel():

    caffe.set_mode_gpu()
#    weights = 'pose_6stage.model'
    weights = 'model/_iter_1000.caffemodel'

    solver = caffe.SGDSolver('pose_solver.prototxt')
    solver.net.copy_from(weights)

    net = solver.net

    for i in range(1000):
        solver.step(1)
        continue
        if i % 50 != 0:
            continue

        img = net.blobs['image'].data
        img = img[0].transpose((1, 2, 0))
        print(img.shape)
        cv2.imwrite('{}_img.jpg'.format(i), (img * 256) + 128 )

        heat_temp = net.blobs['label_heat'].data
        vec_temp = net.blobs['label_vec'].data
        print(heat_temp.shape)
        print(vec_temp.shape)
        for j in range(15):
            heat = heat_temp[0, j].copy()
            heat = cv2.resize(heat, (0,0), fx=8, fy=8) 

            cv2.imwrite('{}_heat_{}.jpg'.format(i, j), heat * 255)


        for j in range(13):
            vec_x = vec_temp[0, j*2]
            vec_x = cv2.resize(vec_x, (0,0), fx=8, fy=8)

            vec_y = vec_temp[0, j*2+1]
            vec_y = cv2.resize(vec_y, (0,0), fx=8, fy=8)

            cv2.imwrite('{}_vec_x_{}.jpg'.format(i, j), vec_x * 255)
            cv2.imwrite('{}_vec_x_{}.jpg'.format(i, j), vec_y * 255)

        vec = np.sum(vec_temp[0], axis=0)

        cv2.imwrite('{}_vec.jpg'.format(i), np.sum(vec_temp[0], axis=0) * 255)
            



        stage6_heat = net.blobs['output/stage6/L2'].data

        for j in range(15):
            heat = stage6_heat[0, j].copy()
            heat = cv2.resize(heat, (0,0), fx=8, fy=8) 
            cv2.imwrite('{}_stage6_heat_{}.jpg'.format(i, j), heat * 255)



if __name__ == '__main__':
    trainModel()

