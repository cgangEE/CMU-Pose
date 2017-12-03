#!/usr/bin/env bash
/home/gchen/code/Realtime_Multi-Person_Pose_Estimation/caffe_train/build/tools/caffe	 train  --solver=pose_solver.prototxt  --gpu=0,1 --weights=../../model/_trained_COCO/pose_iter_440000.caffemodel > log_aichal 2>&1
