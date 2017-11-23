#!/usr/bin/env sh
/home/cgangee/code/Realtime_Multi-Person_Pose_Estimation/caffe_train/build/tools/caffe	 train  --solver=pose_solver.prototxt  --gpu=0 --weights=model/_iter_100000.caffemodel > log_aichal2 2>&1
