#!/usr/bin/env sh
/home/cgangee/code/Realtime_Multi-Person_Pose_Estimation/caffe_train/build/tools/caffe	 train  --solver=pose_solver.prototxt  --gpu=0 --weights=myModel.model > log_aichal 2>&1
