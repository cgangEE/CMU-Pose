#!/usr/bin/python

import google.protobuf.text_format
import google.protobuf as pb
import caffe
import os

def modifyModel(src_model, dst_model):
    fOut = open(dst_model, 'w')

    for stage in [1,2,3,4,5,6]:
        for L in ['L1','L2']:
            model = caffe.proto.caffe_pb2.NetParameter()
            with open(src_model) as f:
                pb.text_format.Merge(f.read(), model)
            
            if stage == 1 or L == 'L2':
                model.layer.remove(model.layer[0])
            if L == 'L1':
                for i in range(2):
                    model.layer.remove(model.layer[-1])
            else:
                for i in range(2):
                    model.layer.remove(model.layer[-3])
                 
            for i, layer in enumerate(model.layer):

                if layer.name == 'output':
                    if L == 'L1':
                        num = 26
                    else:
                        num = 15
                    layer.convolution_param.num_output = num

                suffix = '/stage{}/{}'.format(stage, L)

                for j, top in enumerate(layer.top):
                    layer.top[j] += suffix

                for j, bottom in enumerate(layer.bottom):

                    if layer.bottom[j] == 'conv3_4':
                        if stage >= 2 and layer.name != 'conv3_4':
                            layer.bottom[j] += '/stage{}/L1'.format(stage)
                        continue

                    if layer.name == 'conv3_4' and layer.bottom[j] == 'output':
                        layer.bottom[j] += '/stage{}/L{}'.format(stage-1, j+1)
                        continue

                    if layer.name in ['loss', 'weight'] and j == 1:
                        continue

                    layer.bottom[j] += suffix

                layer.name += suffix
            fOut.write(pb.text_format.MessageToString(model))

    fOut.close()

if __name__ == '__main__':
    modifyModel('stage1_tmp.pt', 'pose_6stage_tmp.pt')

