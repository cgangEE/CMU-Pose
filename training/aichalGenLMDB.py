#!/usr/bin/env python

import scipy.io as sio
import numpy as np
import json
import cv2
import lmdb
import sys, os
# change your caffe path here
# sys.path.insert(0, os.path.join('/home/zhecao/caffe/', 'python/'))
import caffe
import os.path
import struct

def writeLMDB():
	lmdb_path = 'dataset/aichal/lmdb'

	env = lmdb.open(lmdb_path, map_size=int(1e12))
	txn = env.begin(write=True)
	data = []
	numSample = 0

	with open('dataset/aichal/train.json') as data_file:
#	with open('aichalSmall.json') as data_file:
		data = json.load(data_file)
	
	'''
	data = data[:2]
	with open('aichalSmall.json', 'w') as f:
		json.dump(data, f)
	exit(0)	
	'''
	
	numSample = len(data)
	print numSample	


	random_order = np.random.permutation(numSample).tolist()
	
	totalWriteCount = len(data)
	print totalWriteCount;
	writeCount = 0


	for count in range(numSample):
		idx = random_order[count]

		path_header = '/home/cgangee/code/Realtime_Multi-\
Person_Pose_Estimation/training/dataset/aichal/train/'

		print os.path.join(path_header, data[idx]['image_id'] + '.jpg')

		img = cv2.imread(os.path.join(path_header, data[idx]['image_id'] + '.jpg'))
		mask_all =  np.ones(img.shape[:2], dtype=np.uint8) * 255
		mask_miss = np.ones(img.shape[:2], dtype=np.uint8) * 255


		height = img.shape[0]
		width = img.shape[1]

		data[idx]['img_height'] = float(img.shape[0])
		data[idx]['img_width'] = float(img.shape[1])

		if(width < 64):
			img = cv2.copyMakeBorder(img,0,0,0,64-width,cv2.BORDER_CONSTANT,value=(128,128,128))
			print 'saving padded image!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
			cv2.imwrite('padded_img.jpg', img)
			width = 64
			# no modify on width, because we want to keep information

		meta_data = np.zeros(shape=(height,width,1), dtype=np.uint8)
		#print type(img), img.shape
		#print type(meta_data), meta_data.shape

		data[idx]['dataset'] = 'aichal'
		clidx = 0 # current line index
		# dataset name (string)
		for i in range(len(data[idx]['dataset'])):
			meta_data[clidx][i] = ord(data[idx]['dataset'][i])
		clidx = clidx + 1


		# image height, image width
		height_binary = float2bytes(data[idx]['img_height'])
		for i in range(len(height_binary)):
			meta_data[clidx][i] = ord(height_binary[i])
		width_binary = float2bytes(data[idx]['img_width'])
		for i in range(len(width_binary)):
			meta_data[clidx][4+i] = ord(width_binary[i])
		clidx = clidx + 1


		# Gang Chen process data

		kps = data[idx]['keypoint_annotations']
		bboxs = data[idx]['human_annotations']
		data[idx]['isValidation'] = 0
		data[idx]['numOtherPeople'] = len(kps) - 1
		data[idx]['people_index'] = idx
		data[idx]['annolist_index'] = float(idx)

		keys = kps.keys()
		x1, y1, x2, y2 = bboxs[keys[0]]
		data[idx]['objpos'] = [(x1 + x2) / 2, (y1 + y2) / 2]
		data[idx]['scale_provided'] = (y2 - y1 + 1) / 368.0

		kp = np.array(kps[keys[0]]).reshape(14, 3)
		kp[kp[:,2] == 2, 2] = 0
		kp[kp[:,2] == 3, 2] = 2
		data[idx]['joint_self'] = kp.tolist()

		data[idx]['joint_others'] = []
		data[idx]['objpos_other'] = []
		data[idx]['scale_provided_other'] = []

		for i in range(1, len(keys)):

			key = keys[i]
			x1, y1, x2, y2 = bboxs[key]

			data[idx]['objpos_other'].append([(x1 + x2) / 2, (y1 + y2) / 2])
			data[idx]['scale_provided_other'].append((y2 - y1 + 1) / 368.0)

			kp = np.array(kps[key]).reshape(14, 3)
			kp[kp[:,2] == 2, 2] = 0
			kp[kp[:,2] == 3, 2] = 2
			data[idx]['joint_others'].append(kp.tolist())

		# end Gang Chen process data


		# (a) isValidation(uint8), numOtherPeople (uint8), people_index (uint8), annolist_index (float), writeCount(float), totalWriteCount(float)
		meta_data[clidx][0] = data[idx]['isValidation']
		meta_data[clidx][1] = data[idx]['numOtherPeople']
		meta_data[clidx][2] = data[idx]['people_index']
		annolist_index_binary = float2bytes(data[idx]['annolist_index'])
		for i in range(len(annolist_index_binary)): # 3,4,5,6
			meta_data[clidx][3+i] = ord(annolist_index_binary[i])
		count_binary = float2bytes(float(writeCount)) # note it's writecount instead of count!
		for i in range(len(count_binary)):
			meta_data[clidx][7+i] = ord(count_binary[i])
		totalWriteCount_binary = float2bytes(float(totalWriteCount))
		for i in range(len(totalWriteCount_binary)):
			meta_data[clidx][11+i] = ord(totalWriteCount_binary[i])
		nop = int(data[idx]['numOtherPeople'])
		clidx = clidx + 1


		# (b) objpos_x (float), objpos_y (float)
		objpos_binary = float2bytes(data[idx]['objpos'])
		for i in range(len(objpos_binary)):
			meta_data[clidx][i] = ord(objpos_binary[i])
		clidx = clidx + 1


		# (c) scale_provided (float)
		scale_provided_binary = float2bytes(data[idx]['scale_provided'])
		for i in range(len(scale_provided_binary)):
			meta_data[clidx][i] = ord(scale_provided_binary[i])
		clidx = clidx + 1


		# (d) joint_self (3*16) (float) (3 line)
		joints = np.asarray(data[idx]['joint_self']).T.tolist() # transpose to 3*16
		for i in range(len(joints)):
			row_binary = float2bytes(joints[i])
			for j in range(len(row_binary)):
				meta_data[clidx][j] = ord(row_binary[j])
			clidx = clidx + 1


		# (e) check nop, prepare arrays
		if(nop!=0):
			joint_other = data[idx]['joint_others']
			objpos_other = data[idx]['objpos_other']
			scale_provided_other = data[idx]['scale_provided_other']


			# (f) objpos_other_x (float), objpos_other_y (float) (nop lines)
			for i in range(nop):
				objpos_binary = float2bytes(objpos_other[i])
				for j in range(len(objpos_binary)):
					meta_data[clidx][j] = ord(objpos_binary[j])
				clidx = clidx + 1


			# (g) scale_provided_other (nop floats in 1 line)
			scale_provided_other_binary = float2bytes(scale_provided_other)
			for j in range(len(scale_provided_other_binary)):
				meta_data[clidx][j] = ord(scale_provided_other_binary[j])
			clidx = clidx + 1


			# (h) joint_others (3*16) (float) (nop*3 lines)
			for n in range(nop):
				joints = np.asarray(joint_other[n]).T.tolist() # transpose to 3*16
				for i in range(len(joints)):
					row_binary = float2bytes(joints[i])
					for j in range(len(row_binary)):
						meta_data[clidx][j] = ord(row_binary[j])
					clidx = clidx + 1
		

		# print meta_data[0:12,0:48] 
		# total 7+4*nop lines
		img4ch = np.concatenate((img, meta_data, mask_miss[...,None], mask_all[...,None]), axis=2)

		img4ch = np.transpose(img4ch, (2, 0, 1))
		print img4ch.shape
		
		datum = caffe.io.array_to_datum(img4ch, label=0)
		key = '%07d' % writeCount
		txn.put(key, datum.SerializeToString())
		if(writeCount % 1000 == 0):
			txn.commit()
			txn = env.begin(write=True)
		print '%d/%d/%d/%d' % (count,writeCount,idx,numSample)
		writeCount = writeCount + 1

	txn.commit()
	env.close()

def float2bytes(floats):
	if type(floats) is float:
		floats = [floats]
	return struct.pack('%sf' % len(floats), *floats)

if __name__ == "__main__":
	writeLMDB()
