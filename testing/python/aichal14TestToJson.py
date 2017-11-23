#!/usr/bin/env python

import cPickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json


limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
           [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
           [1,16], [16,18], [3,17], [6,18]]


def showImage(imname, subset, candidate, idx):

    c = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']


    im = cv2.imread(imname)
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    fig = ax.imshow(im, aspect='equal')
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)


    for i, sub in enumerate(subset):
        score = sub[-2] / sub[-1]
        first = True

        for j in range(18):
            x, y = candidate[sub.astype(int)[j], :2]
            ax.add_patch(
                    plt.Circle((x, y), 3,
                          fill=True,
                          color = c[j % 8], 
                          linewidth=2.0)
                )

            if first:
                ax.text(x, y - 2, '{:3f}'.format(score), 
                        bbox=dict(facecolor='blue', alpha=0.2),
                        fontsize=8, color='white')
                first = False

        for j in range(17):
            index = sub[np.array(limbSeq[j]) - 1]
            if -1 in index:
                continue

            x, y = index.astype(int)
            p0 = candidate[x, :2]
            p1 = candidate[y, :2]

            ax.add_patch(
                    plt.Arrow(p0[0], p0[1], 
                    float(p1[0]) - p0[0], float(p1[1]) - p0[1], 
                    color = c[(17 in limbSeq[j]) and 2 or 0])
                    )
    

    plt.savefig('{}CMU.png'.format(idx), bbox_inches='tight', pad_inches=0)

        


def drawProbCurve(results):
    prob = []

    for result in results:
        subset = result['subset']
        for sub in subset:
            prob.append(sub[-2])# / sub[-1])

    plt.hist(prob, bins=30)
    plt.xlabel('score')
    plt.ylabel('#boxes')
    plt.savefig('aichal14TestSumProbHist.png')


def main(thresh):

    with open('aichal14Test4Scale.pkl', 'rb') as f:
        results = cPickle.load(f)
    
    #drawProbCurve(results)
    #exit(0)
    data = []


    for i, result in enumerate(results):
        test_image = result['test_image']
        subset = result['subset']
        candidate = result['candidate']

        '''
        showImage(test_image, subset, candidate, i)
        print(i)
        if i == 10:
            exit(0)
            
        continue 
        '''

        print(i)
        image = dict()
        imname = test_image.split('/')[-1][:-4]
        image['image_id'] = imname
        kp_ann = dict()

        idx = 0
        for sub in subset:
            score = sub[-2] #/ sub[-1]
            if score >= thresh:
                idx += 1

                kp = []

                for j in range(14):
                    x = sub[j].astype(int)
                    if x == -1:
                        kp += [0, 0, 3]
                    else:
                        kp += list(candidate[x][:2])
                        kp.append(1)


                kp_ann['human' + str(idx)] = map(int, kp)

        
        image['keypoint_annotations'] = kp_ann
        data.append(image)

    with open('pred_cmu_aichal_test_{}.json'.format(thresh), 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
#    for thresh in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]:
    for thresh in [4]: #[3, 4, 5, 10, 15, 20]:
        main(thresh)

