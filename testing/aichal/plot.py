#!/usr/bin/env python


import cPickle
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json


def drawProbCurve(probSum):

    with open('aichal14Val4Scale.pkl', 'rb') as f:
        results = cPickle.load(f)
    with open('aichal14_40Val4Scale.pkl', 'rb') as f:
        results40 = cPickle.load(f)

    prob = []
    for result in results:
        subset = result['subset']
        for sub in subset:
            if probSum:
                prob.append(sub[-2])
            else:
                prob.append(sub[-2] / sub[-1])


    prob40 = []
    for result in results40:
        subset = result['subset']
        for sub in subset:
            if probSum:
                prob40.append(sub[-2])
            else:
                prob40.append(sub[-2] / sub[-1])

    plt.figure()
    plt.hist(prob, bins=30, alpha=0.7, label='30W')
    plt.hist(prob40, bins=30, alpha=0.7, label='40W')
    plt.legend(loc='upper right', fontsize=10)
    plt.xlabel('score')
    plt.ylabel('#boxes')

    if probSum:
        plt.savefig('aichal14_40ValSumProbHist.png')
    else:
        plt.savefig('aichal14_40ValProbHist.png')


if __name__ == '__main__':
    drawProbCurve(False)
    drawProbCurve(True)

