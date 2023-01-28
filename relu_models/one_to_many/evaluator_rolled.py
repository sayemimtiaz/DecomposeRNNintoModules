#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 21:38:03 2020

@author:
"""
import os
import time
import numpy as np
import itertools

from keras.models import load_model
from sklearn.metrics import accuracy_score

from evaluation.accuracy_computer import getMonolithicModelAccuracyAnyToMany, getModulePredictionAnyToMany
from evaluation.jaccard_computer import findMeanJaccardIndexRolled
from relu_models.many_to_many_same.brown_util import sample
from relu_models.one_to_many.one_to_many_util import loadProcessedDataset, load_toxic_dataset, \
    load_toxic_dataset_combine_math_qa
from util.common import initModularLayers, extract_model_name


def evaluate_rolled(model_name):
    print('evaluating rolled: '+model_name)
    # model_name='model4.h5'

    xT, xt, yT, yt, num_words, timestep, nb_classes = load_toxic_dataset(hot_encode=False, repeat=False)
    # xT, xt, yT, yt, num_words, timestep, nb_classes = load_toxic_dataset_combine_math_qa(hot_encode=False,
    # repeat=False)

    # xT, xt, yT, yt, timestep, nb_classes = loadProcessedDataset(flatten=True, hot_encode=False,
    #                                                                               dataPath='VOC2006')
    # #

    # xt,yt=sample(xt,yt,nb_classes,1000)

    model_path = os.path.dirname(os.path.realpath(__file__))
    # print(model_path)
    model_name = os.path.join(model_path, model_name)

    labs = range(0, nb_classes)

    finalPred = []
    length = len(yt)
    p = []

    # print('Model accuracy (Skipped dummy): ' + str(getMonolithicModelAccuracyAnyToMany(model_name, xt[:length],
    #                                                                    yt[:length], skipDummyLabel=True)))

    for m in labs:
        model = load_model(os.path.join(model_path, 'modules', extract_model_name(model_name),
                                        'module' + str(m) + '.h5'))
        modelLayers = initModularLayers(model.layers)
        p.append(getModulePredictionAnyToMany(modelLayers, xt[:length],
                                              yt[:length], moduleNo=m))

    for i in range(0, length):

        maxPrediction = []
        for ts in range(timestep):

            temp_prediction = []
            for m in labs:
                temp_prediction.append(p[m][i][ts][m])

            maxPrediction.append(temp_prediction.index(max(temp_prediction)))

        finalPred.append(maxPrediction)

    print('Model accuracy: ' + str(getMonolithicModelAccuracyAnyToMany(model_name, xt[:length],
                                                                       yt[:length], skipDummyLabel=False)))

    finalPred = np.asarray(finalPred)
    finalPred = finalPred.flatten()
    yt = yt[:length]
    yt = yt.flatten()
    score = accuracy_score(finalPred, yt)
    print("Modularized Accuracy: " + str(score))

    print('mean jaccard index: ' + str(findMeanJaccardIndexRolled(model_name, model_path, nb_classes)))
