#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 21:38:03 2020

@author:
"""
import time
import numpy as np
import itertools

from keras.models import load_model
from sklearn.metrics import accuracy_score

from evaluation.accuracy_computer import getMonolithicModelAccuracyAnyToMany, getModulePredictionAnyToMany
from evaluation.jaccard_computer import findMeanJaccardIndexRolled
from relu_models.many_to_many_same.brown_util import sample
from util.common import initModularLayers, extract_model_name


def evaluate_rolled(model_name):
    print('evaluating rolled: '+model_name)
    import util.nltk_util

    # #
    if 'combined' not in model_name:
        xT, xt, yT, yt, num_words, timestep, nb_classes = util.nltk_util.load_pos_tagged_dataset(hot_encode=False)
    else:
        xT, xt, yT, yt, num_words, timestep, nb_classes = \
            util.nltk_util.load_pos_tagged_dataset_with_clinc(hot_encode=False)

    xt, yt = sample(xt, yt, nb_classes, 50)

    model_path = util.nltk_util.os.path.dirname(util.nltk_util.os.path.realpath(__file__))

    model_name = util.nltk_util.os.path.join(model_path, model_name)

    labs = range(0, nb_classes)

    finalPred = []
    length = len(yt)

    p = []
    for m in labs:
        model = load_model(util.nltk_util.os.path.join(model_path, 'modules',extract_model_name(model_name),
                                                       'module' + str(m) + '.h5'))
        modelLayers = initModularLayers(model.layers)
        p.append(getModulePredictionAnyToMany(modelLayers, xt[:length], yt[:length], moduleNo=m))

    for i in range(0, length):

        maxPrediction = []
        for ts in range(timestep):
            temp_prediction = []
            for m in labs:
                temp_prediction.append(p[m][i][ts][m])

            maxPrediction.append(temp_prediction.index(max(temp_prediction)))

        finalPred.append(maxPrediction)

    print('Model accuracy: ' + str(getMonolithicModelAccuracyAnyToMany(model_name, xt[:length], yt[:length],
                                                                       skipDummyLabel=False)))

    finalPred = util.nltk_util.np.asarray(finalPred)
    finalPred = finalPred.flatten()
    yt = yt[:length]
    yt = yt.flatten()
    score = accuracy_score(finalPred, yt)
    print("Modularized Accuracy: " + str(score))

    print('mean jaccard index: ' + str(findMeanJaccardIndexRolled(model_name, model_path, nb_classes)))
