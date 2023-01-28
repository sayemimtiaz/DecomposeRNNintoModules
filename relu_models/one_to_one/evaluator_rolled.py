#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 21:38:03 2020

@author:
"""
import os

from keras.models import load_model

from evaluation.accuracy_computer import getModulePredictionAnyToOneUnrolled, getMonolithicModelAccuracyAnyToMany, \
    getMonolithicModelAccuracyAnyToOne
from evaluation.jaccard_computer import findMeanJaccardIndexRolled
from relu_models.one_to_one.one_to_one_util import load_math_dataset, load_math_dataset_with_toxic
from util.common import initModularLayers, repopulateModularWeights, extract_model_name


def evaluate_rolled(model_name):
    # model_name = 'model4_combined.h5'
    print('evaluating rolled: ' + model_name)
    model_path = os.path.dirname(os.path.realpath(__file__))
    print(model_path)
    model_name = os.path.join(model_path, model_name)

    if 'combined' not in model_name:
        xT, xt, yT, yt, num_words, timestep, nb_classes = load_math_dataset(hot_encode=False)
    else:
        xT, xt, yT, yt, num_words, timestep, nb_classes = load_math_dataset_with_toxic(hot_encode=False)

    labs = range(0, nb_classes)
    model = load_model(model_name)

    modules = []
    for m in labs:
        # modules.append(load_model('modules/module' + str(m) + '.h5'))
        modularLayers = initModularLayers(model.layers)
        repopulateModularWeights(modularLayers, os.path.join(model_path, 'modules', extract_model_name(model_name)), m)
        modules.append(modularLayers)

    finalPred = []
    length = len(yt)
    p = []
    for m in labs:
        # p.append(modules[m].predict(xt[:length]))
        p.append(getModulePredictionAnyToOneUnrolled(modules[m], xt, yt, m))

    for i in range(0, length):
        maxPrediction = []
        for m in labs:
            maxPrediction.append(p[m][i][m])

        finalPred.append(maxPrediction.index(max(maxPrediction)))

    from sklearn.metrics import accuracy_score

    score = accuracy_score(finalPred, yt[:length])
    print("Modularized Accuracy: " + str(score))
    # pred = model.predict(xt[:length])
    # pred = pred.argmax(axis=-1)
    # score = accuracy_score(pred, yt[:length])
    score = getMonolithicModelAccuracyAnyToOne(model_name, xt, yt)
    print("Model Accuracy: " + str(score))

    print('mean jaccard index: ' + str(findMeanJaccardIndexRolled(model_name, model_path, nb_classes)))


# evaluate_rolled('h5/model4_combined.h5')
