#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 21:38:03 2020

@author:
"""
import os

from keras.models import load_model

from evaluation.accuracy_computer import getModulePredictionAnyToOneUnrolled, \
    getMonolithicModelAccuracyAnyToOne
from evaluation.jaccard_computer import findMeanJaccardIndexRolled
from util.common import initModularLayers, repopulateModularWeights


def evaluate_rolled(model_name):
    # model_name = 'model2_original.h5'
    print('evaluating rolled: ' + model_name)
    import util.nltk_util

    if 'combined' not in model_name:
        xT, xt, yT, yt, num_words, timestep, nb_classes = util.nltk_util.loadClincOos(hot_encode=False, timestep=15)
    else:
        xT, xt, yT, yt, num_words, timestep, nb_classes = util.nltk_util.loadClincOosWithPosTag(hot_encode=False,
                                                                                                timestep=15)

    model_path = util.nltk_util.os.path.dirname(util.nltk_util.os.path.realpath(__file__))
    model_name = util.nltk_util.os.path.join(model_path, model_name)

    labs = range(0, nb_classes)
    model = load_model(model_name)
    modules = []
    # modules1=[]
    for m in labs:
        # modules1.append(load_model('modules/module' + str(m) + '.h5'))
        modularLayers = initModularLayers(model.layers)
        repopulateModularWeights(modularLayers, os.path.join(model_path, 'modules'), m)
        modules.append(modularLayers)

    finalPred = []
    length = len(yt)
    p = []
    p1 = []
    for m in labs:
        # p1.append(modules1[m].predict(xt[:length]))
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

#
# Constants.disableUnrollMode()
# evaluate_rolled('h5/model1_combined.h5')
