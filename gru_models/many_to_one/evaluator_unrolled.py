#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 21:38:03 2020

@author:
"""
import os
import numpy as np
import itertools

from keras.models import load_model
from sklearn.metrics import accuracy_score

from data_type.constants import Constants
from evaluation.accuracy_computer import getModuleAccuracyAnyToOneUnrolled, getMonolithicModelAccuracyAnyToMany, \
    getMonolithicModelAccuracyAnyToOne
from evaluation.jaccard_computer import findMeanJaccardIndexUnrolled


def evaluate_unrolled(model_name):
    # model_name = 'model2_original.h5'
    print('evaluating unrolled: ' + model_name)
    import util.nltk_util
    model = load_model(model_name)
    model_path=os.path.dirname(os.path.realpath(__file__))
    model_name=os.path.join(model_path,  model_name)

    from util.nltk_util import loadClincOos,loadClincOosWithPosTag

    if 'combined' not in model_name:
        xT, xt, yT, yt, num_words, timestep, nb_classes = loadClincOos(hot_encode=False, timestep=15)
    else:
        xT, xt, yT, yt, num_words, timestep, nb_classes = loadClincOosWithPosTag(hot_encode=False,
                                                                                                   timestep=15)

    labs = range(0, nb_classes)

    base_path = os.path.dirname(os.path.realpath(__file__))

    print("Modularized Accuracy: " + str(getModuleAccuracyAnyToOneUnrolled(base_path,
                                                                           model_name, nb_classes, xt, yt)))
    # pred = model.predict(xt)
    # pred = pred.argmax(axis=-1)
    # score = accuracy_score(pred, yt)
    score = getMonolithicModelAccuracyAnyToOne(model_name, xt, yt)
    print("Model Accuracy: " + str(score))

    print('mean jaccard index: ' + str(findMeanJaccardIndexUnrolled(model_name, model_path, nb_classes)))


# Constants.enableUnrollMode()
# model_name = 'h5/model2.h5'
# evaluate_unrolled(model_name)