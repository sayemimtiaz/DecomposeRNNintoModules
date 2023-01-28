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

from data_type.constants import Constants
from evaluation.accuracy_computer import getMonolithicModelAccuracyAnyToMany, getModulePredictionAnyToMany, \
    getModuleAccuracyAnyToManyUnrolled
from evaluation.jaccard_computer import findMeanJaccardIndexRolled, findMeanJaccardIndexUnrolled
from relu_models.many_to_many_same.brown_util import sample


def evaluate_unrolled(model_name):
    # model_name='model1_many_to_one.h5'
    print('evaluating unrolled: ' + model_name)
    import util.nltk_util

    # #
    if 'combined' not in model_name:
        xT, xt, yT, yt, num_words, timestep, nb_classes = util.nltk_util.load_pos_tagged_dataset(hot_encode=False)
    else:
        xT, xt, yT, yt, num_words, timestep, nb_classes = \
            util.nltk_util.load_pos_tagged_dataset_with_clinc(hot_encode=False)

    # xt, yt = sample(xt, yt, nb_classes, 200)
    xt, yt = sample(xt, yt, nb_classes, 50)

    model_path = util.nltk_util.os.path.dirname(util.nltk_util.os.path.realpath(__file__))

    model_name = util.nltk_util.os.path.join(model_path, model_name)

    length = len(yt)

    print('Model accuracy: ' + str(
        getMonolithicModelAccuracyAnyToMany(model_name, xt[:length], yt[:length], skipDummyLabel=False)))

    print("Modularized Accuracy: " + str(
        getModuleAccuracyAnyToManyUnrolled(model_path, model_name, nb_classes, xt, yt, timestep)))

    print('mean jaccard index: ' + str(findMeanJaccardIndexUnrolled(model_name, model_path, nb_classes)))


# Constants.enableUnrollMode()
# evaluate_unrolled('h5/model3.h5')
