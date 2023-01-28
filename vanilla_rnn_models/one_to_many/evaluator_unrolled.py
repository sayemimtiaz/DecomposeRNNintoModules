#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 21:38:03 2020

@author:
"""
from evaluation.accuracy_computer import getMonolithicModelAccuracyAnyToMany, \
    getModuleAccuracyAnyToManyUnrolled
from evaluation.jaccard_computer import findMeanJaccardIndexUnrolled
from relu_models.one_to_many.one_to_many_util import load_toxic_dataset, load_toxic_dataset_combine_math_qa


def evaluate_unrolled(model_name):
    print('evaluating unrolled: ' + model_name)
    # model_name = 'model4_combined.h5'

    import util.nltk_util

    if 'combined' not in model_name:
        xT, xt, yT, yt, num_words, timestep, nb_classes = load_toxic_dataset(hot_encode=False,
                                                                             repeat=False)
    else:
        xT, xt, yT, yt, num_words, timestep, nb_classes = load_toxic_dataset_combine_math_qa(hot_encode=False,
                                                                                             repeat=False)

    # #

    # xt, yt = sample(xt, yt, nb_classes, 5000)

    model_path = util.nltk_util.os.path.dirname(util.nltk_util.os.path.realpath(__file__))

    model_name = util.nltk_util.os.path.join(model_path, model_name)

    length = len(yt)

    print('Model accuracy: ' + str(getMonolithicModelAccuracyAnyToMany(model_name, xt[:length], yt[:length],
                                                                       skipDummyLabel=False)))

    print("Modularized Accuracy: " + str(
        getModuleAccuracyAnyToManyUnrolled(model_path, model_name, nb_classes, xt, yt, timestep)))

    print('mean jaccard index: ' + str(findMeanJaccardIndexUnrolled(model_name, model_path, nb_classes)))


# Constants.enableUnrollMode()
# evaluate_unrolled('h5/model2.h5')
