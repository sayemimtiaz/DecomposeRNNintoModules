#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 21:38:03 2020

@author:
"""
import os

from keras.models import load_model
import itertools

from evaluation.accuracy_computer import getMonolithicModelPredictionAnyToOne
from relu_models.one_to_one.one_to_one_util import *
from sklearn.metrics import accuracy_score

from util.common import trainModelAndPredictInBinary

activation = 'gru_models'
bestModelName = 'model1'
faultyModelName = 'model4'
base_path = os.path.dirname(os.path.realpath(__file__))
base_path = os.path.dirname(os.path.dirname(os.path.dirname(base_path)))
faulty_module_path = os.path.join(base_path, activation, 'one_to_one', 'modules', faultyModelName)
best_module_path = os.path.join(base_path, activation, 'one_to_one', 'modules', bestModelName)

print(base_path)
x_train, x_test, y_train, y_test, num_words, timestep, nb_classes = load_math_dataset(hot_encode=False)

out = open(os.path.join(base_path, 'replace', 'intra', 'one_to_one', "result_" + activation + ".csv"), "w")
out.write('Replaced Class,Modularized Accuracy\n')

for classToReplace in range(nb_classes):

    predictions = []
    for c in range(nb_classes):
        if c != classToReplace:
            module = load_model(os.path.join(faulty_module_path, 'module' + str(c) + '.h5'))
            # predictions.append(module.predict(x_test))
            predictions.append(getMonolithicModelPredictionAnyToOne(module, x_test, y_test))
        else:
            module = load_model(os.path.join(best_module_path, 'module' + str(c) + '.h5'))
            # predictions.append(module.predict(x_test))
            predictions.append(getMonolithicModelPredictionAnyToOne(module, x_test, y_test))

    finalPred = []
    for i in range(0, len(y_test)):
        temp_pred = []
        maxPr = 0.0
        maxClass = -1
        for c in range(nb_classes):
            # if c != 0:
            #     temp_c_no = predictions[c][i][0]
            # else:
            #     temp_c_no = predictions[c][i][1]
            temp_c = predictions[c][i][c]
            # temp_c = temp_c / temp_c_no
            temp_pred.append(temp_c)

            # if predictions[c][i].argmax()==c:
            #     if temp_c>maxPr:
            #         maxPr=temp_c
            #         maxClass=c

        if maxClass != -1:
            finalPred.append(maxClass)
        else:
            finalPred.append(temp_pred.index(max(temp_pred)))

    scoreModular = accuracy_score(finalPred, y_test)
    print("After replacing " + str(classToReplace) + " Accuracy: " + str(scoreModular))

    out.write(str(classToReplace) + ',' + str(scoreModular) + '\n')

out.close()
