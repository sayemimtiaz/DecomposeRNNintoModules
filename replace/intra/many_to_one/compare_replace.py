#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 21:38:03 2020

@author:
"""
from keras.models import load_model
import itertools

from data_type.constants import Constants
from evaluation.accuracy_computer import getModulePredictionAnyToOneUnrolled
from relu_models.one_to_one.one_to_one_util import *
from sklearn.metrics import accuracy_score

from util.common import trainModelAndPredictInBinary, initModularLayers, repopulateModularWeights
from util.nltk_util import loadClincOos

Constants.enableUnrollMode()
activation = 'gru_models'
bestModelName = 'model1'
faultyModelName = 'model4'
base_path = os.path.dirname(os.path.realpath(__file__))
base_path = os.path.dirname(os.path.dirname(os.path.dirname(base_path)))
faulty_module_path = os.path.join(base_path, activation, 'many_to_one', 'modules', faultyModelName)
best_module_path = os.path.join(base_path, activation, 'many_to_one', 'modules', bestModelName)

print(base_path)
x_train, x_test, y_train, y_test, num_words, timestep, nb_classes = loadClincOos(hot_encode=False)

out = open(os.path.join(base_path, 'replace', 'intra', 'many_to_one', "result_"+activation+".csv"), "w")
out.write('Replaced Class,Modularized Accuracy\n')

faulty_model_path = os.path.join(base_path, activation, 'many_to_one', 'h5', faultyModelName + '.h5')
faulty_model = load_model(faulty_model_path)

faulty_modules = {}
for m in range(nb_classes):
    modularLayers = initModularLayers(faulty_model.layers)
    repopulateModularWeights(modularLayers, faulty_module_path, m)
    faulty_modules[m]=modularLayers
    # faulty_modules[m] = load_model(os.path.join(faulty_module_path, 'module' + str(m) + '.h5'))

best_model_path = os.path.join(base_path, activation, 'many_to_one', 'h5', bestModelName + '.h5')
best_model = load_model(best_model_path)

best_modules = {}
for m in range(nb_classes):
    modularLayers = initModularLayers(best_model.layers)
    repopulateModularWeights(modularLayers, best_module_path, m)
    best_modules[m]=modularLayers
    # best_modules[m] = load_model(os.path.join(best_module_path, 'module' + str(m) + '.h5'))

for classToReplace in range(nb_classes):

    predictions = []
    for c in range(nb_classes):
        if c != classToReplace:
            # predictions.append(faulty_modules[c].predict(x_test[:len(y_test)]))
            predictions.append(getModulePredictionAnyToOneUnrolled(faulty_modules[c], x_test, y_test, moduleNo=c))

        else:
            # predictions.append(best_modules[c].predict(x_test[:len(y_test)]))
            predictions.append(getModulePredictionAnyToOneUnrolled(best_modules[c], x_test, y_test, moduleNo=c))

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

            if predictions[c][i].argmax() == c:
                if temp_c > maxPr:
                    maxPr = temp_c
                    maxClass = c

        if maxClass != -1:
            finalPred.append(maxClass)
        else:
            finalPred.append(temp_pred.index(max(temp_pred)))

    scoreModular = accuracy_score(finalPred, y_test)
    print("After replacing " + str(classToReplace) + " Accuracy: " + str(scoreModular))

    out.write(str(classToReplace) + ',' + str(scoreModular) + '\n')

out.close()
