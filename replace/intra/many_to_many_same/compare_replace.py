#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 21:38:03 2020

@author:
"""
from keras.models import load_model
import itertools

from data_type.constants import Constants
from evaluation.accuracy_computer import getModulePredictionAnyToOneUnrolled, getModulePredictionAnyToManyUnrolled, \
    getModulePredictionAnyToMany
from relu_models.one_to_many.one_to_many_util import load_toxic_dataset
from relu_models.one_to_one.one_to_one_util import *
from sklearn.metrics import accuracy_score

from util.common import trainModelAndPredictInBinary, initModularLayers, repopulateModularWeights
from util.nltk_util import loadClincOos, load_pos_tagged_dataset

Constants.enableUnrollMode()

activation = 'gru_models'
bestModelName = 'model1'
faultyModelName = 'model4'
base_path = os.path.dirname(os.path.realpath(__file__))
base_path = os.path.dirname(os.path.dirname(os.path.dirname(base_path)))
faulty_module_path = os.path.join(base_path, activation, 'many_to_many_same', 'modules', faultyModelName)
best_module_path = os.path.join(base_path, activation, 'many_to_many_same', 'modules', bestModelName)

print(base_path)
x_train, x_test, y_train, y_test, num_words, timestep, nb_classes = load_pos_tagged_dataset(hot_encode=False)


out = open(os.path.join(base_path,'replace', 'intra', 'many_to_many_same',"result_"+activation+".csv"), "w")
out.write('Replaced Class,Modularized Accuracy\n')

faulty_model_path = os.path.join(base_path, activation, 'many_to_many_same', 'h5', faultyModelName + '.h5')
faulty_model = load_model(faulty_model_path)

faulty_modules = []
for m in range(nb_classes):
    modularLayers = initModularLayers(faulty_model.layers)
    repopulateModularWeights(modularLayers, faulty_module_path, m)
    faulty_modules.append(modularLayers)

best_model_path = os.path.join(base_path, activation, 'many_to_many_same', 'h5', bestModelName + '.h5')
best_model = load_model(best_model_path)

best_modules = []
for m in range(nb_classes):
    modularLayers = initModularLayers(best_model.layers)
    repopulateModularWeights(modularLayers, best_module_path, m)
    best_modules.append(modularLayers)

for classToReplace in range(nb_classes):

    predictions = []
    for c in range(nb_classes):
        if c != classToReplace:
            predClass1Masked = getModulePredictionAnyToMany(
                faulty_modules[c], x_test, y_test, moduleNo=c)

            predictions.append(predClass1Masked)
        else:
            predClass1Masked = getModulePredictionAnyToMany(
                best_modules[c], x_test, y_test, moduleNo=c)

            predictions.append(predClass1Masked)

    finalPred = []
    for i in range(0, len(y_test)):
        maxPrediction = []
        for ts in range(timestep):
            temp_prediction = []
            for m in range(nb_classes):
                temp_prediction.append(predictions[m][i][ts][m])

            maxPrediction.append(temp_prediction.index(max(temp_prediction)))

        finalPred.append(maxPrediction)

    finalPred = np.asarray(finalPred)
    finalPred = finalPred.flatten()
    scoreModular = accuracy_score(finalPred, y_test.flatten())
    print("After replacing " + str(classToReplace) + " Accuracy: " + str(scoreModular))

    out.write(str(classToReplace) + ',' + str(scoreModular) + '\n')

out.close()

