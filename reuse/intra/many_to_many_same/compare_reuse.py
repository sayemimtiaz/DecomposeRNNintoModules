#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 21:38:03 2020

@author:
"""
from imblearn.under_sampling import RandomUnderSampler
from keras.models import load_model
import itertools
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score
import os
import numpy as np

from data_type.constants import Constants
from evaluation.accuracy_computer import getModulePredictionAnyToManyUnrolled, getModulePredictionAnyToMany
from util.common import initModularLayers, repopulateModularWeights, binarize_multi_label, \
    trainModelAndPredictInBinaryForManyOutput
from util.nltk_util import load_pos_tagged_dataset, binarize_pos_tagged

Constants.enableUnrollMode()

activation = 'lstm_models'
modelName = 'model1'
base_path = os.path.dirname(os.path.realpath(__file__))
out = open(os.path.join(base_path, "result_" + activation + '_' + modelName + ".csv"), "w")
base_path = os.path.dirname(os.path.dirname(os.path.dirname(base_path)))
module_path = os.path.join(base_path, activation, 'many_to_many_same', 'modules', 'model4')
scratch_model_path = os.path.join(base_path, activation, 'many_to_many_same', 'h5', modelName + '_scratch.h5')
model_path = os.path.join(base_path, activation, 'many_to_many_same', 'h5', modelName + '.h5')

model = load_model(model_path)

print(base_path)
x_train, x_test, y_train, y_test, num_words, timestep, nb_classes = load_pos_tagged_dataset(hot_encode=False)

class1 = 0
class2 = 0
diff = 0.0
no_pairs = 0

modules = []
for m in range(nb_classes):
    modularLayers = initModularLayers(model.layers)
    repopulateModularWeights(modularLayers, module_path, m)
    modules.append(modularLayers)
    # if m == 2:
    #     break

out.write('Class 1,Class 2,Modularized Accuracy,Trained Model Accuracy\n')
for pair in itertools.combinations(range(nb_classes), 2):
    class1 = pair[0]
    class2 = pair[1]

    if class1 == 0 or class2 == 0:
        continue

    moduleClass1 = modules[class1]
    moduleClass2 = modules[class2]

    xT, yT, xt, yt = binarize_pos_tagged(x_train, y_train, x_test, y_test, class1, class2)
    # print(xT.shape)
    # print(xt.shape)
    # predClass1Masked, predClass1Unmasked = getModulePredictionAnyToManyUnrolled(moduleClass1, xt, yt, timestep)
    # predClass2Masked, predClass2Unmasked = getModulePredictionAnyToManyUnrolled(moduleClass2, xt, yt, timestep)
    # predClass0Masked, predClass0Unmasked = getModulePredictionAnyToManyUnrolled(modules[0], xt, yt, timestep)
    preds = []
    for m in range(nb_classes):
        preds.append(getModulePredictionAnyToMany(modules[m], xt, yt, moduleNo=m))

    finalPred = []
    for i in range(0, len(yt)):
        maxPrediction = []
        for ts in range(timestep):
            temp_pred = []
            for m in range(nb_classes):
                temp_pred.append(preds[m][i][ts][m])

            temp_pred = temp_pred.index(max(temp_pred))
            if temp_pred != class1 and temp_pred != class2:
                temp_pred = 0
            maxPrediction.append(temp_pred)

            # a1 = predClass1Masked[i][ts][class1]
            # a2 = predClass1Masked[i][ts][0]
            # if class1 == 1:
            #     a3 = predClass1Masked[i][ts][2]
            # else:
            #     a3 = predClass1Masked[i][ts][1]
            # b1 = predClass2Masked[i][ts][class2]
            # b2 = predClass2Masked[i][ts][0]
            # if class2 == 1:
            #     b3 = predClass2Masked[i][ts][2]
            # else:
            #     b3 = predClass2Masked[i][ts][1]
            # c1 = predClass0Masked[i][ts][0]
            # c2 = predClass0Masked[i][ts][1]
            # ls = [('mod1-pos', a1), ('mod2-pos', b1),
            #       ('mod0-pos', c1)]
            # # ls = [('mod1-pos', a1), ('mod1-0', a2), ('mod1-neg', a3), ('mod2-pos', b1), ('mod2-0', b2),
            # #       ('mod2-neg', b3)]
            # ls = sorted(ls, key=lambda x: x[1], reverse=True)
            #
            # if ls[0][0] == 'mod1-pos':
            #     temp_pred.append(class1)
            # elif ls[0][0] == 'mod2-pos':
            #     temp_pred.append(class2)
            # else:
            #     temp_pred.append(0)

            # if predClass1Unmasked[i][ts] == class1 and predClass2Unmasked[i][ts] != class2:
            #     temp_pred.append(class1)
            # elif predClass1Unmasked[i][ts] != class1 and predClass2Unmasked[i][ts] == class2:
            #     temp_pred.append(class2)
            # elif predClass1Unmasked[i][ts] == class1 and predClass2Unmasked[i][ts] == class2:
            #     if predClass1Masked[i][ts][class1] > predClass2Masked[i][ts][class2]:
            #         temp_pred.append(class1)
            #     else:
            #         temp_pred.append(class2)
            # else:
            #     temp_pred.append(0)

        finalPred.append(np.asarray(maxPrediction))

    finalPred = np.asarray(finalPred)
    finalPred = finalPred.flatten()
    scoreModular = accuracy_score(finalPred, np.asarray(yt).flatten())
    print("Modularized Accuracy (Class " + str(class1) + " - Class " + str(class2) + "): " + str(scoreModular))

    yT = to_categorical(yT)
    modelAccuracy = trainModelAndPredictInBinaryForManyOutput(scratch_model_path, xT, yT, xt, yt,
                                                              nb_classes=yT.shape[2], oneToMany=False)
    print("Trained Model Accuracy (Class " + str(class1) + " - Class " + str(class2) + "): " + str(modelAccuracy))

    diff += (modelAccuracy - scoreModular)
    no_pairs += 1

    out.write(str(class1) + ',' + str(class2) + ',' + str(scoreModular) + ',' + str(modelAccuracy) + '\n')

out.close()

diff = diff / no_pairs
print('Average loss of accuracy: ' + str(diff))
