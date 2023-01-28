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
from sklearn.utils import shuffle

from data_type.constants import Constants
from evaluation.accuracy_computer import getModulePredictionAnyToManyUnrolled, getModulePredictionAnyToMany
from relu_models.one_to_many.one_to_many_util import load_toxic_dataset, binarize_toxic_dataset, unarize_toxic_dataset
from util.common import initModularLayers, repopulateModularWeights, binarize_multi_label, \
    trainModelAndPredictInBinaryForManyOutput

Constants.enableUnrollMode()

activation = 'lstm_models'
modelName = 'model1'
ChannelFlag = True
base_path = os.path.dirname(os.path.realpath(__file__))
out = open(os.path.join(base_path, "result_" + activation + '_' + modelName + ".csv"), "w")
base_path = os.path.dirname(os.path.dirname(os.path.dirname(base_path)))
module_path = os.path.join(base_path, activation, 'one_to_many', 'modules', modelName)
scratch_model_path = os.path.join(base_path, activation, 'one_to_many', 'h5', modelName + '_scratch.h5')
model_path = os.path.join(base_path, activation, 'one_to_many', 'h5', modelName + '.h5')

model = load_model(model_path)

print(base_path)
x_train, x_test, y_train, y_test, num_words, timestep, nb_classes = \
    load_toxic_dataset(repeat=False, hot_encode=False)

class1 = 0
class2 = 0
diff = 0.0
no_pairs = 0

modules = {}
for m in range(nb_classes):
    # if m == 0:
    #     continue
    modularLayers = initModularLayers(model.layers)
    repopulateModularWeights(modularLayers, module_path, m)
    modules[m] = modularLayers
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

    xT, yT, xt, yt = binarize_toxic_dataset(x_train, y_train, x_test, y_test, class1, class2, test_sample=-1)
    print(xT.shape)
    print(xt.shape)

    # predClass1Masked, predClass1Unmasked = getModulePredictionAnyToManyUnrolled(moduleClass1, xt, yt, timestep, moduleNo=class1)
    # predClass2Masked, predClass2Unmasked = getModulePredictionAnyToManyUnrolled(moduleClass2, xt, yt, timestep,  moduleNo=class1)
    # predClass0Masked, predClass0Unmasked = getModulePredictionAnyToManyUnrolled(modules[0], xt, yt, timestep)
    preds = []
    for m in range(nb_classes):
        preds.append(getModulePredictionAnyToMany(modules[m], xt, yt, moduleNo=m))

    finalPred = []
    for i in range(0, len(yt)):
        merged_pred = []
        temp_c1 = []
        temp_c2 = []
        # must_pred = []
        for ts in range(timestep):
            temp_pred = []
            for m in range(nb_classes):
                temp_pred.append(preds[m][i][ts][m])

            temp_pred = temp_pred.index(max(temp_pred))
            if temp_pred != class1 and temp_pred != class2:
                temp_pred = 0
            merged_pred.append(temp_pred)
            temp_c1.append(preds[class1][i][ts][class1])
            temp_c2.append(preds[class2][i][ts][class2])
            # a1 = predClass1Masked[i][ts][class1] / n1
            # a2 = predClass1Masked[i][ts][0] / n1
            # if class1 == 1:
            #     a3 = predClass1Masked[i][ts][2] / n1
            # else:
            #     a3 = predClass1Masked[i][ts][1] / n1
            # b1 = predClass2Masked[i][ts][class2] / n2
            # b2 = predClass2Masked[i][ts][0] / n2
            # if class2 == 1:
            #     b3 = predClass2Masked[i][ts][2] / n2
            # else:
            #     b3 = predClass2Masked[i][ts][1] / n2
            #
            # temp_c1.append(a1/a2)
            # temp_c2.append(b1/b2)
            #
            # if ChannelFlag:
            #     # ls = [('mod1-pos', a1), ('mod1-0', a2), ('mod1-neg', a3), ('mod2-pos', b1), ('mod2-0', b2),
            #     #       ('mod2-neg', b3)]
            #     ls = [('mod1-pos', a1/a2), ('mod2-pos', b1/b2), ('mod0-pos', predClass0Masked[i][ts][0]/predClass0Masked[i][ts][1])]
            #     ls = sorted(ls, key=lambda x: x[1], reverse=True)
            #
            #     if ls[0][0] == 'mod1-pos':
            #         merged_pred.append(class1)
            #     elif ls[0][0] == 'mod2-pos':
            #         merged_pred.append(class2)
            #     else:
            #         merged_pred.append(0)
            # else:
            #     if predClass1Unmasked[i][ts] == class1 and predClass2Unmasked[i][ts] == class2:
            #         if predClass1Masked[i][ts][class1] >= predClass2Masked[i][ts][class2]:
            #             merged_pred.append(class1)
            #         else:
            #             merged_pred.append(class2)
            #     elif predClass1Unmasked[i][ts] == class1:
            #         merged_pred.append(class1)
            #     elif predClass2Unmasked[i][ts] == class2:
            #         merged_pred.append(class2)
            #     else:
            #         merged_pred.append(0)

        merged_pred = binarize_multi_label(merged_pred, class1, class2)
        if len(merged_pred) == 0:
            temp_c1 = np.asarray(temp_c1)
            temp_c2 = np.asarray(temp_c2)
            temp_c1 = temp_c1.max()
            temp_c2 = temp_c2.max()
            if temp_c1 > temp_c2:
                finalPred.append([0, class1])
            elif temp_c1 < temp_c2:
                finalPred.append([0, class2])
            else:
                finalPred.append([class1, class2])
        else:
            finalPred.append(merged_pred)

    finalPred = np.asarray(finalPred)
    finalPred = finalPred.flatten()
    scoreModular = accuracy_score(finalPred, np.asarray(yt).flatten())
    print("Modularized Accuracy (Class " + str(class1) + " - Class " + str(class2) + "): " + str(scoreModular))

    yT = to_categorical(yT)
    modelAccuracy = trainModelAndPredictInBinaryForManyOutput(scratch_model_path, xT, yT, xt, yt,
                                                              nb_classes=yT.shape[2])
    print("Trained Model Accuracy (Class " + str(class1) + " - Class " + str(class2) + "): " + str(modelAccuracy))

    diff += (modelAccuracy - scoreModular)
    no_pairs += 1

    out.write(str(class1) + ',' + str(class2) + ',' + str(scoreModular) + ',' + str(modelAccuracy) + '\n')

out.close()

diff = diff / no_pairs
print('Average loss of accuracy: ' + str(diff))
