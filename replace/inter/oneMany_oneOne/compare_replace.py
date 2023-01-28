#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 21:38:03 2020

@author:
"""
from imblearn.combine import SMOTEENN
from keras.models import load_model
import itertools
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score
import os
import numpy as np
from sklearn.utils import shuffle, class_weight

from data_type.constants import Constants
from evaluation.accuracy_computer import getModulePredictionAnyToManyUnrolled, getMonolithicModelPredictionAnyToOne
from relu_models.one_to_many.one_to_many_util import load_toxic_dataset, binarize_toxic_dataset, unarize_toxic_dataset, \
    toxic_dataset_text, load_toxic_dataset_combine_math_qa
from relu_models.one_to_one.one_to_one_util import load_math_dataset, unarize_math_qa, math_dataset_text, \
    load_math_dataset_with_toxic, load_math_qa_for_replace
from util.common import initModularLayers, repopulateModularWeights, binarize_multi_label, \
    trainModelAndPredictInBinaryForManyOutput, trainModelAndPredictInBinary, trainModelAndPredictOneToOne
from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler, TomekLinks

activation='gru_models'
bestModelName = 'model4_combined'
faultyModelName = 'model4_combined'
base_path = os.path.dirname(os.path.realpath(__file__))
print(base_path)
out = open(os.path.join(base_path, "result_"+activation+"_"+bestModelName+".csv"), "w")
base_path = os.path.dirname(os.path.dirname(os.path.dirname(base_path)))
best_module_path = os.path.join(base_path, activation, 'one_to_many', 'modules', bestModelName)
faulty_module_path = os.path.join(base_path, activation, 'one_to_one', 'modules', faultyModelName)

scratch_model_path = os.path.join(base_path, activation, 'one_to_one', 'h5', faultyModelName + '_scratch.h5')
best_model_path = os.path.join(base_path, activation, 'one_to_many', 'h5', bestModelName + '.h5')
faulty_model_path = os.path.join(base_path, activation, 'one_to_one', 'h5', faultyModelName + '.h5')

bestModel = load_model(best_model_path)
faultyModel = load_model(faulty_model_path)

x_train_best, x_test_best, y_train_best, y_test_best, num_words_best, \
timestep_best, nb_classes_best = load_toxic_dataset_combine_math_qa(repeat=False, hot_encode=False)

x_train_faulty, x_test_faulty, y_train_faulty, y_test_faulty, num_words_faulty, \
timestep_faulty, nb_classes_faulty = load_math_dataset_with_toxic(hot_encode=False)

diff = 0.0
diff2 = 0.0
no_pairs = 0

Constants.enableUnrollMode()
modules_best = []
for m in range(nb_classes_best):
    modularLayers = initModularLayers(bestModel.layers)
    repopulateModularWeights(modularLayers, best_module_path, m)
    modules_best.append(modularLayers)

modules_faulty = []
for m in range(nb_classes_faulty):
    modules_faulty.append(load_model(os.path.join(faulty_module_path,
                                                  'module' + str(m) + '.h5')))

out.write('Replaced Class,Replaced With Class,Modularized Accuracy,Model Accuracy\n')

for faultyClass in range(nb_classes_faulty):

    bestLabel = faultyClass

    for bestClass in range(nb_classes_best):

        if bestClass == 0:
            continue

        bestModule = modules_best[bestClass]

        xT1, yT1, xt1, yt1 = unarize_toxic_dataset(x_train_best, y_train_best,
                                                   x_test_best, y_test_best,
                                                   bestClass, bestLabel)
        xT2, yT2, xt2, yt2 = load_math_qa_for_replace(x_train_faulty,
                                                      y_train_faulty, x_test_faulty, y_test_faulty,
                                                      faultyClass)
        xT = np.concatenate((xT1, xT2))
        yT = np.concatenate((yT1, yT2))
        xt = np.concatenate((xt1, xt2))
        yt = np.concatenate((yt1, yt2))
        xT, yT = shuffle(xT, yT, random_state=0)
        xt, yt = shuffle(xt, yt, random_state=0)

        # print('before yT:', yT.shape)
        enn = RandomUnderSampler(random_state=0)
        # enn = SMOTEENN(random_state=0)

        xT, yT = enn.fit_resample(xT, yT)
        # print('after yT:', yT.shape)
        # print('before yt:', yt.shape)
        # enn = RandomUnderSampler(random_state=0)
        # xt, yt = enn.fit_resample(xt, yt)
        # print('after yt:', yt.shape)

        # temp_weight = class_weight.compute_class_weight('balanced',
        #                                                   np.unique(yT),
        #                                                  yT)
        # class_weights={}
        # for _i,_w in enumerate(temp_weight):
        #     class_weights[_i]=_w
        # print(class_weights)
        predClassBestMasked, predClassBestUnmasked = getModulePredictionAnyToManyUnrolled(bestModule, xt,
                                                                                          yt, timestep_best, moduleNo=bestClass)

        faultyPred = {}
        for c in range(nb_classes_faulty):
            # faultyPred[c] = modules_faulty[c].predict(xt)
            faultyPred[c] = getMonolithicModelPredictionAnyToOne(modules_faulty[c], xt, yt)

        finalPred = []
        finalPred2 = []
        for i in range(0, len(yt)):

            temp_c1 = []
            for ts in range(timestep_best):
                # temp_c1.append(predClassBestMasked[i][ts][bestClass])
                temp_c1.append(predClassBestMasked[i][ts][bestClass] / predClassBestMasked[i][ts][0])

            temp_c1 = np.asarray(temp_c1)

            # temp_c1_no = predClassBestMasked[i][temp_c1.argmax()][0]
            temp_c1 = temp_c1.max()
            # temp_c1_approach_2 = temp_c1
            # temp_c1 = temp_c1 / temp_c1_no

            faultyMaxPr = 0.0
            faultyPrClass = 0
            faultyPrNeg = 0
            # faultyMaxPrApproach2 = 0.0
            # faultyPrClassApproach2 = 0
            for c in range(nb_classes_faulty):
                if c != faultyClass:

                    # if c != 0:
                    #     temp_c2_no = faultyPred[c][i][0]
                    # else:
                    #     temp_c2_no = faultyPred[c][i][1]
                    temp_c2 = faultyPred[c][i][c]
                    # temp_c2_approach_2 = temp_c2
                    # temp_c2 = temp_c2 / temp_c2_no

                    if temp_c2 > faultyMaxPr:
                        faultyMaxPr = temp_c2
                        faultyPrClass = c
                        if c != 0:
                            faultyPrNeg = faultyPred[c][i][0]
                        else:
                            faultyPrNeg = faultyPred[c][i][1]

                    # if temp_c2_approach_2 > faultyMaxPrApproach2:
                    #     faultyMaxPrApproach2 = temp_c2_approach_2
                    #     faultyPrClassApproach2 = c

            faultyMaxPr = faultyMaxPr / faultyPrNeg
            if temp_c1 >= faultyMaxPr:
                finalPred.append(bestLabel)
            else:
                finalPred.append(faultyPrClass)

            # if temp_c1_approach_2 > faultyMaxPrApproach2:
            #     finalPred2.append(bestLabel)
            # else:
            #     finalPred2.append(faultyPrClassApproach2)

        finalPred = np.asarray(finalPred)
        finalPred = finalPred.flatten()
        scoreModular = accuracy_score(finalPred, np.asarray(yt).flatten())
        print("After replacing " + str(faultyClass) + " Modular Accuracy: " + str(scoreModular) + ' (Approach 1)')

        # finalPred2 = np.asarray(finalPred2)
        # finalPred2 = finalPred2.flatten()
        # scoreModular2 = accuracy_score(finalPred2, np.asarray(yt).flatten())
        # print("After replacing " + str(faultyClass) + " Modular Accuracy: " + str(scoreModular2) + ' (Approach 2)')

        yT = to_categorical(yT)
        modelAccuracy = trainModelAndPredictOneToOne(scratch_model_path,
                                                     xT, yT, xt, yt)
        print("After replacing " + str(faultyClass) + " Trained Accuracy: " + str(modelAccuracy))

        diff += (modelAccuracy - scoreModular)
        no_pairs += 1

        # diff2 += (modelAccuracy - scoreModular2)

        out.write(
            str(faultyClass) + ',' + str(bestClass) + ',' + str(scoreModular)  + ',' + str(
                modelAccuracy) + '\n')

out.close()

diff = diff / no_pairs
print('Average loss of accuracy: ' + str(diff) + ' Aprorach 1')

# diff2 = diff2 / no_pairs
# print('Average loss of accuracy: ' + str(diff2) + ' Approach 2')
