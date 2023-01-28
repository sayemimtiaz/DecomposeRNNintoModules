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
from evaluation.accuracy_computer import getModulePredictionAnyToManyUnrolled, getMonolithicModelPredictionAnyToOne
from relu_models.one_to_many.one_to_many_util import load_toxic_dataset, binarize_toxic_dataset, unarize_toxic_dataset, \
    toxic_dataset_text, load_toxic_dataset_combine_math_qa
from relu_models.one_to_one.one_to_one_util import load_math_dataset, unarize_math_qa, math_dataset_text, \
    load_math_dataset_with_toxic
from util.common import initModularLayers, repopulateModularWeights, binarize_multi_label, \
    trainModelAndPredictInBinaryForManyOutput, trainModelAndPredictInBinary, get_max_without

Constants.enableUnrollMode()

activation = 'lstm_models'
modelName = 'model1_combined'
ChannelFlag = True

base_path = os.path.dirname(os.path.realpath(__file__))
out = open(os.path.join(base_path, "result_" + activation + '_' + modelName + ".csv"), "w")
base_path = os.path.dirname(os.path.dirname(os.path.dirname(base_path)))
module_path1 = os.path.join(base_path, activation, 'one_to_many', 'modules', modelName)
module_path2 = os.path.join(base_path, activation, 'one_to_one', 'modules', modelName)

scratch_model_path = os.path.join(base_path, activation, 'one_to_one', 'h5', modelName + '_scratch.h5')
model_path1 = os.path.join(base_path, activation, 'one_to_many', 'h5', modelName + '.h5')
model_path2 = os.path.join(base_path, activation, 'one_to_one', 'h5', modelName + '.h5')

model1 = load_model(model_path1)
model2 = load_model(model_path2)

print(base_path)
x_train1, x_test1, y_train1, y_test1, num_words1, \
timestep1, nb_classes1 = load_toxic_dataset_combine_math_qa(repeat=False, hot_encode=False)

x_train2, x_test2, y_train2, y_test2, num_words2, \
timestep2, nb_classes2 = load_math_dataset_with_toxic(hot_encode=False)

class1 = 0
class2 = 0
diff = 0.0
no_pairs = 0

modules1 = []
for m in range(nb_classes1):
    modularLayers = initModularLayers(model1.layers)
    repopulateModularWeights(modularLayers, module_path1, m)
    modules1.append(modularLayers)
    # if m == 2:
    #     break

modules2 = []
for m in range(nb_classes2):
    modules2.append(load_model(os.path.join(module_path2,
                                            'module' + str(m) + '.h5')))

out.write('Class 1,Class 2,Modularized Accuracy,Trained Model Accuracy\n')

nb_classes = min(nb_classes1, nb_classes2)

for class1 in range(nb_classes1):

    if class1 == 0:
        continue

    for class2 in range(nb_classes2):

        # class1=2
        # class2=1
        moduleClass1 = modules1[class1]
        moduleClass2 = modules2[class2]

        xT1, yT1, xt1, yt1 = unarize_toxic_dataset(x_train1, y_train1,
                                                   x_test1, y_test1,
                                                   class1, 0)
        xT2, yT2, xt2, yt2 = unarize_math_qa(x_train2,
                                             y_train2, x_test2, y_test2,
                                             class2, 1)
        xT = np.concatenate((xT1, xT2))
        yT = np.concatenate((yT1, yT2))
        xt = np.concatenate((xt1, xt2))
        yt = np.concatenate((yt1, yt2))
        xT, yT = shuffle(xT, yT, random_state=0)
        xt, yt = shuffle(xt, yt, random_state=0)

        enn = RandomUnderSampler(random_state=0)

        xT, yT = enn.fit_resample(xT, yT)

        predClass1Masked, predClass1Unmasked = getModulePredictionAnyToManyUnrolled(moduleClass1, xt, yt, timestep1, moduleNo=class1)
        # predClass2 = moduleClass2.predict(xt)
        predClass2 = getMonolithicModelPredictionAnyToOne(moduleClass2, xt, yt)

        finalPred = []
        for i in range(0, len(yt)):

            temp_c1 = []
            for ts in range(timestep1):
                if ChannelFlag:
                    temp_c1.append(predClass1Masked[i][ts][class1] / predClass1Masked[i][ts][0])
                else:
                    temp_c1.append(predClass1Masked[i][ts][class1])

            temp_c1 = np.asarray(temp_c1)
            temp_c1 = temp_c1.max()

            if ChannelFlag:
                if class2 != 0:
                    temp_c2_no = predClass2[i][0]
                else:
                    temp_c2_no = predClass2[i][1]
            else:
                temp_c2_no = get_max_without(predClass2[i], class2)

            temp_c2 = predClass2[i][class2]
            temp_c2 = temp_c2 / temp_c2_no

            if temp_c1 >= temp_c2:
                finalPred.append(0)
            else:
                finalPred.append(1)

        finalPred = np.asarray(finalPred)
        finalPred = finalPred.flatten()
        scoreModular = accuracy_score(finalPred, np.asarray(yt).flatten())
        print("Modularized Accuracy (Class " + str(class1) + " - Class " + str(class2) + "): " + str(scoreModular))

        yT = to_categorical(yT)
        modelAccuracy = trainModelAndPredictInBinary(scratch_model_path,
                                                     xT, yT, xt, yt)
        print("Trained Model Accuracy (Class " + str(class1) + " - Class " + str(class2) + "): " + str(modelAccuracy))

        diff += (modelAccuracy - scoreModular)
        no_pairs += 1

        out.write(str(class1) + ',' + str(class2) + ',' + str(scoreModular) + ',' + str(modelAccuracy) + '\n')

out.close()

diff = diff / no_pairs
print('Average loss of accuracy: ' + str(diff))
