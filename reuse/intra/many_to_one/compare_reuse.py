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

from sklearn.utils import shuffle

from data_type.constants import Constants
from evaluation.accuracy_computer import getModulePredictionAnyToOneUnrolled
from util.common import trainModelAndPredictInBinary, initModularLayers, repopulateModularWeights
from util.nltk_util import binarize_clinc_oos, loadClincOos

Constants.enableUnrollMode()
activation = 'lstm_models'
modelName = 'model1'
base_path = os.path.dirname(os.path.realpath(__file__))
out = open(os.path.join(base_path, "result_" + activation + '_' + modelName + ".csv"), "w")
base_path = os.path.dirname(os.path.dirname(os.path.dirname(base_path)))
module_path = os.path.join(base_path, activation, 'many_to_one', 'modules','model1')
model_path = os.path.join(base_path, activation, 'many_to_one', 'h5', modelName + '.h5')

model = load_model(model_path)

print(base_path)
x_train, x_test, y_train, y_test, num_words, timestep, nb_classes = loadClincOos()

class1 = 0
class2 = 0
diff = 0.0
no_pairs = 0

modules = {}
for m in range(nb_classes):
    modularLayers = initModularLayers(model.layers)
    repopulateModularWeights(modularLayers, module_path, m)
    modules[m]=modularLayers
    # modules[m] = load_model(os.path.join(module_path, 'module' + str(m) + '.h5'))

out.write('Class 1,Class 2,Modularized Accuracy,Trained Model Accuracy\n')
for pair in itertools.combinations(range(nb_classes), 2):
    class1 = pair[0]
    class2 = pair[1]

    moduleClass1 = modules[class1]
    moduleClass2 = modules[class2]

    xT, yT, xt, yt = binarize_clinc_oos(class1, class2)
    xT, yT = shuffle(xT, yT, random_state=0)
    xt, yt = shuffle(xt, yt, random_state=0)

    enn = RandomUnderSampler(random_state=0)

    xT, yT = enn.fit_resample(xT, yT)

    # predClass1 = moduleClass1.predict(xt[:len(yt)])
    # predClass2 = moduleClass2.predict(xt[:len(yt)])
    predClass1 = getModulePredictionAnyToOneUnrolled(moduleClass1, xt, yt, moduleNo=class1)
    predClass2 = getModulePredictionAnyToOneUnrolled(moduleClass2, xt, yt, moduleNo=class2)

    finalPred = []
    for i in range(0, len(yt)):
        Fpred = predClass1[i]
        maxPredF = Fpred.argmax()
        if maxPredF != class1:
            maxPredF = nb_classes + 9

        # M1 prediction
        Mpred = predClass2[i]
        maxPredM = Mpred.argmax()
        if maxPredM != class2:
            maxPredM = nb_classes + 9

        pred = [maxPredF, maxPredM]

        if pred.count(nb_classes + 9) == 2 or pred.count(nb_classes + 9) == 0:
            # maxPrediction = [Mpred[class1], Fpred[class2]]
            maxPrediction = [Fpred[class1], Mpred[class2]]
            finalPred.append(maxPrediction.index(max(maxPrediction)))
        elif pred.count(nb_classes + 9) == 1:
            if pred[0] == class1:
                finalPred.append(0)
            if pred[1] == class2:
                finalPred.append(1)

    scoreModular = accuracy_score(finalPred, yt)
    print("Modularized Accuracy (Class " + str(class1) + " - Class " + str(class2) + "): " + str(scoreModular))

    yT = to_categorical(yT)
    modelAccuracy = trainModelAndPredictInBinary(model_path, xT, yT, xt, yt)
    print("Trained Model Accuracy (Class " + str(class1) + " - Class " + str(class2) + "): " + str(modelAccuracy))

    diff += (modelAccuracy - scoreModular)
    no_pairs += 1

    out.write(str(class1) + ',' + str(class2) + ',' + str(scoreModular) + ',' + str(modelAccuracy) + '\n')

out.close()

diff = diff / no_pairs
print('Average loss of accuracy: ' + str(diff))
