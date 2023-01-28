import os

from keras.models import load_model
from sklearn.metrics import accuracy_score

from data_type.enums import LayerType
from util.common import initModularLayers, repopulateModularWeights, extract_model_name
from util.layer_propagator import LayerPropagator
import numpy as np

from modularization.layer_propagator_modular import LayerPropagatorModular


def getMonolithicModelAccuracyAnyToMany(model_path, xt, yt, skipDummyLabel=True, dummy_label=0):
    if type(model_path) is str:
        model = load_model(model_path)
    else:
        model = model_path
    modelLayers = initModularLayers(model.layers)

    finalPred = []
    newYt = []
    length = len(yt)

    layerPropagator = LayerPropagator()
    for i in range(0, length):

        x_t = xt[i]
        for layerNo, _layer in enumerate(modelLayers):
            if _layer.type == LayerType.RNN or _layer.type == LayerType.LSTM or _layer.type == LayerType.GRU:
                _layer.initHiddenState()

        for layerNo, _layer in enumerate(modelLayers):
            x_t = layerPropagator.propagateThroughLayer(_layer, x_t, apply_activation=True)

        for ts in range(x_t.shape[0]):
            if skipDummyLabel:
                if x_t[ts].argmax() != dummy_label:
                    finalPred.append(x_t[ts].argmax())
                    newYt.append(yt[i][ts])
            else:
                finalPred.append(x_t[ts].argmax())
                newYt.append(yt[i][ts])

    finalPred = np.asarray(finalPred)
    finalPred = finalPred.flatten()
    if skipDummyLabel:
        yt = np.asarray(newYt)
    yt = yt.flatten()
    score = accuracy_score(finalPred, yt)

    return score


def getMonolithicModelAccuracyAnyToOne(model_path, xt, yt):
    model = load_model(model_path)
    modelLayers = initModularLayers(model.layers)

    finalPred = []
    length = len(yt)

    layerPropagator = LayerPropagator()
    for i in range(0, length):

        x_t = xt[i]
        for layerNo, _layer in enumerate(modelLayers):
            if _layer.type == LayerType.RNN or _layer.type == LayerType.LSTM or _layer.type == LayerType.GRU:
                _layer.initHiddenState()

        for layerNo, _layer in enumerate(modelLayers):
            x_t = layerPropagator.propagateThroughLayer(_layer, x_t, apply_activation=True)

        finalPred.append(x_t.argmax())

    finalPred = np.asarray(finalPred)
    finalPred = finalPred.flatten()
    yt = yt.flatten()
    score = accuracy_score(finalPred, yt)

    return score


def getMonolithicModelPredictionAnyToOne(model, xt, yt):
    modelLayers = initModularLayers(model.layers)

    finalPred = []
    length = len(yt)

    layerPropagator = LayerPropagator()
    for i in range(0, length):

        x_t = xt[i]
        for layerNo, _layer in enumerate(modelLayers):
            if _layer.type == LayerType.RNN or _layer.type == LayerType.LSTM or _layer.type == LayerType.GRU:
                _layer.initHiddenState()

        for layerNo, _layer in enumerate(modelLayers):
            x_t = layerPropagator.propagateThroughLayer(_layer, x_t, apply_activation=True)

        finalPred.append(x_t)
    finalPred = np.asarray(finalPred)
    return finalPred


def getModulePredictionAnyToMany(modelLayers, xt, yt, moduleNo=None):
    finalPred = []
    length = len(yt)

    layerPropagator = LayerPropagatorModular(moduleNo)

    for i in range(0, length):

        x_t = xt[i]
        for layerNo, _layer in enumerate(modelLayers):
            if _layer.type == LayerType.RNN or _layer.type == LayerType.LSTM or _layer.type == LayerType.GRU:
                _layer.initHiddenState()

        for layerNo, _layer in enumerate(modelLayers):
            x_t = layerPropagator.propagateThroughLayer(_layer, x_t, apply_activation=True)

        finalPred.append(x_t)

    return finalPred


def getModulePredictionAnyToOneUnrolled(module, xt, yt, moduleNo=None):
    finalPred = []
    length = len(yt)

    layerPropagator = LayerPropagatorModular(moduleNo)
    for i in range(0, length):
        x_t = xt[i]
        for layerNo, _layer in enumerate(module):
            if _layer.type == LayerType.RNN or _layer.type == LayerType.LSTM or _layer.type == LayerType.GRU:
                _layer.initHiddenState()

        for layerNo, _layer in enumerate(module):
            x_t = layerPropagator.propagateThroughLayer(_layer, x_t, apply_activation=True)

        finalPred.append(x_t)

    return finalPred


def getModuleAccuracyAnyToOneUnrolled(model_path, model_name, nb_classes, xt, yt):
    model = load_model(os.path.join(model_path, model_name))
    modulePath = os.path.join(model_path, 'modules', extract_model_name(model_name))
    labs = range(0, nb_classes)

    modules = []
    for m in labs:
        modularLayers = initModularLayers(model.layers)
        repopulateModularWeights(modularLayers, modulePath, m)
        modules.append(modularLayers)

    finalPred = []
    length = len(yt)

    for i in range(0, length):
        maxPrediction = []
        for m in labs:
            layerPropagator = LayerPropagatorModular(m)
            x_t = xt[i]
            for layerNo, _layer in enumerate(modules[m]):
                if _layer.type == LayerType.RNN or _layer.type == LayerType.LSTM or _layer.type == LayerType.GRU:
                    _layer.initHiddenState()

            for layerNo, _layer in enumerate(modules[m]):
                x_t = layerPropagator.propagateThroughLayer(_layer, x_t, apply_activation=True)

            maxPrediction.append(x_t[m])
        finalPred.append(maxPrediction.index(max(maxPrediction)))

    return accuracy_score(finalPred, yt[:length])


def getModulePredictionAnyToManyUnrolled(module, xt, yt, timestep, moduleNo=None):
    p = getModulePredictionAnyToMany(module, xt, yt, moduleNo)

    finalPred = []
    length = len(yt)

    for i in range(0, length):

        maxPrediction = []
        for ts in range(timestep):
            maxPrediction.append(p[i][ts].argmax())

        finalPred.append(maxPrediction)

    finalPred = np.asarray(finalPred)

    return p, finalPred


def getModulePredictionAnyToOneUnrolledWithMasked(module, xt, yt, moduleNo=None):
    p = getModulePredictionAnyToOneUnrolled(module, xt, yt, moduleNo)

    finalPred = []
    length = len(yt)

    for i in range(0, length):
        finalPred.append(p[i].argmax())

    finalPred = np.asarray(finalPred)

    return p, finalPred


def getModuleAccuracyAnyToManyUnrolled(module_path, model_name, nb_classes, xt, yt, timestep):
    model = load_model(model_name)
    modulePath = os.path.join(module_path, 'modules', extract_model_name(model_name))
    labs = range(0, nb_classes)

    p = []
    for m in labs:
        modularLayers = initModularLayers(model.layers)
        repopulateModularWeights(modularLayers, modulePath, m)
        p.append(getModulePredictionAnyToMany(modularLayers, xt, yt, moduleNo=m))

    finalPred = []
    length = len(yt)

    for i in range(0, length):

        maxPrediction = []
        for ts in range(timestep):
            temp_prediction = []
            for m in labs:
                temp_prediction.append(p[m][i][ts][m])

            maxPrediction.append(temp_prediction.index(max(temp_prediction)))

        finalPred.append(maxPrediction)

    finalPred = np.asarray(finalPred)
    finalPred = finalPred.flatten()
    yt = yt[:length]
    yt = yt.flatten()
    score = accuracy_score(finalPred, yt)

    return score
