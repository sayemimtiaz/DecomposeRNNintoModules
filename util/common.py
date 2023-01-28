import math
import os

import numpy as np
import scipy
from keras.layers import Dense, TimeDistributed, RepeatVector, LSTM, GRU, Dropout
from keras.models import load_model
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from data_type.constants import Constants, ALL_GATE
from data_type.enums import ActivationType, LayerType, getLayerType, getActivationType
from data_type.modular_layer_type import ModularLayer


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference
    # return scipy.special.softmax(x)


def sigmoid(x):
    return scipy.special.expit(x)


def tanh(x):
    return np.tanh(x)


def relu(x_t):
    x_t[x_t < 0] = 0
    return x_t


def initModularLayers(layers, timestep=None):
    myLayers = []
    first = True
    lastLayerTimestep = None
    for serial, _layer in enumerate(layers):
        l = ModularLayer(_layer, timestep=timestep)

        if l.type == LayerType.TimeDistributed:
            if lastLayerTimestep is None:
                l.initTimeDistributedWeights(timestep)
            else:
                l.initTimeDistributedWeights(lastLayerTimestep)
            l.setHiddenState()

        l.first_layer = first
        l.layer_serial = serial

        if l.timestep is not None:
            lastLayerTimestep = l.timestep

        if not first:
            myLayers[len(myLayers) - 1].next_layer = l

        if len(layers) == serial + 1:
            l.last_layer = True
            l.DW = l.W
            l.DB = l.B

        myLayers.append(l)
        first = False

    return myLayers


def isNodeActive(layer, nodeNum, threshold=None, timestep=None):
    hs = 0
    if timestep is not None:
        hs = layer.hidden_state[timestep][0, nodeNum]
    else:
        hs = layer.hidden_state[0, nodeNum]

    if layer.activation == ActivationType.Relu:
        return not (hs <= 0)
    if layer.activation == ActivationType.Sigmoid:
        return not (hs < 10)
    if layer.activation == ActivationType.Tanh:
        return not (-0.1 <= hs <= 0.1)


def getDeadNodePercent(layer, timestep=None, isPrint=False):
    W = layer.DW
    if timestep is not None:
        W = layer.DW[timestep]

    totalDeadPercdnt = 0.0
    if isPrint:
        print('Dead Node in ' + str(layer.type) + ':')
    if type(W) == list:
        for ts, w in enumerate(W):
            # print('Timestep: '+str(ts))
            alive = 0
            dead = 0
            for r in range(w.shape[0]):
                for c in range(w.shape[1]):
                    if w[r][c] == 0.0:
                        dead += 1
                    else:
                        alive += 1
            p = 0.0
            if alive + dead != 0:
                p = (dead / (alive + dead)) * 100.0
            totalDeadPercdnt += p
        avgDeadPercent = totalDeadPercdnt / (len(W) + 1)
        if isPrint:
            print('Average dead node: ' + str(avgDeadPercent) + '%')

    else:
        alive = 0
        dead = 0
        for r in range(W.shape[0]):
            for c in range(W.shape[1]):
                if W[r][c] == 0.0:
                    dead += 1
                else:
                    alive += 1

        p = 0.0
        if alive + dead != 0:
            p = (dead / (alive + dead)) * 100.0
        if isPrint:
            print('Dead node: ' + str(p) + '%')


def areArraysSame(a, b):
    for i in range(len(a)):
        if a[i].argmax() != b[i].argmax():
            print(a[i].argmax(), b[i].argmax())
            return False
    return True


def shouldRemove(_layer):
    if _layer.last_layer:
        return False
    if _layer.type == LayerType.Embedding \
            or _layer.type == LayerType.RepeatVector \
            or _layer.type == LayerType.Flatten or \
            _layer.type == LayerType.Dropout:
        return False
    return True


def isIntrinsicallyTrainableLayer(_layer):
    if _layer.type == LayerType.Embedding \
            or _layer.type == LayerType.RepeatVector \
            or _layer.type == LayerType.Flatten \
            or _layer.type == LayerType.Dropout \
            or _layer.type == LayerType.Input \
            or _layer.type == LayerType.Activation:
        return False
    return True


def repopulateModularWeights(modularLayers, module_dir, moduleNo, only_decoder=False):
    # print('module_dir>>', module_dir)
    from modularization.concern.concern_identification_encoder_decoder import ConcernIdentificationEnDe
    # module=module_dir
    module = load_model(
        os.path.join(module_dir, 'module' + str(moduleNo) + '.h5'))
    for layerNo, _layer in enumerate(modularLayers):
        if _layer.type == LayerType.RepeatVector \
                or _layer.type == LayerType.Flatten \
                or _layer.type == LayerType.Input \
                or _layer.type == LayerType.Dropout:
            continue
        if _layer.type == LayerType.RNN or _layer.type == LayerType.LSTM or _layer.type == LayerType.GRU:
            if only_decoder and not ConcernIdentificationEnDe.is_decoder_layer(_layer):
                modularLayers[layerNo].DW, modularLayers[layerNo].DU, \
                modularLayers[layerNo].DB = module.layers[layerNo].get_weights()

            elif Constants.UNROLL_RNN:
                for ts in range(_layer.timestep):
                    tempModel = load_model(
                        os.path.join(module_dir, 'module' + str(moduleNo) + '_layer' + str(layerNo) + '_timestep' + str(
                            ts) + '.h5'))
                    modularLayers[layerNo].DW[ts], modularLayers[layerNo].DU[ts], \
                    modularLayers[layerNo].DB[ts] = tempModel.layers[layerNo].get_weights()
            else:
                modularLayers[layerNo].DW, modularLayers[layerNo].DU, \
                modularLayers[layerNo].DB = module.layers[layerNo].get_weights()

        elif _layer.type == LayerType.Embedding:
            modularLayers[layerNo].DW = module.layers[layerNo].get_weights()[0]
        else:
            modularLayers[layerNo].DW, \
            modularLayers[layerNo].DB = module.layers[layerNo].get_weights()


def trainModelAndPredictInBinary(modelPath, X_train, Y_train, X_test, Y_test, epochs=5, batch_size=32, verbose=0
                                 , nb_classes=2, activation='softmax'):
    model = load_model(modelPath)
    model.pop()
    model.add(Dense(units=nb_classes, activation=activation, name='output'))

    model.fit(X_train, Y_train,
              epochs=epochs,
              batch_size=batch_size,
              verbose=verbose)
    # pred = model.predict(X_test[:len(Y_test)])
    from evaluation.accuracy_computer import getMonolithicModelPredictionAnyToOne
    pred = getMonolithicModelPredictionAnyToOne(model, X_test, Y_test)
    pred = pred.argmax(axis=-1)
    score = accuracy_score(pred, Y_test[:len(Y_test)])
    return score


def trainModelAndPredictInBinaryForManyOutput(modelPath, X_train, Y_train, X_test, Y_test, epochs=5, batch_size=32,
                                              verbose=0
                                              , nb_classes=2, activation='softmax', timestep=2, oneToMany=True):
    model = load_model(modelPath)

    if oneToMany:
        repeatFoundAt = 0
        for i in range(len(model.layers)):
            if getLayerType(model.layers[i]) == LayerType.RepeatVector:
                repeatFoundAt = i
                break

        popedLayers = []
        for i in range(repeatFoundAt, len(model.layers)):
            popedLayers.append(model.layers[i])

        remove = len(model.layers) - repeatFoundAt
        i = 0
        while i < remove:
            i += 1
            model.pop()

        model.add(RepeatVector(timestep))
        for i in range(1, len(popedLayers) - 1):
            if getLayerType(popedLayers[i]) == LayerType.Dropout:
                model.add(Dropout((popedLayers[i]).rate))
            elif getLayerType(popedLayers[i]) == LayerType.LSTM:
                model.add(LSTM(popedLayers[i].units, return_sequences=popedLayers[i].return_sequences,
                               activation=getActivationType(popedLayers[i]).name.lower()))
            elif getLayerType(popedLayers[i]) == LayerType.GRU:
                model.add(GRU(popedLayers[i].units, return_sequences=popedLayers[i].return_sequences, reset_after=False,
                              activation=getActivationType(popedLayers[i]).name.lower()))
            else:
                model.add(popedLayers[i])

    else:
        model.pop()

    model.add(TimeDistributed(Dense(units=nb_classes, activation=activation, name='output')))

    model.fit(X_train, Y_train,
              epochs=epochs,
              batch_size=batch_size,
              verbose=verbose)
    # pred = model.predict(X_test[:len(Y_test)])
    # pred = pred.argmax(axis=-1)
    # pred = pred.flatten()
    # Y_test = Y_test.flatten()
    # score = accuracy_score(pred, Y_test)
    from evaluation.accuracy_computer import getMonolithicModelAccuracyAnyToMany
    score = getMonolithicModelAccuracyAnyToMany(model, X_test, Y_test, skipDummyLabel=False)
    return score


def trainModelAndPredictOneToOne(modelPath, X_train, Y_train, X_test, Y_test, epochs=5, batch_size=32, verbose=0,
                                 class_weights=None):
    model = load_model(modelPath)

    model.fit(X_train, Y_train,
              epochs=epochs,
              batch_size=batch_size,
              verbose=verbose)
    # pred = model.predict(X_test[:len(Y_test)])
    from evaluation.accuracy_computer import getMonolithicModelPredictionAnyToOne
    pred = getMonolithicModelPredictionAnyToOne(model, X_test, Y_test)
    pred = pred.argmax(axis=-1)
    score = accuracy_score(pred, Y_test[:len(Y_test)])
    return score


def binarize_multi_label(y, class1, class2):
    class1Found = False
    class2Found = False
    for j in range(len(y)):
        if y[j] == class1:
            class1Found = True
        if y[j] == class2:
            class2Found = True

    if class1Found and class2Found:
        return [class1, class2]
    elif class1Found:
        return [0, class1]
    elif class2Found:
        return [0, class2]

    return []


def calculate_50th_percentile_of_nodes_rolled(observed_values, refLayer, normalize=True, overrideReturnSequence=False):
    num_node = refLayer.num_node

    if ALL_GATE:
        if refLayer.type == LayerType.LSTM:
            num_node = num_node * 4
        if refLayer.type == LayerType.GRU:
            num_node = num_node * 3

    for nodeNum in range(num_node):
        tl = []

        if ALL_GATE and ((refLayer.type == LayerType.LSTM and (
                nodeNum < 2 * refLayer.num_node or nodeNum >= 3 * refLayer.num_node)) or \
                         (refLayer.type == LayerType.GRU and nodeNum < 2 * refLayer.num_node)):
            inactive = 0
            active = 0
            for o in observed_values:
                if math.fabs(o[:, nodeNum]) < 0.5:
                    inactive += 1
                else:
                    active += 1
            refLayer.median_node_val[:, nodeNum] = active / (active + inactive)
        else:
            for o in observed_values:
                if not overrideReturnSequence and refLayer.return_sequence:
                    tl.append(math.fabs(o[refLayer.timestep - 1, :, nodeNum]))

                else:
                    tl.append(math.fabs(o[:, nodeNum]))
            tl = np.asarray(tl).flatten()
            # median = np.percentile(tl, 50)
            median = get_mean_minus_outliers(tl)
            refLayer.median_node_val[:, nodeNum] = median
    #
    # df_describe = pd.DataFrame(refLayer.median_node_val.flatten())
    # print(df_describe.describe())

    if normalize:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(refLayer.median_node_val.reshape(-1, 1))

        # df_describe = pd.DataFrame(scaled.flatten())
        # print(df_describe.describe())

        scaled = scaled.reshape(1, -1)
        refLayer.median_node_val = scaled


def get_mean_minus_outliers(data, m=2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.
    d = data[s < m]
    return np.mean(d)


def calculate_50th_percentile_of_nodes_unrolled(observed_values, refLayer, normalize=True):
    num_node = refLayer.num_node
    if ALL_GATE:
        if refLayer.type == LayerType.LSTM:
            num_node = num_node * 4
        if refLayer.type == LayerType.GRU:
            num_node = num_node * 3
    for nodeNum in range(num_node):
        for ts in range(refLayer.timestep):
            tl = []

            if ALL_GATE and ((refLayer.type == LayerType.LSTM and (
                    nodeNum < 2 * refLayer.num_node or nodeNum >= 3 * refLayer.num_node)) or \
                             (refLayer.type == LayerType.GRU and nodeNum < 2 * refLayer.num_node)):
                inactive = 0
                active = 0
                for o in observed_values:
                    if math.fabs(o[ts][:, nodeNum]) < 0.5:
                        inactive += 1
                    else:
                        active += 1
                refLayer.median_node_val[ts][:, nodeNum] = active / (active + inactive)
            else:
                for o in observed_values:
                    tl.append(math.fabs(o[ts][:, nodeNum]))

                tl = np.asarray(tl).flatten()
                # median = np.percentile(tl, 50)
                # median=np.mean(tl)
                median = get_mean_minus_outliers(tl)
                refLayer.median_node_val[ts][:, nodeNum] = median

    if normalize:
        for ts in range(refLayer.timestep):
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(refLayer.median_node_val[ts].reshape(-1, 1))

            scaled = scaled.reshape(1, -1)
            refLayer.median_node_val[ts] = scaled


def calculate_relative_importance(observed_values, refLayer, noRemoveThreshold=0.5, percentile=50,
                                  overrideReturnSequence=False):
    maxValue = 0.0
    num_node = refLayer.num_node
    if refLayer.type == LayerType.LSTM:
        num_node = num_node * 4
    for nodeNum in range(num_node):
        tl = []
        for o in observed_values:
            if not overrideReturnSequence and refLayer.return_sequence:
                tl.append(math.fabs(o[refLayer.timestep - 1][:, nodeNum]))

            else:
                tl.append(math.fabs(o[:, nodeNum]))
        tl = np.asarray(tl).flatten()
        median = np.percentile(tl, percentile)
        refLayer.median_node_val[:, nodeNum] = median
        maxValue = max(maxValue, median)

    for nodeNum in range(num_node):
        if refLayer.median_node_val[:, nodeNum] > noRemoveThreshold:
            refLayer.median_node_val[:, nodeNum] = 1.0
        else:
            refLayer.median_node_val[:, nodeNum] = refLayer.median_node_val[:, nodeNum] / maxValue


def calculate_relative_importance_unrolled(observed_values, refLayer, noRemoveThreshold=0.5, percentile=50):
    maxValue = {}
    num_node = refLayer.num_node
    if refLayer.type == LayerType.LSTM:
        num_node = num_node * 4
    for nodeNum in range(num_node):
        for ts in range(refLayer.timestep):
            if ts not in maxValue:
                maxValue[ts] = 0.0

            tl = []
            for o in observed_values:
                tl.append(math.fabs(o[ts][:, nodeNum]))

            tl = np.asarray(tl).flatten()
            median = np.percentile(tl, percentile)
            refLayer.median_node_val[ts][:, nodeNum] = median
            maxValue[ts] = max(maxValue[ts], median)

    for nodeNum in range(num_node):
        for ts in range(refLayer.timestep):
            if refLayer.median_node_val[ts][:, nodeNum] > noRemoveThreshold:
                refLayer.median_node_val[ts][:, nodeNum] = 1.0
            else:
                refLayer.median_node_val[ts][:, nodeNum] = refLayer.median_node_val[ts][:, nodeNum] / \
                                                           maxValue[ts]


def get_max_without(a, exceptIdx):
    b = np.append(a[0:exceptIdx], a[exceptIdx + 1:])
    return b.max()


def calculate_active_rate_rolled(observed_values, refLayer):
    num_node = refLayer.num_node

    for nodeNum in range(num_node):
        inactiveCount = 0
        activeCount = 0
        for o in observed_values:
            if o[:, nodeNum] <= 0.0:
                inactiveCount += 1
            else:
                activeCount += 1

        refLayer.median_node_val[:, nodeNum] = activeCount / (activeCount + inactiveCount)


def calculate_active_rate_unrolled(observed_values, refLayer):
    num_node = refLayer.num_node
    for nodeNum in range(num_node):
        for ts in range(refLayer.timestep):
            inactiveCount = 0
            activeCount = 0
            for o in observed_values:

                if o[ts][:, nodeNum] <= 0.0:
                    inactiveCount += 1
                else:
                    activeCount += 1

            refLayer.median_node_val[ts][:, nodeNum] = activeCount / (activeCount + inactiveCount)


def extract_model_name(model_path):
    if model_path.find('/')!=-1:
        model_path=model_path[model_path.rindex('/')+1:]
    return model_path[:-3]