import os

from keras.models import load_model
import numpy as np

from data_type.enums import LayerType
from modularization.concern.concern_identification_encoder_decoder import ConcernIdentificationEnDe
from util.common import initModularLayers, repopulateModularWeights, extract_model_name


def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


def getFlattenedWeightsRolled(model_path, only_decoder=False, timestep=None):
    model = load_model(model_path)
    modelLayers = initModularLayers(model.layers, timestep=timestep)
    S = []
    for layerNo, layer in enumerate(modelLayers):
        if layer.type == LayerType.RepeatVector or layer.type == LayerType.Flatten \
                or layer.type == LayerType.Input or layer.type == LayerType.Dropout \
                or layer.type == LayerType.Embedding:
            continue

        if only_decoder:
            if not ConcernIdentificationEnDe.is_decoder_layer(layer):
                continue

        W, U, B = np.array([]), np.array([]), np.array([])
        if layer.type == LayerType.RNN or layer.type == LayerType.LSTM or layer.type == LayerType.GRU:
            W, _, _ = model.layers[layerNo].get_weights()
        elif layer.type == LayerType.Embedding:
            _ = model.layers[layerNo].get_weights()[0]
        else:
            W, _ = model.layers[layerNo].get_weights()

        W = np.array(W.flatten())
        # U = np.array(U.flatten())
        # B = np.array(B.flatten())
        N = np.concatenate((W, U, B))
        S.append(N)

    # S = np.array(S)

    T = []
    for s in S:
        for t in s:
            T.append(t)

    return T


def getFlattenedWeightsUnrolledModule(originalFlattenedModelWeights, modelLayers, only_decoder=False):
    jaccard_index = []
    for layerNo, layer in enumerate(modelLayers):
        if layer.type == LayerType.RepeatVector or layer.type == LayerType.Flatten \
                or layer.type == LayerType.Input or layer.type == LayerType.Dropout \
                or layer.type == LayerType.Embedding:
            continue

        if only_decoder:
            if not ConcernIdentificationEnDe.is_decoder_layer(layer):
                continue

        if layer.type == LayerType.RNN or layer.type == LayerType.LSTM or layer.type == LayerType.GRU:

            for ts in range(layer.timestep):
                S = [layer.DW[ts].flatten()]

                for otherLayerNo in range(layerNo + 1, len(modelLayers)):
                    if modelLayers[otherLayerNo].type == LayerType.RepeatVector or modelLayers[otherLayerNo].type == LayerType.Flatten \
                            or modelLayers[otherLayerNo].type == LayerType.Input or modelLayers[otherLayerNo].type == LayerType.Dropout \
                            or modelLayers[otherLayerNo].type == LayerType.Embedding:
                        continue
                    if only_decoder:
                        if not ConcernIdentificationEnDe.is_decoder_layer(modelLayers[otherLayerNo]):
                            continue
                    if modelLayers[otherLayerNo].type == LayerType.RNN or modelLayers[otherLayerNo].type == LayerType.LSTM or layer.type == LayerType.GRU:
                        S.append(modelLayers[otherLayerNo].DW[ts].flatten())
                    else:
                        S.append(modelLayers[otherLayerNo].DW.flatten())

                T = []
                for s in S:
                    for t in s:
                        T.append(t)
                jaccard_index.append(jaccard_similarity(list(originalFlattenedModelWeights), list(T)))

            break

    return np.mean(jaccard_index)


def findMeanJaccardIndexRolled(base_model_path, target_module_path, nb_modules, only_decoder=False, timestep=None):
    flattenedModelWeights = getFlattenedWeightsRolled(base_model_path, only_decoder=only_decoder, timestep=timestep)

    labs = range(0, nb_modules)

    flattenedModuleWeights = []
    for m in labs:
        module_path = os.path.join(target_module_path, 'modules',extract_model_name(base_model_path),
                                   'module' + str(m) + '.h5')
        flattenedModuleWeights.append(
            getFlattenedWeightsRolled(module_path, only_decoder=only_decoder, timestep=timestep))

    jaccard_index = []
    for i in labs:
        temp = jaccard_similarity(list(flattenedModelWeights), list(flattenedModuleWeights[i]))
        jaccard_index.append(temp)
    JI = np.mean(jaccard_index)
    # print('mean jaccard index: ' + str(round(JI, 2)))
    return JI


def findMeanJaccardIndexUnrolled(base_model_path, target_module_path, nb_modules):
    flattenedModelWeights = getFlattenedWeightsRolled(base_model_path)
    model = load_model(base_model_path)

    labs = range(0, nb_modules)

    modulePath = os.path.join(target_module_path, 'modules', extract_model_name(base_model_path))
    jaccard_index = []

    for m in labs:
        modularLayers = initModularLayers(model.layers)
        repopulateModularWeights(modularLayers, modulePath, m)

        jaccard_index.append(getFlattenedWeightsUnrolledModule(flattenedModelWeights, modularLayers))

    return np.mean(jaccard_index)


def findMeanJaccardIndexUnrolledEnDe(model_path, modules,
                                     only_decoder=True, timestep=None):
    flattenedModelWeights = getFlattenedWeightsRolled(model_path, only_decoder=only_decoder,
                                                      timestep=timestep)
    jaccard_index = []

    for m in modules:

        jaccard_index.append(getFlattenedWeightsUnrolledModule(flattenedModelWeights, m,
                                                               only_decoder=only_decoder))

    return np.mean(jaccard_index)

# print(jaccard_similarity([1,2],[3,2]))
