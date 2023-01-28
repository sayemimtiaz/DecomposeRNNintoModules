from datetime import datetime
from time import sleep

import numpy as np
from data_type.constants import Constants
from gru_models.many_to_one.evaluator_unrolled import evaluate_unrolled

from modularization.channeling import channel
from modularization.concern.concern_identification import *

from keras.models import load_model

from util.nltk_util import loadClincOos, loadClincOosWithPosTag
from util.sampling_util import sample_for_one_output

Constants.enableUnrollMode()

root = os.path.dirname(os.path.realpath(__file__))
model_name = os.path.join(root, 'h5', 'model1.h5')
firstModel = load_model(model_name)
concernIdentifier = ConcernIdentification()
module_path = os.path.join(root, 'modules', extract_model_name(model_name))

if 'combined' not in model_name:
    x_train, x_test, y_train, y_test, num_words, timestep, nb_classes = loadClincOos(hot_encode=False, timestep=15)
else:
    x_train, x_test, y_train, y_test, num_words, timestep, nb_classes = loadClincOosWithPosTag(hot_encode=False,
                                                                                               timestep=15)

labs = range(0, nb_classes)
print("Start Time:" + datetime.now().strftime("%H:%M:%S"))

numPosSample = 500
for j in labs:

    model = load_model(model_name)
    positiveConcern = initModularLayers(model.layers)
    negativeConcern = initModularLayers(model.layers)

    print("#Module " + str(j) + " in progress....")
    sx, sy = sample_for_one_output(x_train, y_train, j, numPosSample)

    hidden_values_pos = {}
    for x in sx:
        x_t = x

        for layerNo, _layer in enumerate(positiveConcern):
            if _layer.type == LayerType.GRU:
                _layer.initHiddenState()

        for layerNo, _layer in enumerate(positiveConcern):
            x_t = concernIdentifier.propagateThroughLayer(_layer, x_t, apply_activation=True)

            if _layer.type == LayerType.GRU:
                if layerNo not in hidden_values_pos:
                    hidden_values_pos[layerNo] = []
                if not ALL_GATE:
                    hidden_values_pos[layerNo].append(_layer.copyHiddenState())
                else:
                    hidden_values_pos[layerNo].append(_layer.copyLSTMNodeVal())

    sx, sy = sample_for_one_output(x_train, y_train, j, numPosSample / (nb_classes - 1), positiveSample=False)

    hidden_values_neg = {}
    for x in sx:
        x_t = x

        for layerNo, _layer in enumerate(negativeConcern):
            if _layer.type == LayerType.GRU:
                _layer.initHiddenState()

        for layerNo, _layer in enumerate(negativeConcern):
            x_t = concernIdentifier.propagateThroughLayer(_layer, x_t, apply_activation=True)

            if _layer.type == LayerType.GRU:
                if layerNo not in hidden_values_neg:
                    hidden_values_neg[layerNo] = []
                if not ALL_GATE:
                    hidden_values_neg[layerNo].append(_layer.copyHiddenState())
                else:
                    hidden_values_neg[layerNo].append(_layer.copyLSTMNodeVal())

    for layerNo, _layer in enumerate(positiveConcern):
        if _layer.type == LayerType.GRU:
            calculate_50th_percentile_of_nodes_unrolled(hidden_values_pos[layerNo], _layer,
                                                        normalize=False)
            calculate_50th_percentile_of_nodes_unrolled(hidden_values_neg[layerNo],
                                                        negativeConcern[layerNo],
                                                        normalize=False)

    for layerNo, _layer in enumerate(positiveConcern):
        if shouldRemove(_layer):
            if _layer.type == LayerType.GRU:
                for ts in range(_layer.timestep):
                    _layer.DW[ts], _layer.DU[ts], _layer.DB[ts] = firstModel.layers[layerNo].get_weights()
                    removeAndTangleConcernBasedOnComparison(positiveConcern[layerNo],
                                                            negativeConcern[layerNo], timestep=ts,
                                                            tangleThreshold=0.0, maxRemove=0.05)

    for layerNo, _layer in enumerate(positiveConcern):
        if _layer.type == LayerType.RepeatVector or _layer.type == LayerType.Flatten or _layer.type == LayerType.Dropout:
            continue

        if _layer.type == LayerType.GRU:
            for ts in range(_layer.timestep):
                model.layers[layerNo].set_weights([_layer.DW[ts], _layer.DU[ts], _layer.DB[ts]])
                getDeadNodePercent(_layer, ts)
                model.save(os.path.join(module_path,
                                        'module' + str(j) + '_layer' +
                                        str(layerNo) + '_timestep' + str(ts) + '.h5'))
        elif _layer.type == LayerType.Embedding:
            model.layers[layerNo].set_weights([_layer.W])
        else:
            channel(_layer, labs, positiveIntent=j)
            model.layers[layerNo].set_weights([_layer.DW, _layer.DB])
            getDeadNodePercent(_layer)

    model.save(os.path.join(module_path,'module' + str(j) + '.h5'))

evaluate_unrolled(model_name)
