from datetime import datetime
from time import sleep

import numpy as np
from data_type.constants import Constants
from vanilla_rnn_models.many_to_one.evaluator_rolled import evaluate_rolled

from modularization.channeling import channel
from modularization.concern.concern_identification import *

from keras.models import load_model

from util.nltk_util import loadClincOos, loadClincOosWithPosTag
from util.sampling_util import sample_for_one_output

# initially set unrolled mode , then disable
Constants.disableUnrollMode()

root = os.path.dirname(os.path.realpath(__file__))
model_name = os.path.join(root, 'h5', 'model1.h5')
module_path = os.path.join(root, 'modules')

firstModel = load_model(model_name)
concernIdentifier = ConcernIdentification()

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
            if _layer.type == LayerType.RNN:
                _layer.initHiddenState()

        for layerNo, _layer in enumerate(positiveConcern):
            x_t = concernIdentifier.propagateThroughLayer(_layer, x_t, apply_activation=True)

            if _layer.type == LayerType.RNN:
                if layerNo not in hidden_values_pos:
                    hidden_values_pos[layerNo] = []
                if ALL_GATE:
                    hidden_values_pos[layerNo].append(_layer.lstm_node_val[_layer.timestep - 1])
                else:
                    if _layer.unrolled or _layer.return_sequence:
                        hidden_values_pos[layerNo].append(_layer.hidden_state[_layer.timestep - 1])
                    else:
                        hidden_values_pos[layerNo].append(_layer.full_hidden_state[_layer.timestep - 1])

    sx, sy = sample_for_one_output(x_train, y_train, j, numPosSample / (nb_classes - 1), positiveSample=False)

    hidden_values_neg = {}
    for x in sx:
        x_t = x

        for layerNo, _layer in enumerate(negativeConcern):
            if _layer.type == LayerType.RNN:
                _layer.initHiddenState()

        for layerNo, _layer in enumerate(negativeConcern):
            x_t = concernIdentifier.propagateThroughLayer(_layer, x_t, apply_activation=True)

            if _layer.type == LayerType.RNN:
                if layerNo not in hidden_values_neg:
                    hidden_values_neg[layerNo] = []
                if ALL_GATE:
                    hidden_values_neg[layerNo].append(_layer.lstm_node_val[_layer.timestep - 1])
                else:
                    if _layer.unrolled or _layer.return_sequence:
                        hidden_values_neg[layerNo].append(_layer.hidden_state[_layer.timestep - 1])
                    else:
                        hidden_values_neg[layerNo].append(_layer.full_hidden_state[_layer.timestep - 1])

    for layerNo, _layer in enumerate(positiveConcern):
        if _layer.type == LayerType.RNN:
            calculate_50th_percentile_of_nodes_rolled(hidden_values_pos[layerNo], _layer,
                                                      overrideReturnSequence=True, normalize=False)
            calculate_50th_percentile_of_nodes_rolled(hidden_values_neg[layerNo],
                                                      negativeConcern[layerNo],
                                                      overrideReturnSequence=True, normalize=False)

    maxRemove = 0.05
    for layerNo, _layer in enumerate(positiveConcern):
        if shouldRemove(_layer):
            if _layer.type == LayerType.RNN:
                _layer.DW = _layer.W
                _layer.DU = _layer.U
                _layer.DB = _layer.B
                removeAndTangleConcernBasedOnComparison(positiveConcern[layerNo],
                                                        negativeConcern[layerNo],
                                                        maxRemove=maxRemove)

    for layerNo, _layer in enumerate(positiveConcern):
        if _layer.type == LayerType.RepeatVector or _layer.type == LayerType.Flatten or _layer.type == LayerType.Dropout:
            continue

        if _layer.type == LayerType.RNN:
            model.layers[layerNo].set_weights([_layer.DW, _layer.DU, _layer.DB])
            getDeadNodePercent(_layer)

        elif _layer.type == LayerType.Embedding:
            model.layers[layerNo].set_weights([_layer.W])
        else:
            channel(_layer, labs, positiveIntent=j)
            model.layers[layerNo].set_weights([_layer.DW, _layer.DB])
            getDeadNodePercent(_layer)

    model.save('modules/module' + str(j) + '.h5')

Constants.disableUnrollMode()
evaluate_rolled(model_name)
