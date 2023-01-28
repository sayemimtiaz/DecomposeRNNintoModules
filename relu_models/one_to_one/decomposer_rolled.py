from datetime import datetime
from time import sleep

import numpy as np
from data_type.constants import Constants
from relu_models.one_to_one.evaluator_rolled import evaluate_rolled
from relu_models.one_to_one.one_to_one_util import load_math_dataset, load_math_dataset_with_toxic

from modularization.channeling import channel
from modularization.concern.concern_identification import *

from keras.models import load_model

from util.sampling_util import sample_for_one_output

# initially set unrolled mode , then disable
Constants.disableUnrollMode()

root = os.path.dirname(os.path.realpath(__file__))
model_name = os.path.join(root, 'h5', 'model1.h5')
module_path = os.path.join(root, 'modules', extract_model_name(model_name))

firstModel = load_model(model_name)
concernIdentifier = ConcernIdentification()

if 'combined' not in model_name:
    x_train, x_test, y_train, y_test, num_words, timestep, nb_classes = load_math_dataset(hot_encode=False)
else:
    x_train, x_test, y_train, y_test, num_words, timestep, nb_classes = \
        load_math_dataset_with_toxic(hot_encode=False)

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
            calculate_active_rate_rolled(hidden_values_pos[layerNo], _layer)
            calculate_active_rate_rolled(hidden_values_neg[layerNo], negativeConcern[layerNo])

    maxRemove = 0.05

    for layerNo, _layer in enumerate(positiveConcern):
        if shouldRemove(_layer):
            if _layer.type == LayerType.RNN:
                removeAndTangleConcernBasedActiveStat(positiveConcern[layerNo],
                                                      negativeConcern[layerNo], maxRemove=maxRemove,
                                                      tangleThreshold=-0.5)

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

    model.save(os.path.join(module_path,'module' + str(j) + '.h5'))

Constants.disableUnrollMode()
evaluate_rolled(model_name)
