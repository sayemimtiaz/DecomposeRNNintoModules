from datetime import datetime

from vanilla_rnn_models.many_to_many_same.evaluator_unrolled import evaluate_unrolled
from util.nltk_util import load_pos_tagged_dataset
from util.sampling_util import sample_for_many_output
from modularization.channeling import channel, channel_multi_output
from modularization.concern.concern_identification import *

from keras.models import load_model

from data_type.constants import Constants

# first disable unroll_rnn and the enable before evlaute

# Constants.disableUnrollMode()
root = os.path.dirname(os.path.realpath(__file__))
model_name = os.path.join(root, 'h5', 'model1.h5')
module_path = os.path.join(root, 'modules', extract_model_name(model_name))

firstModel = load_model(model_name)
concernIdentifier = ConcernIdentification()

x_train, x_test, y_train, y_test, num_words, timestep, nb_classes = load_pos_tagged_dataset(hot_encode=False)

labs = range(0, nb_classes)
print("Start Time:" + datetime.now().strftime("%H:%M:%S"))

hidden_values = {}

numPosSample = 500

for j in labs:
    print("#Module " + str(j) + " in progress....")

    model = load_model(model_name)

    concerns = []

    for ts in range(timestep):
        sx, sy = sample_for_many_output(x_train, y_train, ts, j, numPosSample)

        positiveConcern = initModularLayers(model.layers)

        hidden_values_pos = {}
        hidden_values_neg = {}

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
                    if not ALL_GATE:
                        hidden_values_pos[layerNo].append(_layer.getHiddenState(ts))
                    else:
                        hidden_values_pos[layerNo].append(_layer.lstm_node_val[ts])
        sx, sy = sample_for_many_output(x_train, y_train, ts, j, numPosSample / (nb_classes - 1), positiveSample=False)

        negativeConcern = initModularLayers(model.layers)

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
                    if not ALL_GATE:
                        hidden_values_neg[layerNo].append(_layer.getHiddenState(ts))
                    else:
                        hidden_values_neg[layerNo].append(_layer.lstm_node_val[ts])

        already_modularized = False
        for layerNo, _layer in enumerate(positiveConcern):
            if _layer.type == LayerType.RNN:
                if layerNo not in hidden_values_pos or layerNo not in hidden_values_neg:
                    already_modularized = True
                    continue
                calculate_50th_percentile_of_nodes_rolled(hidden_values_pos[layerNo], _layer,
                                                          overrideReturnSequence=True, normalize=False)
                calculate_50th_percentile_of_nodes_rolled(hidden_values_neg[layerNo],
                                                          negativeConcern[layerNo],
                                                          overrideReturnSequence=True, normalize=False)

        maxRemove = 0.05

        for layerNo, _layer in enumerate(positiveConcern):
            if shouldRemove(_layer):
                if _layer.type == LayerType.RNN:
                    if already_modularized:
                        _layer.DW, _layer.DU, _layer.DB = firstModel.layers[layerNo].get_weights()
                        continue
                    _layer.DW, _layer.DU, _layer.DB = firstModel.layers[layerNo].get_weights()
                    removeAndTangleConcernBasedOnComparison(positiveConcern[layerNo],
                                                            negativeConcern[layerNo], maxRemove=maxRemove)

        concerns.append(positiveConcern)

    for layerNo, _layer in enumerate(concerns[0]):
        if _layer.type == LayerType.RepeatVector or _layer.type == LayerType.Flatten or\
                _layer.type == LayerType.Dropout:
            continue

        if _layer.type == LayerType.RNN:
            for ts in range(_layer.timestep):
                model.layers[layerNo].set_weights([concerns[ts][layerNo].DW, concerns[ts][layerNo].DU,
                                                   concerns[ts][layerNo].DB])
                getDeadNodePercent(concerns[ts][layerNo])
                model.save(os.path.join(module_path,
                                        'module' + str(j) + '_layer' +
                                        str(layerNo) + '_timestep' + str(ts) + '.h5'))
        elif _layer.type == LayerType.Embedding:
            model.layers[layerNo].set_weights([_layer.W])
        else:
            channel(_layer, labs, positiveIntent=j)
            # channel_multi_output(_layer, labs, positiveIntent=j)
            model.layers[layerNo].set_weights([_layer.DW, _layer.DB])
            getDeadNodePercent(_layer)

    model.save('modules/module' + str(j) + '.h5')

Constants.enableUnrollMode()
evaluate_unrolled(model_name)
