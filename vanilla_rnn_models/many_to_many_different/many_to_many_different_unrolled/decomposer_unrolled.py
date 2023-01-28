import os
from datetime import datetime

from data_type.enums import LayerType
from vanilla_rnn_models.many_to_many_different.many_to_many_different_unrolled.evaluator_unrolled import \
    evaluate_unrolled
from relu_models.many_to_many_different.many_to_many_different_rolled.many_to_many_different_util import load_tatoeba, samplePositive, \
    sampleNegative
from modularization.concern.concern_identification_encoder_decoder import ConcernIdentificationEnDe, \
    removeAndTangleConcernBasedOnComparison
from keras.models import load_model

from data_type.constants import Constants
from util.common import initModularLayers, calculate_50th_percentile_of_nodes_rolled, getDeadNodePercent, \
    extract_model_name

Constants.disableUnrollMode()
root = os.path.dirname(os.path.realpath(__file__))
model_name = os.path.join(root, 'h5', 'model1.h5')
firstModel = load_model(model_name)
concernIdentifier = ConcernIdentificationEnDe()

train_ds, val_ds, test_pairs, \
source_vectorization, target_vectorization, target_languages = load_tatoeba(sequence_length=20, just_pairs=True)
module_path = os.path.join(root, 'modules', extract_model_name(model_name))

labs = range(0, len(target_languages))
print("Start Time:" + datetime.now().strftime("%H:%M:%S"))
timestep = source_vectorization(train_ds[0]).numpy().shape[1]

hidden_values = {}

numPosSample = 1000

for j in labs:

    all_hidden_values_pos = {}
    all_hidden_values_neg = {}

    model = load_model(model_name)
    positiveConcern = initModularLayers(model.layers, timestep=timestep)
    negativeConcern = initModularLayers(model.layers, timestep=timestep)

    print("#Module " + target_languages[j] + " in progress....")

    positive_data = samplePositive(train_ds, target_languages[j],
                                   source_vectorization, target_vectorization, num_sample=numPosSample)
    print('Positive data loaded: ' + str(len(positive_data)))
    for source, target in positive_data:
        concernIdentifier.propagateThroughEncoderDecoder(positiveConcern, source, target)

        for layerNo, _layer in enumerate(positiveConcern):
            if _layer.type == LayerType.RNN and ConcernIdentificationEnDe.is_decoder_layer(_layer):
                if layerNo not in all_hidden_values_pos:
                    all_hidden_values_pos[layerNo] = []
                all_hidden_values_pos[layerNo].append(_layer.copyHiddenState())

    negativeLanguages = list(set(target_languages) - {target_languages[j]})
    negative_data = sampleNegative(train_ds, negativeLanguages,
                                   source_vectorization, target_vectorization, num_sample=numPosSample / 2, asList=True)
    print('Negative data loaded: ' + str(len(negative_data)))
    for source, target in negative_data:
        concernIdentifier.propagateThroughEncoderDecoder(negativeConcern, source, target)

        for layerNo, _layer in enumerate(negativeConcern):
            if _layer.type == LayerType.RNN and ConcernIdentificationEnDe.is_decoder_layer(_layer):
                if layerNo not in all_hidden_values_neg:
                    all_hidden_values_neg[layerNo] = []
                all_hidden_values_neg[layerNo].append(_layer.copyHiddenState())

    print("#Module " + str(j) + " in progress....")

    concerns = []

    for ts in range(timestep+1):

        hidden_values_pos = {}
        hidden_values_neg = {}

        positiveConcern = initModularLayers(model.layers, timestep=timestep)
        negativeConcern = initModularLayers(model.layers, timestep=timestep)

        for i in range(len(positive_data)):

            for layerNo, _layer in enumerate(positiveConcern):
                if _layer.type == LayerType.RNN and ConcernIdentificationEnDe.is_decoder_layer(_layer):
                    if layerNo not in hidden_values_pos:
                        hidden_values_pos[layerNo] = []
                    hidden_values_pos[layerNo].append(all_hidden_values_pos[layerNo][i][ts])

        for i in range(len(negative_data)):

            for layerNo, _layer in enumerate(negativeConcern):
                if _layer.type == LayerType.RNN and ConcernIdentificationEnDe.is_decoder_layer(_layer):
                    if layerNo not in hidden_values_neg:
                        hidden_values_neg[layerNo] = []
                    hidden_values_neg[layerNo].append(all_hidden_values_neg[layerNo][i][ts])

        already_modularized = False
        for layerNo, _layer in enumerate(positiveConcern):
            if _layer.type == LayerType.RNN and ConcernIdentificationEnDe.is_decoder_layer(_layer):
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
            if _layer.type == LayerType.RNN and ConcernIdentificationEnDe.is_decoder_layer(_layer):
                if already_modularized:
                    _layer.DW, _layer.DU, _layer.DB = firstModel.layers[layerNo].get_weights()
                    continue
                _layer.DW, _layer.DU, _layer.DB= firstModel.layers[layerNo].get_weights()
                removeAndTangleConcernBasedOnComparison(positiveConcern[layerNo],
                                                        negativeConcern[layerNo], maxRemove=maxRemove)

        concerns.append(positiveConcern)

    for layerNo, _layer in enumerate(concerns[0]):
        if _layer.type == LayerType.Input or _layer.type == LayerType.Dropout \
                or not ConcernIdentificationEnDe.is_decoder_layer(_layer):
            continue

        if _layer.type == LayerType.RNN:
            for ts in range(_layer.timestep):
                model.layers[layerNo].set_weights([concerns[ts][layerNo].DW, concerns[ts][layerNo].DU,
                                                   concerns[ts][layerNo].DB])
                getDeadNodePercent(concerns[ts][layerNo])
                model.save(os.path.join(module_path,
                                        'module' + str(j) + '_layer' +
                                        str(layerNo) + '_timestep' + str(ts) + '.h5'))
        elif _layer.type == LayerType.Dense and not _layer.last_layer:

            model.layers[layerNo].set_weights([_layer.DW, _layer.DB])

            getDeadNodePercent(_layer)

        elif _layer.type == LayerType.Embedding:
            model.layers[layerNo].set_weights([_layer.W])
        else:
            model.layers[layerNo].set_weights([_layer.DW, _layer.DB])
            getDeadNodePercent(_layer)

    model.save('modules/module' + str(j) + '.h5')

Constants.enableUnrollMode()
evaluate_unrolled(model_name, 500)
