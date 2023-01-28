from motivating_example.problem2.original.evaluator_rolled import evaulate_rolled
from data_type.enums import LayerType
from modularization.concern.concern_identification_encoder_decoder import ConcernIdentificationEnDe, \
    removeAndTangleConcernBasedOnComparison
from keras.models import load_model
from relu_models.many_to_many_different.many_to_many_different_rolled.many_to_many_different_util import  samplePositive, \
    sampleNegative
from data_type.constants import Constants
from motivating_example.problem2.problem2_data_util import load_problem2_original_data
from util.common import initModularLayers, shouldRemove, calculate_50th_percentile_of_nodes_rolled, getDeadNodePercent

Constants.disableUnrollMode()
model_name = 'h5/original_problem2.h5'

model = load_model(model_name)
concernIdentifier = ConcernIdentificationEnDe()

train_ds, val_ds, test_pairs, \
source_vectorization, target_vectorization, target_languages = load_problem2_original_data(sequence_length=20,
                                                                                           just_pairs=True)
labs = range(0, len(target_languages))

timestep = source_vectorization(train_ds[0]).numpy().shape[1]

hidden_values = {}

numPosSample = 1000

for j in labs:

    hidden_values_pos = {}
    hidden_values_neg = {}

    model = load_model(model_name)
    positiveConcern = initModularLayers(model.layers, timestep=timestep)
    negativeConcern = initModularLayers(model.layers, timestep=timestep)

    print("#Module " + target_languages[j] + " in progress....")

    positive_data = samplePositive(train_ds, target_languages[j],
                                   source_vectorization, target_vectorization, num_sample=numPosSample)
    print('Positive data loaded: ' + str(len(positive_data)))
    for source, target in positive_data:
        concernIdentifier.propagateThroughEncoderDecoder(positiveConcern, source, target)

        decoderLayersPositive = ConcernIdentificationEnDe.get_decoder_layers(positiveConcern)

        for layerNo, _layer in enumerate(decoderLayersPositive):
            if _layer.type == LayerType.LSTM:
                if layerNo not in hidden_values_pos:
                    hidden_values_pos[layerNo] = []
                for ts in range(timestep):
                    hidden_values_pos[layerNo].append(_layer.getHiddenState(ts))

    negativeLanguages = list(set(target_languages) - {target_languages[j]})
    negative_data = sampleNegative(train_ds, negativeLanguages,
                                   source_vectorization, target_vectorization, num_sample=numPosSample/2, asList=True)
    print('Negative data loaded: ' + str(len(negative_data)))
    for source, target in negative_data:
        concernIdentifier.propagateThroughEncoderDecoder(negativeConcern, source, target)

        decoderLayersNegative = ConcernIdentificationEnDe.get_decoder_layers(negativeConcern)

        for layerNo, _layer in enumerate(decoderLayersNegative):
            if _layer.type == LayerType.LSTM:
                if layerNo not in hidden_values_neg:
                    hidden_values_neg[layerNo] = []
                for ts in range(timestep):
                    hidden_values_neg[layerNo].append(_layer.getHiddenState(ts))

    decoderLayersPositive = ConcernIdentificationEnDe.get_decoder_layers(positiveConcern)
    decoderLayersNegative = ConcernIdentificationEnDe.get_decoder_layers(negativeConcern)

    already_modularized = False
    for layerNo, _layer in enumerate(decoderLayersPositive):
        if _layer.type == LayerType.LSTM:
            if layerNo not in hidden_values_pos or layerNo not in hidden_values_neg:
                already_modularized = True
                continue
            calculate_50th_percentile_of_nodes_rolled(hidden_values_pos[layerNo], decoderLayersPositive[layerNo],
                                                      overrideReturnSequence=True, normalize=False)
            calculate_50th_percentile_of_nodes_rolled(hidden_values_neg[layerNo],
                                                      decoderLayersNegative[layerNo],
                                                      overrideReturnSequence=True, normalize=False)

    maxRemove = 0.05

    for layerNo, _layer in enumerate(decoderLayersPositive):
        if shouldRemove(_layer):
            if _layer.type == LayerType.LSTM:
                if already_modularized:
                    _layer.DW, _layer.DU, _layer.DB = _layer.W, _layer.U, _layer.B
                    continue
                removeAndTangleConcernBasedOnComparison(decoderLayersPositive[layerNo],
                                                        decoderLayersNegative[layerNo], maxRemove=maxRemove)

    for layerNo, _layer in enumerate(positiveConcern):
        if _layer.type == LayerType.Input or _layer.type == LayerType.Dropout \
                or not ConcernIdentificationEnDe.is_decoder_layer(_layer):
            continue

        if _layer.type == LayerType.LSTM:
            model.layers[layerNo].set_weights([_layer.DW, _layer.DU, _layer.DB])
            getDeadNodePercent(_layer)

        elif _layer.type == LayerType.Dense and not _layer.last_layer:
            model.layers[layerNo].set_weights([_layer.DW, _layer.DB])
            getDeadNodePercent(_layer)

        elif _layer.type == LayerType.Embedding:
            model.layers[layerNo].set_weights([_layer.W])
        else:
            model.layers[layerNo].set_weights([_layer.DW, _layer.DB])
            getDeadNodePercent(_layer)

    model.save('modules/module' + str(j) + '.h5')

evaulate_rolled(model_name, 500)
