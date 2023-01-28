from nltk.translate.bleu_score import corpus_bleu
import numpy as np
import warnings

from data_type.enums import LayerType
from modularization.layer_propagator_modular_en_de import LayerPropagatorModularEnDe
from util.layer_propagator_en_de import LayerPropagatorEnDe

warnings.filterwarnings("ignore")


def decode_sequence(model, input_sentence, decoded_sentence, end_tag, source_vectorization, target_vectorization,
                    max_decoded_sentence_length, isModule=False):
    tokenized_input_sentence = source_vectorization([input_sentence])
    spa_vocab = target_vectorization.get_vocabulary()
    spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))

    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorization([decoded_sentence])

        if not isModule:
            # next_token_predictions = model.predict(
            #     [tokenized_input_sentence, tokenized_target_sentence])
            # sampled_token_index = np.argmax(next_token_predictions[0, i, :])

            layerPropagator = LayerPropagatorEnDe()
            source = tokenized_input_sentence.numpy().flatten()
            target = tokenized_target_sentence.numpy().flatten()

            for layerNo, _layer in enumerate(model):
                if _layer.type == LayerType.RNN or _layer.type == LayerType.LSTM or _layer.type == LayerType.GRU:
                    _layer.initHiddenState()

            next_token_predictions = layerPropagator.propagateThroughEncoderDecoder(model, source,
                                                                                    target,
                                                                                    apply_activation=True)
            sampled_token_index = np.argmax(next_token_predictions[i, :])

        else:
            layerPropagator = LayerPropagatorModularEnDe(None)
            source = tokenized_input_sentence.numpy().flatten()
            target = tokenized_target_sentence.numpy().flatten()

            for layerNo, _layer in enumerate(model):
                if _layer.type == LayerType.RNN or _layer.type == LayerType.LSTM or _layer.type == LayerType.GRU:
                    _layer.initHiddenState()

            next_token_predictions = layerPropagator.propagateThroughEncoderDecoder(model, source,
                                                                                    target,
                                                                                    apply_activation=True)

            sampled_token_index = np.argmax(next_token_predictions[i, :])

        sampled_token = spa_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == end_tag:
            break
    return decoded_sentence


def get_bleu(model, test_pairs, source_vectorization, target_vectorization,
             max_decoded_sentence_length, isModule=False):
    bleu_dic = {}
    test_eng_texts = [pair[0] for pair in test_pairs]
    test_fr_texts = [pair[1] for pair in test_pairs]
    actual, predicted = [], []
    for i in range(len(test_pairs)):

        if (i+1) % 1000 == 0:
            print('Tested ' + str(i) + ' data')

        acts = test_fr_texts[i].split()
        actual.append([acts])

        # if isModule:
        #     for layerNo, _layer in enumerate(model):
        #         if _layer.type == LayerType.RNN:
        #             _layer.initHiddenState()

        predicted.append(decode_sequence(model, test_eng_texts[i], acts[0], acts[-1],
                                         source_vectorization, target_vectorization,
                                         max_decoded_sentence_length,
                                         isModule=isModule).split())

    bleu_dic['1-grams'] = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
    bleu_dic['1-2-grams'] = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
    bleu_dic['1-3-grams'] = corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0))
    bleu_dic['1-4-grams'] = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))
    print(bleu_dic)
    return bleu_dic
