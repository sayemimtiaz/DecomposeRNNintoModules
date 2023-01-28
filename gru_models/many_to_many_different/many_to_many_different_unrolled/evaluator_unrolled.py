#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 21:38:03 2020

@author:
"""

from keras.models import load_model
import os

from evaluation.calculate_bleu import get_bleu
from evaluation.jaccard_computer import findMeanJaccardIndexUnrolledEnDe
from relu_models.many_to_many_different.many_to_many_different_rolled.many_to_many_different_util import load_tatoeba, sampleNegative
from util.common import initModularLayers, repopulateModularWeights, extract_model_name


def evaluate_unrolled(model_name, num_sample=-1):
    # model_name = 'h5/model1_many_to_one.h5'

    train_ds, val_ds, test_pairs, \
    source_vectorization, target_vectorization, target_languages = load_tatoeba(sequence_length=20,
                                                                                just_pairs=True, seed=89)

    sampled_data = sampleNegative(test_pairs, target_languages,
                                  source_vectorization, target_vectorization, num_sample=num_sample, vectorize=False)

    labs = range(0, len(target_languages))

    timestep = source_vectorization(train_ds[0]).numpy().shape[1]

    model_path = os.path.dirname(os.path.realpath(__file__))
    model_name = os.path.join(model_path, model_name)
    model = load_model(model_name)
    modelLayers = initModularLayers(model.layers, timestep=timestep)
    module_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'modules',
                               extract_model_name(model_name))

    modules = []
    for m in labs:
        modularLayers = initModularLayers(model.layers, timestep=timestep)
        repopulateModularWeights(modularLayers, module_path, m, only_decoder=True)
        modules.append(modularLayers)

    i = 0
    for m, l in enumerate(target_languages):
        print('Bleu Score: (en-' + l + '): ')
        print('Monolithic Model: ')
        get_bleu(modelLayers, sampled_data[l], source_vectorization, target_vectorization, timestep)

        print('Modularized Model: ')
        get_bleu(modules[m], sampled_data[l], source_vectorization,
                 target_vectorization, timestep, isModule=True)

        i += num_sample
        print('\n')

    print('mean jaccard index: ' + str(findMeanJaccardIndexUnrolledEnDe(model_name, modules,
                                                                        only_decoder=True, timestep=timestep)))


if __name__ == '__main__':
    evaluate_unrolled()
