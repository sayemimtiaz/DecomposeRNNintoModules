#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 21:38:03 2020

@author:
"""

from keras.models import load_model
import os
from evaluation.jaccard_computer import findMeanJaccardIndexRolled
from evaluation.calculate_bleu import get_bleu
from relu_models.many_to_many_different.many_to_many_different_rolled.many_to_many_different_util import sampleNegative
from motivating_example.problem1.problem1_data_util import load_problem1_solution2_data


def evaulate_rolled(model_name,num_sample=-1):

    train_ds, val_ds, test_pairs, \
    source_vectorization, target_vectorization, target_languages = load_problem1_solution2_data(sequence_length=20,
                                                                                just_pairs=True, seed=89)

    sampled_data = sampleNegative(test_pairs, target_languages,
                                  source_vectorization, target_vectorization, num_sample=num_sample, vectorize=False)

    timestep = source_vectorization(train_ds[0]).numpy().shape[1]

    model_path = os.path.dirname(os.path.realpath(__file__))
    model_name = os.path.join(os.path.dirname(model_path), model_name)
    model = load_model(model_name)

    i = 0
    for m, l in enumerate(target_languages):
        print('Bleu Score: (en-' + l + '): ')
        print('Monolithic Model: ')
        get_bleu(model, sampled_data[l], source_vectorization, target_vectorization, timestep)

        print('Modularized Model: ')
        module = load_model(os.path.join(model_path, 'modules/module' + str(m) + '.h5'))
        get_bleu(module, sampled_data[l], source_vectorization, target_vectorization, timestep)

        i += num_sample
        print('\n')

    print('mean jaccard index: ' + str(findMeanJaccardIndexRolled(model_name, model_path,
                                                                  len(target_languages),
                                                                  only_decoder=True, timestep=timestep)))


if __name__ == '__main__':
    evaulate_rolled()
