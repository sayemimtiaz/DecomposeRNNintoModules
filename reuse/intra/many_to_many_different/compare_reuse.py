#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 21:38:03 2020

@author:
"""
from keras.models import load_model
import os

from evaluation.calculate_bleu import get_bleu
from relu_models.many_to_many_different.many_to_many_different_rolled.many_to_many_different_util import load_tatoeba, sampleNegative

model_names = ['h5/model4_de_fr.h5', 'h5/model4_de_it.h5', 'h5/model4_fr_it.h5']
base_path = os.path.dirname(os.path.realpath(__file__))

train_ds, val_ds, test_pairs, \
source_vectorization, target_vectorization, target_languages = load_tatoeba(sequence_length=20,
                                                                            just_pairs=True, seed=89)

sampled_data = sampleNegative(test_pairs, target_languages,
                              source_vectorization, target_vectorization, num_sample=500, vectorize=False)

timestep = source_vectorization(train_ds[0]).numpy().shape[1]

model_path = os.path.dirname(os.path.realpath(__file__))

out = open(os.path.join(base_path, "result_13_avg.csv"), "w")

out.write('Reuse Model,Target Language,Bleu-1,Bleu-2\n')
for model_name in model_names:
    tl = []
    for l in target_languages:
        if '_' + l in model_name:
            tl.append(l)

    model_name = os.path.join(model_path, model_name)
    model = load_model(model_name)

    for m, l in enumerate(tl):
        print('Bleu Score: (en-' + l + '): ')
        print('Monolithic Model: ')
        bleu=get_bleu(model, sampled_data[l], source_vectorization, target_vectorization, timestep)

        out.write(" ".join(tl)+','+l+','+str(bleu['1-grams'])+','+str(bleu['1-2-grams'])+'\n')

out.close()
