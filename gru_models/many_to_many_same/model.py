import os

from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, TimeDistributed, Dropout, GRU
from keras.utils import np_utils
from keras import backend as K
import numpy as np
from sklearn.metrics import accuracy_score

from evaluation.accuracy_computer import getMonolithicModelAccuracyAnyToMany
from util.nltk_util import load_pos_tagged_dataset, load_pos_tagged_dataset_with_clinc

X_train, X_test, Y_train, Y_test, num_words, timestep, nb_classes = load_pos_tagged_dataset()
# X_train, X_test, Y_train, Y_test, num_words, timestep, nb_classes = \
#     load_pos_tagged_dataset_with_clinc()
embed_size = 100

# Number of hidden units to use:
nb_units = 100

# create architecture

rnn_model = Sequential()

# create embedding layer - usually the first layer in text problems
rnn_model.add(Embedding(input_dim=num_words + 1,  # vocabulary size - number of unique words in data
                        output_dim=embed_size,  # length of vector with which each word is represented
                        input_length=timestep  # length of input sequence))
                        ))


# rnn_model.add(GRU(nb_units, return_sequences=True, reset_after=False, activation='tanh'))
# rnn_model.add(GRU(nb_units, return_sequences=True, reset_after=False, activation='tanh'))
# rnn_model.add(GRU(nb_units, return_sequences=True, reset_after=False, activation='tanh'))
rnn_model.add(GRU(nb_units, return_sequences=True, reset_after=False, activation='tanh'))
# rnn_model.add(Dropout(0.3))

rnn_model.add(TimeDistributed(Dense(nb_classes, activation='softmax')))

rnn_model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

print(rnn_model.summary())

# epochs = 10
#
# history = rnn_model.fit(X_train,
#                         Y_train,
#                         epochs=epochs,
#                         batch_size=128,
#                         verbose=2)
#
# scores = rnn_model.evaluate(X_test, Y_test, verbose=2)
#
# print("%s: %.2f%%" % (rnn_model.metrics_names[1], scores[1] * 100))

rnn_model.save('h5/model1_scratch.h5')
# X_train,X_test,Y_train,Y_test, num_words,timestep,nb_classes = load_pos_tagged_dataset(hot_encode=False)
# #
# print(findModelAccuracyForManyToMany(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model1_many_to_one.h5'), X_test, Y_test, skipDummyLabel=False))
