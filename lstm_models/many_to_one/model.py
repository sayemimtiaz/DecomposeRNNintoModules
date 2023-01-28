from keras.models import Sequential
from keras.layers import Dense, Embedding, TimeDistributed, Dropout, LSTM
from keras.utils import np_utils
from keras import backend as K
import numpy as np
from util.nltk_util import load_brown_category_dataset, loadTensorFlowDataset, loadClincOos, loadClincOosWithPosTag

# X_train, X_test, Y_train, Y_test, num_words, timestep, nb_classes = loadClincOos()
X_train, X_test, Y_train, Y_test, num_words, timestep, nb_classes = loadClincOosWithPosTag(timestep=15)

embed_size = 50

# Number of hidden units to use:
nb_units = 100

# create architecture

rnn_model = Sequential()

# create embedding layer - usually the first layer in text problems
rnn_model.add(Embedding(input_dim=num_words + 1,  # vocabulary size - number of unique words in data
                        output_dim=embed_size,  # length of vector with which each word is represented
                        input_length=timestep  # length of input sequence))
                        ))

rnn_model.add(LSTM(nb_units,return_sequences=True))
# rnn_model.add(Dropout(0.1))
rnn_model.add(LSTM(nb_units,return_sequences=True))
# rnn_model.add(Dropout(0.1))
rnn_model.add(LSTM(nb_units,return_sequences=True))
# rnn_model.add(Dropout(0.1))
rnn_model.add(LSTM(nb_units,return_sequences=False))
rnn_model.add(Dropout(0.2))

rnn_model.add(Dense(nb_classes, activation='softmax'))

rnn_model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

print(rnn_model.summary())

epochs = 15

history = rnn_model.fit(X_train,
                        Y_train,
                        epochs=epochs,
                        batch_size=128,
                        verbose=2)
scores = rnn_model.evaluate(X_test, Y_test, verbose=2)
print("%s: %.2f%%" % (rnn_model.metrics_names[1], scores[1] * 100))

rnn_model.save('h5/model4_combined.h5')
