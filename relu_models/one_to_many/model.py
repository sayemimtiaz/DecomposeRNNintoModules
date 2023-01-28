from keras.models import Sequential
from keras.layers import Dense, Activation, TimeDistributed, RepeatVector, Flatten, Conv2D, Embedding, Dropout, \
    SimpleRNN
from keras.utils import np_utils
from relu_models.one_to_many.one_to_many_util import *

x_train, x_test, y_train, y_test, num_words, timestep, nb_classes = load_toxic_dataset(repeat=False)
# x_train, x_test, y_train, y_test, num_words, timestep, nb_classes = load_toxic_dataset_combine_math_qa(repeat=False)

embed_size = 50

model = Sequential()

rnn_model = Sequential()

model.add(Embedding(num_words, embed_size, input_length=20))

model.add(Flatten())
model.add(RepeatVector(timestep))

model.add(SimpleRNN(100, return_sequences=True, activation='relu'))
model.add(SimpleRNN(100, return_sequences=True,  activation='relu'))
model.add(SimpleRNN(100, return_sequences=True,  activation='relu'))
model.add(SimpleRNN(100, return_sequences=True,  activation='tanh'))
# model.add(Dropout(0.2))

model.add(TimeDistributed(Dense(nb_classes, activation='softmax')))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())

epochs = 5

history = model.fit(x_train,
                    y_train,
                    epochs=epochs,
                    batch_size=32,
                    verbose=2)
scores = model.evaluate(x_test, y_test, verbose=2)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

scores = model.predict(x_test)
model.save('h5/model4.h5')
