from keras.models import Sequential
from keras.layers import Dense, Activation, TimeDistributed, RepeatVector, Flatten, Conv2D, Embedding, Dropout, LSTM

from relu_models.one_to_one.one_to_one_util import load_math_dataset, load_math_dataset_with_toxic

# x_train, x_test, y_train, y_test, num_words, timestep, nb_classes = load_math_dataset()
x_train, x_test, y_train, y_test, num_words, timestep, nb_classes = load_math_dataset_with_toxic()

embed_size = 50

model = Sequential()

rnn_model = Sequential()

model.add(Embedding(num_words, embed_size, input_length=timestep))

model.add(Flatten())
model.add(RepeatVector(1))

model.add(LSTM(100, return_sequences=True))
# # model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=True))
# # model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=True))
# model.add(Dropout(0.2))
model.add(LSTM(100))
# model.add(Dropout(0.2))
model.add(Dense(nb_classes, activation='softmax'))

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

# scores=model.predict(x_test)
model.save('h5/model4_combined.h5')

