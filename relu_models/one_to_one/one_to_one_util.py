import csv
import os
import numpy as np
from keras.preprocessing.text import Tokenizer
import pandas as pd
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds
from sklearn.utils import shuffle


def load_math_dataset(hot_encode=True, inputTimestep=20):
    data = \
        tfds.as_numpy(tfds.load('math_qa',
                                batch_size=-1))
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    list_classes = {'gain': 0, 'general': 1, 'geometry': 2, 'other': 3,
                    'physics': 4, 'probability': 5}
    for i in range(len(data['test']['category'])):
        y_test.append(list_classes[data['test']['category'][i].decode('utf-8')])
        x_test.append(data['test']['Problem'][i].decode('utf-8'))
    for i in range(len(data['train']['category'])):
        y_train.append(list_classes[data['train']['category'][i].decode('utf-8')])
        x_train.append(data['train']['Problem'][i].decode('utf-8'))

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    if hot_encode:
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        num_tags = y_train.shape[1]
    else:
        num_tags = 6

    vocab_size = 5000
    # inputTimestep = 20
    oov_tok = "<OOV>"

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(x_train)
    tokenizer.fit_on_texts(x_test)

    x_train = tokenizer.texts_to_sequences(x_train)
    x_train = pad_sequences(x_train, maxlen=inputTimestep, padding="pre", truncating="post")

    x_test = tokenizer.texts_to_sequences(x_test)
    x_test = pad_sequences(x_test, maxlen=inputTimestep, padding="pre", truncating="post")

    return x_train, x_test, y_train, y_test, vocab_size, inputTimestep, num_tags


def load_math_dataset_with_toxic(hot_encode=True):
    from relu_models.one_to_many.one_to_many_util import toxic_dataset_text

    x_train1, x_test1, y_train1, y_test1 = toxic_dataset_text()

    data = \
        tfds.as_numpy(tfds.load('math_qa',
                                batch_size=-1))
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    list_classes = {'gain': 0, 'general': 1, 'geometry': 2, 'other': 3, 'physics': 4, 'probability': 5}
    for i in range(len(data['test']['category'])):
        y_test.append(list_classes[data['test']['category'][i].decode('utf-8')])
        x_test.append(data['test']['Problem'][i].decode('utf-8'))
    for i in range(len(data['train']['category'])):
        y_train.append(list_classes[data['train']['category'][i].decode('utf-8')])
        x_train.append(data['train']['Problem'][i].decode('utf-8'))

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    if hot_encode:
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        num_tags = y_train.shape[1]
    else:
        num_tags = 6

    vocab_size = 7000
    inputTimestep = 20
    oov_tok = "<OOV>"

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(x_train)
    tokenizer.fit_on_texts(x_test)
    tokenizer.fit_on_texts(x_train1)
    tokenizer.fit_on_texts(x_test1)

    x_train = tokenizer.texts_to_sequences(x_train)
    x_train = pad_sequences(x_train, maxlen=inputTimestep, padding="pre", truncating="post")

    x_test = tokenizer.texts_to_sequences(x_test)
    x_test = pad_sequences(x_test, maxlen=inputTimestep, padding="pre", truncating="post")

    return x_train, x_test, y_train, y_test, vocab_size, inputTimestep, num_tags


def math_dataset_text():
    data = \
        tfds.as_numpy(tfds.load('math_qa',
                                batch_size=-1))
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    list_classes = {'gain': 0, 'general': 1, 'geometry': 2, 'other': 3, 'physics': 4, 'probability': 5}
    for i in range(len(data['test']['category'])):
        y_test.append(list_classes[data['test']['category'][i].decode('utf-8')])
        x_test.append(data['test']['Problem'][i].decode('utf-8'))
    for i in range(len(data['train']['category'])):
        y_train.append(list_classes[data['train']['category'][i].decode('utf-8')])
        x_train.append(data['train']['Problem'][i].decode('utf-8'))

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    return x_train, x_test, y_train, y_test


def binarize_math_qa(class1, class2, hot_encode=False):
    x_train_original, x_test_original, y_train_original, y_test_original, num_words, \
    timestep, nb_classes = load_math_dataset(hot_encode)

    # train data change
    x_train = x_train_original[y_train_original == class1]
    x_train = np.append(x_train, x_train_original[y_train_original == class2], axis=0)
    y_train = []
    for x in y_train_original[y_train_original == class1]:
        y_train.append(0)
    y_train = np.array(y_train)
    temp = []
    for x in y_train_original[y_train_original == class2]:
        temp.append(1)
    temp = np.array(temp)
    y_train = np.append(y_train, temp)
    x_train, y_train = shuffle(x_train, y_train, random_state=0)

    x_test = x_test_original[y_test_original == class1]  # test Data change
    x_test = np.append(x_test, x_test_original[y_test_original == class2], axis=0)
    y_test = []
    for x in y_test_original[y_test_original == class1]:
        y_test.append(0)
    y_test = np.array(y_test)
    temp = []
    for x in y_test_original[y_test_original == class2]:
        temp.append(1)
    temp = np.array(temp)
    y_test = np.append(y_test, temp)
    x_test, y_test = shuffle(x_test, y_test, random_state=0)

    return x_train, y_train, x_test, y_test


def unarize_math_qa(x_train_original, y_train_original, x_test_original, y_test_original, class1, label1):
    # train data change
    x_train = x_train_original[y_train_original == class1]
    y_train = []
    for x in y_train_original[y_train_original == class1]:
        y_train.append(label1)
    y_train = np.array(y_train)

    x_train, y_train = shuffle(x_train, y_train, random_state=0)

    x_test = x_test_original[y_test_original == class1]  # test Data change
    y_test = []
    for x in y_test_original[y_test_original == class1]:
        y_test.append(label1)
    y_test = np.array(y_test)
    x_test, y_test = shuffle(x_test, y_test, random_state=0)

    return x_train, y_train, x_test, y_test


def load_math_qa_for_replace(x_train_original, y_train_original, x_test_original, y_test_original, class1):
    # train data change

    x_train = x_train_original[y_train_original != class1]
    y_train = []
    for x in y_train_original[y_train_original != class1]:
        y_train.append(x)

    y_train = np.array(y_train)

    x_train, y_train = shuffle(x_train, y_train, random_state=0)

    x_test = x_test_original[y_test_original != class1]  # test Data change
    y_test = []
    for x in y_test_original[y_test_original != class1]:
        y_test.append(x)

    y_test = np.array(y_test)
    x_test, y_test = shuffle(x_test, y_test, random_state=0)

    return x_train, y_train, x_test, y_test

# load_math_dataset()
# binarize_math_qa(2,3)
