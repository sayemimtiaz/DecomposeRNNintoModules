import csv
import os
import random

import numpy as np
from keras.preprocessing.text import Tokenizer
import pandas as pd
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from relu_models.one_to_one.one_to_one_util import math_dataset_text
from util.common import binarize_multi_label


def preprocessVocDataset(split='train', target_img_width=32, target_img_height=32, all_labels={},
                         imagePath='PNGImages', dataPath='VOC2006', imageExt='png', fileExt='.txt'):
    dataPath = '/Users/Sayem/tensorflow_datasets/' + dataPath + '/'
    trainPath = os.path.join(dataPath, split)

    out = open(os.path.join(dataPath, split + ".csv"), "w")

    totalFlattenSize = target_img_height * target_img_width * 3
    for i in range(totalFlattenSize):
        out.write(str(i) + ',')
    out.write('label\n')

    nextLabel = 1
    trainSize = 0
    for file in os.listdir(os.path.join(trainPath, imagePath)):
        if file.endswith(imageExt):
            image_path = os.path.join(trainPath, imagePath, file)
            trainSize += 1
            image = tf.keras.preprocessing.image.load_img(image_path, target_size=(target_img_width, target_img_height))
            input_arr = tf.keras.preprocessing.image.img_to_array(image)
            input_arr /= 255.0
            # input_arr=input_arr.reshape(1, input_arr.shape[0]*input_arr.shape[1]*input_arr.shape[2])
            input_arr = input_arr.flatten()

            current_lb = set()
            skip = False
            objectFlag = False
            with open(os.path.join(trainPath, 'Annotations', file[:file.find(imageExt)] + fileExt), 'r') as f:
                line = f.readline()
                while line:
                    line = f.readline()
                    line = line.lower()
                    if fileExt == '.txt':
                        if line.startswith('objects with ground truth'):
                            tmp = line[line.find('{'):]
                            tmp = tmp.replace("{", '')
                            tmp = tmp.replace("}", '')
                            tmp = tmp.replace('"', '')
                            tmp = tmp.split(' ')

                            for lb in tmp:
                                if lb.strip() == '':
                                    continue
                                lb = lb.replace('pas', '')
                                if lb.startswith('car'):
                                    lb = 'car'
                                elif lb.startswith('bicycle'):
                                    lb = 'bicycle'
                                elif lb.startswith('bus'):
                                    lb = 'bus'
                                elif lb.startswith('cat'):
                                    lb = 'cat'
                                elif lb.startswith('cow'):
                                    lb = 'cow'
                                elif lb.startswith('dog'):
                                    lb = 'dog'
                                elif lb.startswith('horse'):
                                    lb = 'horse'
                                elif lb.startswith('motorbike'):
                                    lb = 'motorbike'
                                elif lb.startswith('person'):
                                    lb = 'person'
                                elif lb.startswith('sheep'):
                                    lb = 'sheep'
                                elif lb.startswith('null'):
                                    skip = True
                                    break
                                else:
                                    raise Exception(lb + ' not recognized')

                                if lb not in all_labels:
                                    all_labels[lb] = nextLabel
                                    nextLabel += 1

                                current_lb.add(all_labels[lb])

                            break
                    else:
                        if objectFlag and line.strip().startswith('<name>'):
                            tmp = line.replace("<name>", '')
                            tmp = tmp.replace("</name>", '')
                            tmp = tmp.strip()
                            if tmp not in all_labels:
                                all_labels[tmp] = nextLabel
                                nextLabel += 1

                            current_lb.add(all_labels[tmp])

                        if line.strip().startswith('<object>'):
                            objectFlag = True
                        if line.strip().startswith('</object>'):
                            objectFlag = False

            if skip:
                continue

            for i in input_arr:
                out.write(str(i) + ',')

            tlb = ''
            for lb in current_lb:
                tlb += str(lb) + ';'

            out.write(tlb[:-1] + '\n')

    print(trainSize)
    print(all_labels)

    out.close()

    return all_labels


def getSplitData(split='train', imagePath='PNGImages', dataPath='VOC2006'):
    dataPath = '/Users/Sayem/tensorflow_datasets/' + dataPath + '/'

    x_train = []
    y_train = []
    maxOutputTimestep = 0

    with open(os.path.join(dataPath, 'train.csv'), 'r') as csvfile:
        csvreader = csv.reader(csvfile)

        # extracting field names through first row
        fields = next(csvreader)

        # extracting each data row one by one
        for row in csvreader:
            r = []
            for i, f in enumerate(row):
                if fields[i] == 'label':
                    tlb = []
                    for lb in f.split(';'):
                        tlb.append(int(lb))
                    y_train.append(tlb)
                    maxOutputTimestep = max(maxOutputTimestep, len(tlb))
                else:
                    r.append(float(f))
            x_train.append(r)

    x_train = np.asarray(x_train)
    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])

    return x_train, y_train, maxOutputTimestep


def repeatVector(a, times):
    b = []
    for t in a:
        c = []
        for x in range(times):
            c.append(t)
        b.append(c)
    b = np.asarray(b)
    return b


def loadProcessedDataset(hot_encode=True, test_size=0.2, dataPath='VOC2006', flatten=False):
    x_train, y_train, ts1 = getSplitData('train', dataPath=dataPath)
    x_test, y_test, ts2 = getSplitData('test', dataPath=dataPath)
    maxOutputTimestep = max(ts1, ts2)

    # x_train = x_train.reshape(x_train.shape[0], 32,32,3)
    # x_test = x_test.reshape(x_test.shape[0],32,32,3)

    if flatten:
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

        x_train = repeatVector(x_train, maxOutputTimestep)
        x_test = repeatVector(x_test, maxOutputTimestep)

    print(maxOutputTimestep)
    y_train_padded = pad_sequences(y_train, maxlen=maxOutputTimestep, padding="pre", truncating="post")
    y_test_padded = pad_sequences(y_test, maxlen=maxOutputTimestep, padding="pre", truncating="post")

    if hot_encode:
        y_train = to_categorical(y_train_padded)
        y_test = to_categorical(y_test_padded)
        num_tags = y_train.shape[2]
    else:
        y_train = y_train_padded
        y_test = y_test_padded
        num_tags = 11

    # x = np.concatenate([x_train, x_test])
    # y = np.concatenate([y_train, y_test])
    #
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size,
    #                                                     random_state=4)
    # print(x_train.shape)
    return x_train, x_test, y_train, y_test, maxOutputTimestep, num_tags


def clean_toxic_testset():
    test = pd.read_csv('data/ToxicClassification/input/test.csv')
    test_label = pd.read_csv('data/ToxicClassification/input/test_labels.csv')

    outTest = open('data/ToxicClassification/input/test_cleaned.csv', "w")

    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y = test_label[list_classes].values
    outTest.write('id,comment_text,toxic,severe_toxic,obscene,threat,insult,identity_hate\n')
    d = []
    for i, _y in enumerate(y):
        if -1 not in _y:
            # d.append(test.iloc[i])
            text = test.iloc[i].comment_text
            # text=text.replace(',',';')
            text = text.replace('"', '\'')

            outTest.write(test.iloc[i].id + ',"' + text + '",' +
                          str(test_label.iloc[i]['toxic']) + ',' + str(test_label.iloc[i]['severe_toxic']) +
                          ',' + str(test_label.iloc[i]['obscene']) + ',' + str(test_label.iloc[i]['threat']) + ',' +
                          str(test_label.iloc[i]['insult']) + ',' + str(test_label.iloc[i]['identity_hate'])
                          + '\n')
    outTest.close()
    # test = test[~test['id'].isin(d)]
    # test_label = test_label[~test_label['id'].isin(d)]

    # return test,test_label


def get_multi_labels(y):
    l = []
    maxOutT = 0
    for _y in y:
        tl = []
        for i, t in enumerate(_y):
            if t > 0:
                tl.append(i + 2)
        if len(tl) > 0:
            l.append(tl)
            maxOutT = max(maxOutT, len(tl))

        elif len(tl) == 0:
            l.append([1])
    return l, maxOutT


def load_toxic_dataset(hot_encode=True, repeat=False, inputTimestep=20):
    data_path = os.path.dirname(os.path.realpath(__file__))
    train = pd.read_csv(os.path.join(data_path, 'data/ToxicClassification/input/train.csv'))
    test = pd.read_csv(os.path.join(data_path, 'data/ToxicClassification/input/test_cleaned.csv'))

    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    labels = train[list_classes].values
    test_label = test[list_classes].values

    y_train, ts1 = get_multi_labels(labels)
    y_test, ts2 = get_multi_labels(test_label)

    maxOutputTimeStep = max(ts1, ts2)

    list_sentences_train = train["comment_text"]
    list_sentences_test = test["comment_text"]

    vocab_size = 2000
    oov_tok = "<OOV>"

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(list_sentences_train)
    tokenizer.fit_on_texts(list_sentences_test)
    word_index = tokenizer.word_index

    x_train_encoded = tokenizer.texts_to_sequences(list_sentences_train)
    x_train_padded = pad_sequences(x_train_encoded, maxlen=inputTimestep, padding="pre", truncating="post")
    y_train_padded = pad_sequences(y_train, maxlen=maxOutputTimeStep, padding="pre", truncating="post")

    x_test_encoded = tokenizer.texts_to_sequences(list_sentences_test)
    x_test_padded = pad_sequences(x_test_encoded, maxlen=inputTimestep, padding="pre", truncating="post")
    y_test_padded = pad_sequences(y_test, maxlen=maxOutputTimeStep, padding="pre", truncating="post")

    if repeat:
        # x_train = x_train_encoded.reshape(x_train_encoded.shape[0], x_train.shape[1] * x_train.shape[2])
        # x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

        x_train_padded = repeatVector(x_train_padded, maxOutputTimeStep)
        x_test_padded = repeatVector(x_test_padded, maxOutputTimeStep)

    if hot_encode:
        y_test_padded = to_categorical(y_test_padded)
        y_train_padded = to_categorical(y_train_padded)

    return x_train_padded, x_test_padded, y_train_padded, y_test_padded, vocab_size, \
           maxOutputTimeStep, len(list_classes) + 2


def load_toxic_dataset_combine_math_qa(hot_encode=True, repeat=False, inputTimestep=20):
    x_train1, x_test1, y_train1, y_test1 = math_dataset_text()

    data_path = os.path.dirname(os.path.realpath(__file__))
    train = pd.read_csv(os.path.join(data_path, 'data/ToxicClassification/input/train.csv'))
    test = pd.read_csv(os.path.join(data_path, 'data/ToxicClassification/input/test_cleaned.csv'))

    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    labels = train[list_classes].values
    test_label = test[list_classes].values

    y_train, ts1 = get_multi_labels(labels)
    y_test, ts2 = get_multi_labels(test_label)

    maxOutputTimeStep = max(ts1, ts2)

    list_sentences_train = train["comment_text"]
    list_sentences_test = test["comment_text"]

    vocab_size = 7000
    oov_tok = "<OOV>"

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(list_sentences_train)
    tokenizer.fit_on_texts(list_sentences_test)
    tokenizer.fit_on_texts(x_train1)
    tokenizer.fit_on_texts(x_test1)

    x_train_encoded = tokenizer.texts_to_sequences(list_sentences_train)
    x_train_padded = pad_sequences(x_train_encoded, maxlen=inputTimestep, padding="pre", truncating="post")
    y_train_padded = pad_sequences(y_train, maxlen=maxOutputTimeStep, padding="pre", truncating="post")

    x_test_encoded = tokenizer.texts_to_sequences(list_sentences_test)
    x_test_padded = pad_sequences(x_test_encoded, maxlen=inputTimestep, padding="pre", truncating="post")
    y_test_padded = pad_sequences(y_test, maxlen=maxOutputTimeStep, padding="pre", truncating="post")

    if repeat:
        # x_train = x_train_encoded.reshape(x_train_encoded.shape[0], x_train.shape[1] * x_train.shape[2])
        # x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

        x_train_padded = repeatVector(x_train_padded, maxOutputTimeStep)
        x_test_padded = repeatVector(x_test_padded, maxOutputTimeStep)

    if hot_encode:
        y_test_padded = to_categorical(y_test_padded)
        y_train_padded = to_categorical(y_train_padded)

    return x_train_padded, x_test_padded, y_train_padded, y_test_padded, vocab_size, \
           maxOutputTimeStep, len(list_classes) + 2


def toxic_dataset_text():
    data_path = os.path.dirname(os.path.realpath(__file__))
    train = pd.read_csv(os.path.join(data_path, 'data/ToxicClassification/input/train.csv'))
    test = pd.read_csv(os.path.join(data_path, 'data/ToxicClassification/input/test_cleaned.csv'))

    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    labels = train[list_classes].values
    test_label = test[list_classes].values

    y_train, ts1 = get_multi_labels(labels)
    y_test, ts2 = get_multi_labels(test_label)

    list_sentences_train = train["comment_text"]
    list_sentences_test = test["comment_text"]

    return list_sentences_train, list_sentences_test, y_train, y_test


def binarize_toxic_dataset(x_train_original, y_train_original, x_test_original, y_test_original,
                           class1, class2, test_sample=1000, train_balanced_sample=True):
    y_train = []
    x_train = []
    train_count = {(class1, class2): 0, (0, class1): 0, (0, class2): 0}
    for i in range(len(y_train_original)):
        y_temp = binarize_multi_label(y_train_original[i], class1, class2)

        if class1 in y_temp and class2 in y_temp:
            y_train.append([class1, class2])
            x_train.append(x_train_original[i])
            train_count[(class1, class2)] += 1

        elif class1 in y_temp:
            y_train.append([0, class1])
            x_train.append(x_train_original[i])
            train_count[(0, class1)] += 1
        elif class2 in y_temp:
            y_train.append([0, class2])
            x_train.append(x_train_original[i])
            train_count[(0, class2)] += 1

    if train_balanced_sample:
        min_train = min(train_count[(class1, class2)], train_count[(0, class1)])
        if min_train == 0:
            min_train = max(train_count[(class1, class2)], train_count[(0, class1)])
        min_train = min(min_train, train_count[(0, class2)])
        if min_train == 0:
            min_train = max(min_train, train_count[(0, class2)])
        min_train = max(2*min_train, 20000)
        for c in train_count:
            while train_count[c] > min_train:
                idd = random.randint(0, train_count[c] - 1)
                del x_train[idd]
                del y_train[idd]
                train_count[c] -= 1

    y_test = []
    x_test = []
    test_count = {(class1, class2): 0, (0, class1): 0, (0, class2): 0}
    for i in range(len(y_test_original)):

        y_temp = binarize_multi_label(y_test_original[i], class1, class2)

        if class1 in y_temp and class2 in y_temp:
            if test_count[(class1, class2)] >= test_sample != -1:
                continue
            y_test.append([class1, class2])
            x_test.append(x_test_original[i])
            test_count[(class1, class2)] += 1
        elif class1 in y_temp:
            if test_count[(0, class1)] >= test_sample!=-1:
                continue
            y_test.append([0, class1])
            x_test.append(x_test_original[i])
            test_count[(0, class1)] += 1

        elif class2 in y_temp:
            if test_count[(0, class2)] >= test_sample != -1:
                continue
            y_test.append([0, class2])
            x_test.append(x_test_original[i])
            test_count[(0, class2)] += 1

    return np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test)


def unarize_toxic_dataset(x_train_original, y_train_original, x_test_original, y_test_original,
                          class1, label1):
    y_train = []
    x_train = []
    for i in range(len(y_train_original)):
        if class1 in y_train_original[i]:
            y_train.append(label1)
            x_train.append(x_train_original[i])

    y_test = []
    x_test = []
    for i in range(len(y_test_original)):
        if class1 in y_test_original[i]:
            y_test.append(label1)
            x_test.append(x_test_original[i])

    return np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test)

# all_labels=preprocessVocDataset(dataPath='VOC2007', imagePath='JPEGImages',
#                                     imageExt='.jpg', fileExt='.xml')
# preprocessVocDataset(split='test', all_labels=all_labels, dataPath='VOC2007', imagePath='JPEGImages',
#                                     imageExt='.jpg', fileExt='.xml')
# loadProcessedDataset(dataPath='VOC2007')

# load_toxic_dataset()
# clean_toxic_testset()
# binarize_toxic_dataset(2, 3)
