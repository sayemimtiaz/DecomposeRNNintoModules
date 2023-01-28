import csv
import os
import random
import re

import nltk
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus import brown, conll2000, treebank

from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle

from data_type.constants import *
from keras.preprocessing.text import Tokenizer

import tensorflow as tf
import tensorflow_datasets as tfds
import spacy


def load_pos_tagged_dataset(test_size=0.25, timestep=15, hot_encode=True):
    # nltk.download('brown')
    # nltk.download('treebank')
    # nltk.download('conll2000')
    # nltk.download('universal_tagset')

    # list of sentences that are list of tuples like (<word>, <tag>)
    treebank_corpus = []
    brown_corpus = []
    conll_corpus = []
    if USE_TREEBANK_CORPUS:
        treebank_corpus = treebank.tagged_sents(tagset='universal')
    if USE_BROWN_CORPUS:
        brown_corpus = brown.tagged_sents(tagset='universal')
    if USE_CONLL_CORPUS:
        conll_corpus = conll2000.tagged_sents(tagset='universal')

    tagged_sentences = treebank_corpus + brown_corpus + conll_corpus

    X = []  # store input sequence
    Y = []  # store output sequence

    for sentence in tagged_sentences:
        X_sentence = []
        Y_sentence = []
        for entity in sentence:
            X_sentence.append(entity[0])  # entity[0] contains the word
            Y_sentence.append(entity[1])  # entity[1] contains corresponding tag

        X.append(X_sentence)
        Y.append(Y_sentence)

    num_words = len(set([word.lower() for sentence in X for word in sentence]))
    num_tags = len(set([word.lower() for sentence in Y for word in sentence]))

    # print("Total number of tagged sentences: {}".format(len(X)))
    # print("Vocabulary size: {}".format(num_words))
    # print("Total number of tags: {}".format(num_tags))
    # print(set([word.lower() for sentence in Y for word in sentence]))
    word_tokenizer = Tokenizer()  # instantiate tokeniser
    word_tokenizer.fit_on_texts(X)  # fit tokeniser on data
    X_encoded = word_tokenizer.texts_to_sequences(X)

    tag_tokenizer = Tokenizer()
    tag_tokenizer.fit_on_texts(Y)
    Y_encoded = tag_tokenizer.texts_to_sequences(Y)

    # make sure that each sequence of input and output is same length

    different_length = [1 if len(input) != len(output) else 0 for input, output in zip(X_encoded, Y_encoded)]
    # print("{} sentences have disparate input-output lengths.".format(sum(different_length)))

    # check length of longest sentence
    lengths = [len(seq) for seq in X_encoded]
    # print("Length of longest sentence: {}".format(max(lengths)))
    X_padded = pad_sequences(X_encoded, maxlen=timestep, padding="pre", truncating="post")
    Y_padded = pad_sequences(Y_encoded, maxlen=timestep, padding="pre", truncating="post")
    X, Y = X_padded, Y_padded

    if hot_encode:
        Y = to_categorical(Y)
    # print(Y.shape)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=4)
    return X_train, X_test, Y_train, Y_test, num_words, timestep, num_tags + 1


def load_pos_tagged_dataset_just_text():
    treebank_corpus = []
    brown_corpus = []
    conll_corpus = []
    if USE_TREEBANK_CORPUS:
        treebank_corpus = treebank.tagged_sents(tagset='universal')
    if USE_BROWN_CORPUS:
        brown_corpus = brown.tagged_sents(tagset='universal')
    if USE_CONLL_CORPUS:
        conll_corpus = conll2000.tagged_sents(tagset='universal')

    tagged_sentences = treebank_corpus + brown_corpus + conll_corpus

    X = []  # store input sequence
    Y = []  # store output sequence

    for sentence in tagged_sentences:
        X_sentence = []
        Y_sentence = []
        for entity in sentence:
            X_sentence.append(entity[0])  # entity[0] contains the word
            Y_sentence.append(entity[1])  # entity[1] contains corresponding tag

        X.append(X_sentence)
        Y.append(Y_sentence)

    return X, Y


def load_pos_tagged_dataset_with_clinc(test_size=0.25, timestep=15, hot_encode=True):
    x_train1, x_test1, _, _ = loadClincOosJustText()
    treebank_corpus = []
    brown_corpus = []
    conll_corpus = []
    if USE_TREEBANK_CORPUS:
        treebank_corpus = treebank.tagged_sents(tagset='universal')
    if USE_BROWN_CORPUS:
        brown_corpus = brown.tagged_sents(tagset='universal')
    if USE_CONLL_CORPUS:
        conll_corpus = conll2000.tagged_sents(tagset='universal')

    tagged_sentences = treebank_corpus + brown_corpus + conll_corpus

    X = []  # store input sequence
    Y = []  # store output sequence

    for sentence in tagged_sentences:
        X_sentence = []
        Y_sentence = []
        for entity in sentence:
            X_sentence.append(entity[0])  # entity[0] contains the word
            Y_sentence.append(entity[1])  # entity[1] contains corresponding tag

        X.append(X_sentence)
        Y.append(Y_sentence)

    # num_words = len(set([word.lower() for sentence in X for word in sentence]))
    num_tags = len(set([word.lower() for sentence in Y for word in sentence]))

    # print("Total number of tagged sentences: {}".format(len(X)))
    # print("Vocabulary size: {}".format(num_words))
    # print("Total number of tags: {}".format(num_tags))
    # print(set([word.lower() for sentence in Y for word in sentence]))
    word_tokenizer = Tokenizer()  # instantiate tokeniser
    word_tokenizer.fit_on_texts(X)  # fit tokeniser on data
    word_tokenizer.fit_on_texts(x_train1)  # fit tokeniser on data
    word_tokenizer.fit_on_texts(x_test1)  # fit tokeniser on data
    X_encoded = word_tokenizer.texts_to_sequences(X)

    tag_tokenizer = Tokenizer()
    tag_tokenizer.fit_on_texts(Y)
    Y_encoded = tag_tokenizer.texts_to_sequences(Y)

    # make sure that each sequence of input and output is same length

    different_length = [1 if len(input) != len(output) else 0 for input, output in zip(X_encoded, Y_encoded)]
    # print("{} sentences have disparate input-output lengths.".format(sum(different_length)))

    # check length of longest sentence
    lengths = [len(seq) for seq in X_encoded]
    # print("Length of longest sentence: {}".format(max(lengths)))
    X_padded = pad_sequences(X_encoded, maxlen=timestep, padding="pre", truncating="post")
    Y_padded = pad_sequences(Y_encoded, maxlen=timestep, padding="pre", truncating="post")
    X, Y = X_padded, Y_padded

    if hot_encode:
        Y = to_categorical(Y)
    # print(Y.shape)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=4)
    return X_train, X_test, Y_train, Y_test, len(word_tokenizer.word_index), timestep, num_tags + 1


def binarize_pos_tagged(x_train_original, y_train_original, x_test_original, y_test_original,
                        class1, class2, test_sample=1000, train_balanced_sample=True):
    y_train = []
    x_train = []
    train_count = {(class1, class2): 0, class1: 0, class2: 0}
    for i in range(len(y_train_original)):
        y_temp = []
        for j in y_train_original[i]:
            if j == class1:
                y_temp.append(class1)
            elif j == class2:
                y_temp.append(class2)
            else:
                y_temp.append(0)

        if class1 in y_temp and class2 in y_temp:
            train_count[(class1, class2)] += 1
        elif class1 in y_temp:
            train_count[class1] += 1
        elif class2 in y_temp:
            train_count[class2] += 1

        y_temp = np.asarray(y_temp)
        if class1 in y_temp or class2 in y_temp:
            y_train.append(y_temp)
            x_train.append(x_train_original[i])

    if train_balanced_sample:
        min_train = min(train_count[(class1, class2)], train_count[class1])
        if min_train == 0:
            min_train = max(train_count[(class1, class2)], train_count[class1])
        min_train = min(min_train, train_count[class2])
        if min_train == 0:
            min_train = max(min_train, train_count[class2])
        min_train = max(min_train, 10000)
        for c in train_count:
            while train_count[c] > min_train:
                idd = random.randint(0, train_count[c] - 1)
                del x_train[idd]
                del y_train[idd]
                train_count[c] -= 1
    y_test = []
    x_test = []
    test_count = {(class1, class2): 0, class1: 0, class2: 0}
    for i in range(len(y_test_original)):
        y_temp = []
        for j in y_test_original[i]:
            if j == class1:
                y_temp.append(class1)
            elif j == class2:
                y_temp.append(class2)
            else:
                y_temp.append(0)

        y_temp = np.asarray(y_temp)

        if class1 in y_temp and class2 in y_temp:
            if test_count[(class1, class2)] >= test_sample != -1:
                continue
            test_count[(class1, class2)] += 1
        elif class1 in y_temp:
            if test_count[class1] >= test_sample!=-1:
                continue
            test_count[class1] += 1
        elif class2 in y_temp:
            if test_count[class2] >= test_sample!=-1:
                continue
            test_count[class2] += 1

        if class1 in y_temp or class2 in y_temp:
            y_test.append(y_temp)
            x_test.append(x_test_original[i])

    return np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test)


def load_brown_category_dataset(test_size=0.20, timestep=400, hot_encode=True):
    # nltk.download('brown')

    for f in brown.fileids():
        if len(brown.categories(f)) > 1:
            print('multi-label')

    X = []  # store input sequence
    Y = []  # store output sequence
    for fileId in brown.fileids():
        X_sentence = []
        for entity in brown.words(fileId):
            X_sentence.append(entity)

        X.append(X_sentence)
        tmpY = brown.categories(fileId)[0]
        tmpY = tmpY.replace('_', '')
        Y.append(tmpY)

    num_tags = len(brown.categories())
    num_words = len(set([word.lower() for sentence in X for word in sentence]))
    print(num_tags)

    word_tokenizer = Tokenizer()  # instantiate tokeniser
    word_tokenizer.fit_on_texts(X)  # fit tokeniser on data
    X_encoded = word_tokenizer.texts_to_sequences(X)

    tag_tokenizer = Tokenizer()
    tag_tokenizer.fit_on_texts(Y)
    Y_encoded = tag_tokenizer.texts_to_sequences(Y)
    for i in range(len(Y_encoded)):
        Y_encoded[i][0] = Y_encoded[i][0] - 1

    # check length of longest sentence
    lengths = [len(seq) for seq in X_encoded]
    print("Length of longest sentence: {}".format(max(lengths)))
    X_padded = pad_sequences(X_encoded, maxlen=timestep, padding="pre", truncating="post")
    # Y_padded = pad_sequences(Y_encoded, maxlen=timestep, padding="pre", truncating="post")
    X, Y = X_padded, Y_encoded
    if hot_encode:
        Y = to_categorical(Y)
        # Y=np.delete(Y, 0, axis=1)
    # print(X.shape)
    # print(Y.shape)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=4)
    if not hot_encode:
        return X_train, X_test, Y_train, Y_test, num_words, timestep, num_tags
    return X_train, X_test, Y_train, Y_test, num_words, timestep, Y_train.shape[1]


def text_processing(tweet, decode=True):
    # remove https links
    if decode:
        tweet = tweet.decode('utf-8')
    clean_tweet = re.sub(r'http\S+', '', tweet)
    # remove punctuation marks
    punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'
    clean_tweet = ''.join(ch for ch in clean_tweet if ch not in set(punctuation))
    # convert text to lowercase
    clean_tweet = clean_tweet.lower()
    # remove numbers
    clean_tweet = re.sub('\d', ' ', clean_tweet)
    # remove whitespaces
    clean_tweet = ' '.join(clean_tweet.split())
    clean_tweet = nltk.word_tokenize(clean_tweet)
    return clean_tweet


def lemmatization(tweet):
    # python -m spacy download en
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    # lemma_tweet = []
    # for i in tweets:
    #     t = [token.lemma_ for token in nlp(i)]
    #     lemma_tweet.append(' '.join(t))
    return [token.lemma_.lower() for token in nlp(tweet)]


def stem(tweet):
    tokenizer = nltk.RegexpTokenizer('[a-zA-Z0-9@]+')
    stemmer = nltk.LancasterStemmer()

    return [stemmer.stem(token) if not token.startswith('@') else token
            for token in tokenizer.tokenize(tweet)]


def loadClincOos(hot_encode=True, timestep=15):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    train, test = \
        tfds.as_numpy(tfds.load('clinc_oos',
                                split=['train', 'test'],
                                batch_size=-1))
    for i in range(len(test['domain'])):
        y_test.append(test['domain'][i])
        X_test.append(test['text'][i])
    for i in range(len(train['domain'])):
        y_train.append(train['domain'][i])
        X_train.append(train['text'][i])

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    if hot_encode:
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        num_tags = y_train.shape[1]
    else:
        num_tags = 10

    cleaned_x_train = []
    cleaned_x_test = []
    for tweet in X_train:
        cleaned_x_train.append(text_processing(tweet, decode=True))
    for tweet in X_test:
        cleaned_x_test.append(text_processing(tweet))

    X_train = cleaned_x_train
    X_test = cleaned_x_test

    unique_in_train = set([word for sentence in X_train for word in sentence])
    unique_in_test = set([word for sentence in X_test for word in sentence])
    num_words = len(unique_in_train.union(unique_in_test))
    # print(num_words)
    word_tokenizer = Tokenizer()  # instantiate tokeniser
    word_tokenizer.fit_on_texts(X_train)  # fit tokeniser on data
    word_tokenizer.fit_on_texts(X_test)  # fit tokeniser on data
    X_train_encoded = word_tokenizer.texts_to_sequences(X_train)
    X_test_encoded = word_tokenizer.texts_to_sequences(X_test)

    lengths_train = [len(seq) for seq in X_train_encoded]
    lengths_test = [len(seq) for seq in X_test_encoded]
    # timestep=max(max(lengths_train), max(lengths_test))
    # timestep=20

    # print("Length of longest sentence: {}".format(timestep))
    X_train_padded = pad_sequences(X_train_encoded, maxlen=timestep, padding="pre", truncating="post")
    X_test_padded = pad_sequences(X_test_encoded, maxlen=timestep, padding="pre", truncating="post")

    X_train = np.asarray(X_train_padded)
    X_test = np.asarray(X_test_padded)

    return X_train, X_test, y_train, y_test, num_words, timestep, num_tags


def loadClincOosJustText():
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    train, test = \
        tfds.as_numpy(tfds.load('clinc_oos',
                                split=['train', 'test'],
                                batch_size=-1))
    for i in range(len(test['domain'])):
        y_test.append(test['domain'][i])
        X_test.append(test['text'][i])
    for i in range(len(train['domain'])):
        y_train.append(train['domain'][i])
        X_train.append(train['text'][i])

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    cleaned_x_train = []
    cleaned_x_test = []
    for tweet in X_train:
        cleaned_x_train.append(text_processing(tweet, decode=True))
    for tweet in X_test:
        cleaned_x_test.append(text_processing(tweet))

    X_train = cleaned_x_train
    X_test = cleaned_x_test

    return X_train, X_test, y_train, y_test


def loadClincOosWithPosTag(hot_encode=True, timestep=15):
    X_train1, y_train1 = load_pos_tagged_dataset_just_text()
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    train, test = \
        tfds.as_numpy(tfds.load('clinc_oos',
                                split=['train', 'test'],
                                batch_size=-1))
    for i in range(len(test['domain'])):
        y_test.append(test['domain'][i])
        X_test.append(test['text'][i])
    for i in range(len(train['domain'])):
        y_train.append(train['domain'][i])
        X_train.append(train['text'][i])

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    if hot_encode:
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        num_tags = y_train.shape[1]
    else:
        num_tags = 10

    cleaned_x_train = []
    cleaned_x_test = []
    for tweet in X_train:
        cleaned_x_train.append(text_processing(tweet, decode=True))
    for tweet in X_test:
        cleaned_x_test.append(text_processing(tweet))

    X_train = cleaned_x_train
    X_test = cleaned_x_test

    # unique_in_train = set([word for sentence in X_train for word in sentence])
    # unique_in_test = set([word for sentence in X_test for word in sentence])
    # num_words = len(unique_in_train.union(unique_in_test))
    # print(num_words)
    word_tokenizer = Tokenizer()  # instantiate tokeniser
    word_tokenizer.fit_on_texts(X_train)  # fit tokeniser on data
    word_tokenizer.fit_on_texts(X_test)  # fit tokeniser on data

    word_tokenizer.fit_on_texts(X_train1)  # fit tokeniser on data

    X_train_encoded = word_tokenizer.texts_to_sequences(X_train)
    X_test_encoded = word_tokenizer.texts_to_sequences(X_test)

    lengths_train = [len(seq) for seq in X_train_encoded]
    lengths_test = [len(seq) for seq in X_test_encoded]
    # timestep=max(max(lengths_train), max(lengths_test))
    # timestep=20

    # print("Length of longest sentence: {}".format(timestep))
    X_train_padded = pad_sequences(X_train_encoded, maxlen=timestep, padding="pre", truncating="post")
    X_test_padded = pad_sequences(X_test_encoded, maxlen=timestep, padding="pre", truncating="post")

    X_train = np.asarray(X_train_padded)
    X_test = np.asarray(X_test_padded)

    return X_train, X_test, y_train, y_test, len(word_tokenizer.word_index), timestep, num_tags


def binarize_clinc_oos(class1, class2, hot_encode=False):
    x_train_original, x_test_original, y_train_original, y_test_original, num_words, \
    timestep, nb_classes = loadClincOos(hot_encode)

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


def loadTensorFlowDataset(datasetName, trainSize=1, test_size=0.25, hot_encode=True):
    (X_train, y_train), \
    (X_test, y_test) = \
        tfds.as_numpy(tfds.load(datasetName,
                                split=['train[:' + str(trainSize) + '%]', 'test'],
                                batch_size=-1,
                                as_supervised=True))
    X_train = []
    y_train = []

    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           'cleaned_sentiment140.csv'), 'r') as csvfile:
        csvreader = csv.reader(csvfile)

        # extracting field names through first row
        fields = next(csvreader)

        # extracting each data row one by one
        for row in csvreader:
            X_train.append(row[0].replace('"', ''))
            y_train.append(int(row[1]))
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    # X_train=X_train[:5000]
    # y_train = y_train[:5000]
    # X_train, y_train = \
    #     tfds.as_numpy(tfds.load(datasetName,
    #                             split='train[:' + str(trainSize) + '%]',
    #                             batch_size=-1,
    #                             as_supervised=True))
    if hot_encode:
        y_train = to_categorical(y_train)
        # y_test = to_categorical(y_test)
        num_tags = y_train.shape[1]
    else:
        num_tags = 5

    cleaned_x_train = []
    # # cleaned_x_test = []
    for tweet in X_train:
        cleaned_x_train.append(text_processing(tweet, decode=False))
    # # for tweet in X_test:
    # #     cleaned_x_test.append(text_processing(tweet))
    #
    X_train = cleaned_x_train
    # X_test = cleaned_x_test
    # X_train = lemmatization(X_train)
    # X_test = lemmatization(cleaned_x_test)
    # X_train=np.asarray(cleaned_x_train)
    # X_test = np.asarray(cleaned_x_test)

    unique_in_train = set([word for sentence in X_train for word in sentence])
    unique_in_test = set()
    # unique_in_test = set([word for sentence in X_test for word in sentence])
    num_words = len(unique_in_train.union(unique_in_test))

    word_tokenizer = Tokenizer()  # instantiate tokeniser
    word_tokenizer.fit_on_texts(X_train)  # fit tokeniser on data
    # word_tokenizer.fit_on_texts(X_test)  # fit tokeniser on data
    X_train_encoded = word_tokenizer.texts_to_sequences(X_train)
    # X_test_encoded = word_tokenizer.texts_to_sequences(X_test)

    lengths_train = [len(seq) for seq in X_train_encoded]
    # lengths_test = [len(seq) for seq in X_test_encoded]
    # timestep=max(max(lengths_train), max(lengths_test))
    timestep = max(lengths_train)

    print("Length of longest sentence: {}".format(timestep))
    X_train_padded = pad_sequences(X_train_encoded, maxlen=timestep, padding="pre", truncating="post")
    # X_test_padded = pad_sequences(X_test_encoded, maxlen=timestep, padding="pre", truncating="post")

    X_train_padded, X_test_padded, y_train, y_test = train_test_split(X_train_padded, y_train, test_size=test_size,
                                                                      random_state=4)

    return X_train_padded, X_test_padded, y_train, y_test, num_words, timestep, num_tags


def saveTFDataset(datasetName, trainSize=100):
    X_train, y_train = \
        tfds.as_numpy(tfds.load(datasetName,
                                split='train[:' + str(trainSize) + '%]',
                                batch_size=-1,
                                as_supervised=True))

    out = open('cleaned_sentiment140.csv', 'w')
    out.write('Twwet,Sentiment\n')

    taken = np.array([0] * 5)
    for index, tweet in enumerate(X_train):
        if np.sum(taken) >= 10:
            break
        if taken[y_train[index]] > 2:
            continue

        taken[y_train[index]] += 1

        tweet = lemmatization(text_processing(tweet))
        tweet = ' '.join(tweet)
        tweet = tweet.replace('.', ' ')
        tweet = tweet.replace('\\', '')
        tweet = tweet.replace('\'', '')
        tweet = tweet.replace('"', '')
        out.write('"' + tweet + '",' + str(y_train[index]) + '\n')

# loadClincOos()
# saveTFDataset('Sentiment140')
# print(' '.join(lemmatization('i am going there.')))
# loadTensorFlowDataset('clinc_oos')

# load_brown_category_dataset()
# load_pos_tagged_dataset()
