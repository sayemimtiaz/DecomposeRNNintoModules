import os
import tensorflow as tf
import string
import re
from keras import layers


def load_data(path):
    """
    Load dataset
    """
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data.split('\n')


def custom_standardization(input_string):
    strip_chars = string.punctuation + "¿"
    strip_chars = strip_chars.replace("[", "")
    strip_chars = strip_chars.replace("]", "")

    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(
        lowercase, f"[{re.escape(strip_chars)}]", "")


def load_tatoeba(batch_size=64, sequence_length=20, vocab_size=30000, just_pairs=False, seed=19):
    target_languages = ['fr', 'de', 'it']
    data_path = os.path.dirname(os.path.realpath(__file__))

    english_sentences = load_data(os.path.join(data_path, 'Tatoeba/en_train.txt'))
    french_sentences = load_data(os.path.join(data_path, 'Tatoeba/fr_train.txt'))
    german_sentences = load_data(os.path.join(data_path, 'Tatoeba/de_train.txt'))
    italian_sentences = load_data(os.path.join(data_path, 'Tatoeba/it_train.txt'))

    text_pairs = []
    for english, french, german, italian in zip(english_sentences, french_sentences, german_sentences,
                                                italian_sentences):
        french = "[startfr] " + french + " [endfr]"
        text_pairs.append((english, french))
        german = "[startde] " + german + " [endde]"
        text_pairs.append((english, german))
        italian = "[startit] " + italian + " [endit]"
        text_pairs.append((english, italian))

    import random

    print(random.choice(text_pairs))

    random.seed(seed)
    random.shuffle(text_pairs)
    num_val_samples = int(0.15 * len(text_pairs))
    num_train_samples = len(text_pairs) - 2 * num_val_samples
    train_pairs = text_pairs[:num_train_samples]
    val_pairs = text_pairs[num_train_samples:num_train_samples + num_val_samples]
    test_pairs = text_pairs[num_train_samples + num_val_samples:]

    source_vectorization = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=sequence_length,
    )
    target_vectorization = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=sequence_length + 1,
        standardize=custom_standardization,
    )
    train_english_texts = [pair[0] for pair in train_pairs]
    train_spanish_texts = [pair[1] for pair in train_pairs]
    source_vectorization.adapt(train_english_texts)
    target_vectorization.adapt(train_spanish_texts)

    def format_dataset(eng, spa):
        eng = source_vectorization(eng)
        spa = target_vectorization(spa)
        return ({
                    "source": eng,
                    "target": spa[:, :-1],
                }, spa[:, 1:])

    def make_dataset(pairs):
        eng_texts, spa_texts = zip(*pairs)
        eng_texts = list(eng_texts)
        spa_texts = list(spa_texts)
        dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(format_dataset, num_parallel_calls=4)
        return dataset.shuffle(2048).prefetch(16).cache()

    train_ds = make_dataset(train_pairs)
    val_ds = make_dataset(val_pairs)

    if just_pairs:
        return train_pairs, val_pairs, test_pairs, source_vectorization, target_vectorization, \
               target_languages
    return train_ds, val_ds, test_pairs, source_vectorization, target_vectorization


def samplePositive(data, langauge, source_vectorization, target_vectorization, num_sample=500,
                   vectorize=True, seed=19):
    # import random
    # random.seed(seed)
    # random.shuffle(data)

    sampled_data = []
    takenCounter = 0
    for source, target in data:
        if num_sample != -1 and takenCounter >= num_sample:
            break
        if target.startswith('[start' + langauge + ']'):
            if vectorize:
                source = source_vectorization(source).numpy()
                target = target_vectorization(target).numpy()
                if len(source) == 0 or len(target) == 0:
                    continue
            sampled_data.append((source, target))
            takenCounter += 1
    return sampled_data


def sampleNegative(data, langauges, source_vectorization, target_vectorization, num_sample=100,
                   vectorize=True, seed=19, asList=False):
    sampled_data = {}
    result_list = []
    for l in langauges:
        sampled_data[l] = samplePositive(data, l, source_vectorization,
                                         target_vectorization, num_sample=num_sample,
                                         vectorize=vectorize, seed=19)
        if asList:
            for d in sampled_data[l]:
                result_list.append(d)

    if asList:
        import random
        random.seed(seed)
        random.shuffle(result_list)
        return result_list

    return sampled_data


def load_glob_embedding(num_words, embed_size=100, word_index=None):
    from numpy import asarray
    from numpy import zeros

    embeddings_dictionary = dict()

    glove_file = open('/Users/sayem/tensorflow_datasets/Globe/glove.6B.' + str(embed_size) + 'd.txt', encoding="utf8")

    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions
    glove_file.close()

    embedding_matrix = zeros((num_words, embed_size))

    spa_vocab = word_index.get_vocabulary()

    for index, word in enumerate(spa_vocab):
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

    return embedding_matrix

# load_tatoeba()
#
# train_ds, val_ds, test_pairs, \
#     source_vectorization, target_vectorization, target_languages = load_tatoeba(sequence_length=20,
#                                                                                 just_pairs=True, seed=277)
#
# # # num_sample = -1
# sampled_data = sampleNegative(test_pairs, ['it'],
#                               source_vectorization, target_vectorization, num_sample=3, vectorize=False, asList=True)
# #
# print(sampled_data)
# import pickle
#
# with open('Tatoeba/test_data_it', 'wb') as fp:
#     pickle.dump(sampled_data, fp)
#
# with open ('Tatoeba/test_data_fr', 'rb') as fp:
#     itemlist = pickle.load(fp)
#     print(len(itemlist))
