import os
import tensorflow as tf
from keras import layers

from relu_models.many_to_many_different.many_to_many_different_rolled.many_to_many_different_util import load_data, custom_standardization


def load_problem2_original_data(batch_size=64, sequence_length=20, vocab_size=30000, just_pairs=False, seed=19):
    target_languages = ['en', 'de', 'it']
    data_path = os.path.dirname(os.path.realpath(__file__))

    english_sentences = load_data(os.path.join(data_path, 'Tatoeba/en_et_train.txt'))
    german_sentences = load_data(os.path.join(data_path, 'Tatoeba/de_et_train.txt'))
    italian_sentences = load_data(os.path.join(data_path, 'Tatoeba/it_et_train.txt'))
    estonian_sentences = load_data(os.path.join(data_path, 'Tatoeba/et_et_train.txt'))
    text_pairs = []
    for english, german, italian, estonian in zip(english_sentences, german_sentences, italian_sentences,
                                                  estonian_sentences):
        english = "[starten] " + english + " [enden]"
        german = "[startde] " + german + " [endde]"
        italian = "[startit] " + italian + " [endit]"

        text_pairs.append((estonian, english))
        text_pairs.append((estonian, german))
        text_pairs.append((estonian, italian))

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

def load_problem2_solution1_data(batch_size=64, sequence_length=20, vocab_size=30000, just_pairs=False, seed=19):
    target_languages = ['en', 'it']
    data_path = os.path.dirname(os.path.realpath(__file__))

    english_sentences = load_data(os.path.join(data_path, 'Tatoeba/en_et_train.txt'))
    italian_sentences = load_data(os.path.join(data_path, 'Tatoeba/it_et_train.txt'))
    estonian_sentences = load_data(os.path.join(data_path, 'Tatoeba/et_et_train.txt'))
    text_pairs = []
    for english, italian, estonian in zip(english_sentences, italian_sentences,
                                                  estonian_sentences):
        english = "[starten] " + english + " [enden]"
        italian = "[startit] " + italian + " [endit]"

        text_pairs.append((estonian, english))
        text_pairs.append((estonian, italian))

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


def load_problem2_solution2_data(batch_size=64, sequence_length=20, vocab_size=30000, just_pairs=False, seed=19):
    target_languages = ['en']
    data_path = os.path.dirname(os.path.realpath(__file__))

    english_sentences = load_data(os.path.join(data_path, 'Tatoeba/en_et_train_s2.txt'))
    estonian_sentences = load_data(os.path.join(data_path, 'Tatoeba/et_et_train_s2.txt'))
    text_pairs = []
    for english, estonian in zip(english_sentences, estonian_sentences):
        english = "[starten] " + english + " [enden]"

        text_pairs.append((estonian, english))

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

# #
# train_ds, val_ds, test_pairs, \
#     source_vectorization, target_vectorization, target_languages = load_problem1_data(sequence_length=20,
#                                                                                 just_pairs=True, seed=277)
# print(len(train_ds), len(test_pairs))
# # # num_sample = -1
# sampled_data = sampleNegative(test_pairs, ['ua'],
#                               source_vectorization, target_vectorization, num_sample=3, vectorize=False, asList=True)
# #
# print(sampled_data)
