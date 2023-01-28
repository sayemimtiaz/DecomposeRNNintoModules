import numpy as np


def sample_for_many_output(x, y, ts, target_class, num_label, positiveSample=True):
    samples_x = []
    samples_y = []

    count = 0
    for i in range(0, len(y)):
        if (positiveSample and y[i][ts] == target_class) or \
                (not positiveSample and y[i][ts] != target_class):
            samples_x.append(x[i])
            samples_y.append(y[i])
            count += 1
            if count >= num_label:
                break

    return np.asarray(samples_x), np.asarray(samples_y)


def sample_for_one_output(x, y, target_class, num_label, positiveSample=True):
    samples_x = []
    samples_y = []

    count = 0
    for i in range(0, len(y)):
        if (positiveSample and y[i] == target_class) or \
                (not positiveSample and y[i] != target_class):
            samples_x.append(x[i])
            samples_y.append(y[i])
            count += 1
            if count >= num_label:
                break

    return np.asarray(samples_x), np.asarray(samples_y)
