import numpy as np


def sample(x, y, nb_classes, label_per_sample):
    samples_x = []
    samples_y = []
    taken = np.array([0] * nb_classes)

    for i in range(0, len(y)):
        for j in range(0, nb_classes):
            if j in y[i] and taken[j] < label_per_sample:
                taken[j] += 1
                samples_x.append(x[i])
                samples_y.append(y[i])
                break
    return np.asarray(samples_x), np.asarray(samples_y)
