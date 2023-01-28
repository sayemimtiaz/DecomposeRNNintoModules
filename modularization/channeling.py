import numpy as np


def channel(layer, classes=range(0, 10), positiveIntent=0, negativeIntent=0):
    if positiveIntent == 0:
        negativeIntent = 1

    for i in range(layer.DW.shape[0]):
        temp = layer.DW[i, :]
        tempW2 = [0] * (len(temp) - 1)
        k = 0
        for j in classes:
            if j != positiveIntent:
                tempW2[k] = temp[j]
                k = k + 1
        temp[negativeIntent] = np.mean(tempW2)
        for j in classes:
            if j != positiveIntent and j != negativeIntent:
                temp[j] = 0
        layer.DW[i, :] = temp

    temp = [0] * (layer.num_node - 1)
    k = 0
    for i in range(layer.num_node):
        if i != positiveIntent:
            temp[k] = layer.DB[i]
            k = k + 1
    layer.DB[negativeIntent] = np.mean(temp)
    for i in range(layer.num_node):
        if i != positiveIntent and i != negativeIntent:
            layer.DB[i] = 0


def channel_multi_output(layer, classes=range(0, 10), positiveIntent=1, negativeIntent=1, fixedIntent=0):
    if positiveIntent==0:
        return

    if positiveIntent == 1:
        negativeIntent = 2

    for i in range(layer.DW.shape[0]):
        temp = layer.DW[i, :]
        tempW2 = [0] * (len(temp) - 1)
        k = 0
        for j in classes:
            if j != positiveIntent and j != fixedIntent:
                tempW2[k] = temp[j]
                k = k + 1
        temp[negativeIntent] = np.mean(tempW2)
        for j in classes:
            if j != positiveIntent and j != negativeIntent and j != fixedIntent:
                temp[j] = 0
        layer.DW[i, :] = temp

    temp = [0] * (layer.num_node - 1)
    k = 0
    for i in range(layer.num_node):
        if i != positiveIntent and i != fixedIntent:
            temp[k] = layer.DB[i]
            k = k + 1
    layer.DB[negativeIntent] = np.mean(temp)
    for i in range(layer.num_node):
        if i != positiveIntent and i != negativeIntent and i != fixedIntent:
            layer.DB[i] = 0
