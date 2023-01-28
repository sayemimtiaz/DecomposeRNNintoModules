import enum


class LayerType(enum.Enum):
    Dense = 1
    RNN = 2
    LSTM = 3,
    Activation = 4,
    Embedding = 5,
    TimeDistributed = 6,
    RepeatVector = 7,
    Flatten = 8,
    Input = 9,
    Dropout = 10,
    GRU = 11,
    NotRecognized = 0


class ActivationType(enum.Enum):
    Relu = 1
    Tanh = 2
    Sigmoid = 3,
    Softmax = 4,
    Linear = 5,
    NotRecognized = 0


def getLayerType(layer):
    if type(layer).__name__.lower() == 'dense':
        return LayerType.Dense
    if type(layer).__name__.lower() == 'simplernn':
        return LayerType.RNN
    if type(layer).__name__.lower() == 'lstm':
        return LayerType.LSTM
    if type(layer).__name__.lower() == 'gru':
        return LayerType.GRU
    if type(layer).__name__.lower() == 'activation':
        return LayerType.Activation
    if type(layer).__name__.lower() == 'embedding':
        return LayerType.Embedding
    if type(layer).__name__.lower() == 'timedistributed':
        return LayerType.TimeDistributed
    if type(layer).__name__.lower() == 'repeatvector':
        return LayerType.RepeatVector
    if type(layer).__name__.lower() == 'flatten':
        return LayerType.Flatten
    if type(layer).__name__.lower() == 'inputlayer':
        return LayerType.Input
    if type(layer).__name__.lower() == 'dropout':
        return LayerType.Dropout
    return LayerType.NotRecognized


def getActivationType(layer):
    if getLayerType(layer) == LayerType.TimeDistributed:
        layer = layer.submodules[0]

    if not hasattr(layer, 'activation'):
        return ActivationType.Linear
    if layer.activation is None:
        return ActivationType.Linear

    if layer.activation.__name__.lower() == 'relu':
        return ActivationType.Relu
    if layer.activation.__name__.lower() == 'tanh':
        return ActivationType.Tanh
    if layer.activation.__name__.lower() == 'sigmoid':
        return ActivationType.Sigmoid
    if layer.activation.__name__.lower() == 'softmax':
        return ActivationType.Softmax
    if layer.activation.__name__.lower() == 'linear':
        return ActivationType.Linear

    return ActivationType.NotRecognized
