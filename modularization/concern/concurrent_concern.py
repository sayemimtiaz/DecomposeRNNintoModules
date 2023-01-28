from data_type.constants import Constants
from data_type.enums import LayerType, ActivationType
import numpy as np

from modularization.concern.util import updateNodeConcernForRefLayer, updateNodeConcernForOutputLayerForRefLayer
from util.common import softmax, sigmoid, initModularLayers


class ConcurrentConcernIdentification:
    conerns = None
    referenceLayers = None

    def __init__(self, concerns):
        self.conerns = concerns

    def updateConcern(self, current_label, current_timestep):
        concernLayers = self.conerns[current_label]

        for layerNo, concernLayer in enumerate(concernLayers):

            if concernLayer.type == LayerType.RNN:
                updateNodeConcernForRefLayer(concernLayer, self.referenceLayers[layerNo], current_timestep,
                                             unrolled=concernLayer.unrolled)

            elif concernLayer.type == LayerType.TimeDistributed:

                if concernLayer.last_layer:
                    updateNodeConcernForOutputLayerForRefLayer(concernLayer, self.referenceLayers[layerNo],
                                                               current_timestep, unrolled=concernLayer.unrolled)
                else:
                    updateNodeConcernForRefLayer(concernLayer, self.referenceLayers[layerNo], current_timestep,
                                                 unrolled=concernLayer.unrolled)
    @staticmethod
    def initConcern(refModelLayers):
        concern=initModularLayers(refModelLayers)

        for layerNo, concernLayer in enumerate(concern):
            if concernLayer.type == LayerType.Embedding\
                    or concernLayer.type == LayerType.RepeatVector\
                    or concernLayer.type == LayerType.Flatten:
                continue

            if not Constants.UNROLL_RNN and concernLayer.type == LayerType.RNN:
                concern[layerNo].unrolled = False
                concern[layerNo].DW = np.zeros_like(concern[layerNo].W)
                concern[layerNo].DU = np.zeros_like(concern[layerNo].U)
                concern[layerNo].DB = np.zeros_like(concern[layerNo].B)
                concern[layerNo].active_count = np.array([0] * concern[layerNo].num_node)
                concern[layerNo].inactive_count = np.array([0] * concern[layerNo].num_node)

            if concernLayer.type == LayerType.TimeDistributed:
                concern[layerNo].unrolled=False
                concern[layerNo].DW = np.zeros_like(concern[layerNo].W)
                concern[layerNo].DB = np.zeros_like(concern[layerNo].B)
                concern[layerNo].active_count = np.array([0] * concern[layerNo].num_node)
                concern[layerNo].inactive_count = np.array([0] * concern[layerNo].num_node)

        return concern

    def mergeAndGetPositiveAndNegConcern(self, exclude_label, all_label, refModelLayers):
        positiveConcern = self.conerns[exclude_label]
        negativeConcern = self.initConcern(refModelLayers)

        for l in all_label:
            if l==exclude_label:
                continue

            for layerNo, concernLayer in enumerate(self.conerns[l]):
                if concernLayer.type == LayerType.Embedding or concernLayer.last_layer\
                        or concernLayer.type == LayerType.RepeatVector\
                        or concernLayer.type == LayerType.Flatten:
                    continue

                if (not Constants.UNROLL_RNN and concernLayer.type==LayerType.RNN) or (concernLayer.type==LayerType.TimeDistributed):
                    negativeConcern[layerNo].active_count+=concernLayer.active_count
                    negativeConcern[layerNo].inactive_count += concernLayer.inactive_count

                else:
                    for ts in range(len(concernLayer.hidden_state)):
                        negativeConcern[layerNo].active_count[ts] += concernLayer.active_count[ts]
                        negativeConcern[layerNo].inactive_count[ts] += concernLayer.inactive_count[ts]

        return positiveConcern, negativeConcern

    def feedback_loop(self, layer, x, apply_activation=True):

        for ts in range(layer.timestep):
            x_t = x[ts].reshape(1, layer.number_features)

            if ts==0:
                x_t = self.propagateThroughRNN(layer, x_t,
                                               layer.getHiddenState(ts), apply_activation=apply_activation)
            else:
                x_t = self.propagateThroughRNN(layer, x_t,
                                           layer.getHiddenState(ts-1), apply_activation=apply_activation)
            layer.setHiddenState(x_t, ts)

        return layer.hidden_state

    def propagateThroughLayer(self, layer, x_t=None, apply_activation=True):
        if layer.type == LayerType.RNN:
            layer.hidden_state = self.feedback_loop(layer, x_t, apply_activation=apply_activation)

        elif layer.type == LayerType.Embedding:

            return self.embeddingLookup(layer, x_t=x_t)
        elif layer.type == LayerType.RepeatVector:
            layer.hidden_state = self.repeatVector(layer, x_t)

        elif layer.type == LayerType.Flatten:
            layer.hidden_state = self._flatten(x_t)
        elif layer.type == LayerType.TimeDistributed:

            layer.hidden_state = self.propagateThroughTimeDistributed(layer, x_t=x_t,
                                                                      apply_activation=apply_activation)
        if layer.type == LayerType.RNN and not layer.return_sequence:
            return layer.getHiddenState(layer.timestep-1)
        return layer.hidden_state

    def _flatten(self, x):
        return x.flatten()

    def repeatVector(self, layer, a):
        c = []
        for x in range(layer.timestep):
            c.append(a)
        c = np.asarray(c)
        return c
    def embeddingLookup(self, layer, x_t=None):
        embed = []
        for _x in x_t:
            embed.append(layer.W[_x])

        return np.asarray(embed)

    def propagateThroughTimeDistributed(self, layer, x_t=None, apply_activation=True):

        for ts in range(len(x_t)):
            h_t = self.propagateThroughDense(layer, x_t[ts], apply_activation=apply_activation)
            layer.setHiddenState(h_t, ts)
            if layer.last_layer:
                outLabel = h_t.argmax()
                # if outLabel>1:
                #     print(outLabel)
                self.updateConcern(outLabel, ts)

        return layer.hidden_state

    def propagateThroughRNN(self, layer, x_t=None, h_t_previous=None, apply_activation=True):
        x_t = (x_t.dot(layer.W) +
               h_t_previous.dot(layer.U) +
               layer.B)

        return self.propagateThroughActivation(layer, x_t, apply_activation)

    def propagateThroughDense(self, layer, x_t=None, apply_activation=True):

        x_t = (x_t.dot(layer.W) + layer.B)

        return self.propagateThroughActivation(layer, x_t, apply_activation)

    def propagateThroughActivation(self, layer, x_t, apply_activation=True, ):
        if not apply_activation or layer.activation == ActivationType.Linear:
            return x_t

        if ActivationType.Softmax == layer.activation:
            x_t = x_t.reshape(layer.num_node)
            x_t = softmax(x_t)
        elif ActivationType.Relu == layer.activation:
            x_t[x_t < 0] = 0
        elif ActivationType.Tanh == layer.activation:
            x_t = np.tanh(x_t)
        elif ActivationType.Sigmoid == layer.activation:
            x_t = sigmoid(x_t)

        return x_t
