from modularization.concern.concern_identification_encoder_decoder import ConcernIdentificationEnDe
from modularization.concern.util import *
from util.common import *
from data_type.enums import LayerType

from util.layer_propagator import LayerPropagator


class LayerPropagatorEnDe(LayerPropagator):
    encoderLoop = True

    def feedback_loop_rnn(self, layer, x, apply_activation=True, mask=None, initial_state=None):
        previous_state = initial_state
        if initial_state is None:
            previous_state = layer.getHiddenState(0)
        for ts in range(layer.timestep):
            if mask is not None and mask[ts] == False:
                layer.setHiddenState(previous_state, ts)
            else:

                x_t = x[ts].reshape(1, layer.number_features)

                x_t = self.propagateThroughRNN(layer, x_t, previous_state, apply_activation=apply_activation)

                previous_state = x_t

                layer.setHiddenState(x_t, ts)

        return layer.hidden_state

    def feedback_loop_lstm(self, layer, x, mask=None, initial_state=None):
        num_node = layer.num_node
        if initial_state is None:
            c_t = np.array([0] * num_node).reshape(1, num_node)
            h_t = np.array([0] * num_node).reshape(1, num_node)
        else:
            h_t = initial_state[0]
            c_t = initial_state[1]

        for ts in range(layer.timestep):
            if mask is not None and mask[ts] == False:
                layer.setHiddenState(h_t, ts)
            else:
                x_t = x[ts].reshape(1, layer.number_features)

                h_t, c_t = self.propagateThroughLSTM(layer, x_t, h_t_previous=h_t,
                                                                  c_t_previous=c_t)
                layer.setHiddenState(h_t, ts)

        return layer.hidden_state, c_t

    def feedback_loop_gru(self, layer, x, initial_state=None, mask=None):
        num_node = layer.num_node
        if initial_state is None:
            h_t = np.array([0] * num_node).reshape(1, num_node)
        else:
            h_t = initial_state

        for ts in range(layer.timestep):
            if mask is not None and mask[ts] == False:
                layer.setHiddenState(h_t, ts)
            else:
                x_t = x[ts].reshape(1, layer.number_features)

                h_t = self.propagateThroughGRU(layer, x_t, h_t_previous=h_t)
                layer.setHiddenState(h_t, ts)

        return layer.hidden_state

    def compute_mask(self, layer, x):
        if not layer.mask_zero:
            return None
        return x != 0

    def get_encoder_output(self, layers, x, apply_activation=True):
        self.encoderLoop = True
        layers = ConcernIdentificationEnDe.get_encoder_layers(layers)
        mask = None
        c_t = None
        for layerNo, _layer in enumerate(layers):
            if _layer.type == LayerType.Embedding:
                x, mask = self.propagateThroughLayer(_layer, x,
                                                     apply_activation=apply_activation)
            elif _layer.type == LayerType.RNN or _layer.type == LayerType.GRU:
                x = self.propagateThroughLayer(_layer, x, apply_activation=apply_activation, mask=mask)
            elif _layer.type == LayerType.LSTM:
                x, c_t = self.propagateThroughLayer(_layer, x, apply_activation=apply_activation, mask=mask)
            else:
                x = self.propagateThroughLayer(_layer, x, apply_activation=apply_activation)
        if c_t is not None:
            return x, c_t
        return x

    def get_decoder_output(self, layers, x, encoder_state, apply_activation=True):
        self.encoderLoop = False
        layers = ConcernIdentificationEnDe.get_decoder_layers(layers)
        mask = None
        for layerNo, _layer in enumerate(layers):
            if _layer.type == LayerType.Embedding:
                x, mask = self.propagateThroughLayer(_layer, x,
                                                     apply_activation=apply_activation)
            elif _layer.type == LayerType.RNN or _layer.type == LayerType.GRU:
                x = self.propagateThroughLayer(_layer, x, apply_activation=apply_activation,
                                               mask=mask, initial_state=encoder_state)
                encoder_state = None
            elif _layer.type == LayerType.LSTM:
                x, c_t = self.propagateThroughLayer(_layer, x, apply_activation=apply_activation,
                                                    mask=mask, initial_state=encoder_state)
                encoder_state = None
            else:
                x = self.propagateThroughLayer(_layer, x, apply_activation=apply_activation)
        return x

    def propagateThroughEncoderDecoder(self, layers, x, y, apply_activation=True):
        encoder_state = self.get_encoder_output(layers, x, apply_activation=apply_activation)
        decoder_state = self.get_decoder_output(layers, y, encoder_state, apply_activation=apply_activation)
        return self.propagateThroughLayer(ConcernIdentificationEnDe.get_output_layer(layers), decoder_state,
                                          apply_activation=True)

    def propagateThroughLayer(self, layer, x_t=None, apply_activation=True, mask=None,
                              initial_state=None):
        c_t = None
        if layer.type == LayerType.RNN:
            layer.hidden_state = self.feedback_loop_rnn(layer, x_t,
                                                        apply_activation=apply_activation,
                                                        mask=mask, initial_state=initial_state)

        elif layer.type == LayerType.LSTM:
            layer.hidden_state, c_t = self.feedback_loop_lstm(layer, x_t, mask=mask, initial_state=initial_state)

        elif layer.type == LayerType.GRU:
            layer.hidden_state = self.feedback_loop_gru(layer, x_t, mask=mask, initial_state=initial_state)

        elif layer.type == LayerType.Dense:

            layer.hidden_state = self.propagateThroughDense(layer, x_t=x_t,
                                                            apply_activation=apply_activation)

            if layer.last_layer:
                updateNodeConcernForOutputLayer(layer)
            else:
                updateNodeConcernForRegular(layer)

        elif layer.type == LayerType.Embedding:

            return self.embeddingLookup(layer, x_t=x_t)

        elif layer.type == LayerType.TimeDistributed:

            layer.hidden_state = self.propagateThroughTimeDistributed(layer, x_t=x_t,
                                                                      apply_activation=apply_activation)
        elif layer.type == LayerType.RepeatVector:
            layer.hidden_state = self.repeatVector(layer, x_t)

        elif layer.type == LayerType.Flatten:
            layer.hidden_state = self._flatten(x_t)

        if (layer.type == LayerType.RNN or layer.type == LayerType.GRU) and not layer.return_sequence:
            return layer.getHiddenState(layer.timestep - 1)

        if layer.type == LayerType.LSTM:
            if not layer.return_sequence:
                return layer.getHiddenState(layer.timestep - 1), c_t
            else:
                return layer.hidden_state, c_t

        if layer.type == LayerType.Dropout:
            return x_t

        return layer.hidden_state

    def embeddingLookup(self, layer, x_t=None):
        embed = []
        for _x in x_t:
            embed.append(layer.W[_x])

        embed = np.asarray(embed)

        mask = self.compute_mask(layer, x_t)

        return embed, mask

