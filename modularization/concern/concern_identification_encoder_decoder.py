from modularization.concern.util import *
from util.common import *
from data_type.enums import LayerType, ActivationType


class ConcernIdentificationEnDe:
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

            # if layer.unrolled or layer.return_sequence:
            #     updateNodeConcernForRnnAtTimestep(layer, ts)

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

                h_t, c_t, node_status = self.propagateThroughLSTM(layer, x_t, h_t_previous=h_t,
                                                                  c_t_previous=c_t)
                layer.setHiddenState(h_t, ts)
                layer.lstm_node_val[ts] = node_status

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

                h_t, node_status = self.propagateThroughGRU(layer, x_t, h_t_previous=h_t)
                layer.setHiddenState(h_t, ts)
                layer.lstm_node_val[ts] = node_status

        return layer.hidden_state

    def compute_mask(self, layer, x):
        if not layer.mask_zero:
            return None
        return x != 0

    @staticmethod
    def get_output_layer(layers):
        for layer in layers:
            if layer.type == LayerType.Input or layer.type == LayerType.Dropout:
                continue
            if 'output' in layer.name.lower():
                return layer
        return layers[-1]

    @staticmethod
    def get_encoder_layers(layers):
        out = []
        prevLayer = None
        for layer in layers:
            if layer.type == LayerType.Input or layer.type == LayerType.Dropout:
                continue
            if 'encoder' in layer.name.lower():
                out.append(layer)
                if prevLayer is not None:
                    prevLayer.next_layer = layer
                prevLayer = layer
        return out

    @staticmethod
    def is_decoder_layer(layer):
        if 'decoder' in layer.name.lower():
            return True
        return False

    @staticmethod
    def get_decoder_layers(layers):
        out = []
        prevLayer = None
        for layer in layers:
            if layer.type == LayerType.Input or layer.type == LayerType.Dropout:
                continue
            if 'decoder' in layer.name.lower():
                out.append(layer)
                if prevLayer is not None:
                    prevLayer.next_layer = layer
                prevLayer = layer
        return out

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

    def propagateThroughEncoderDecoder(self, layers, x, y):
        encoder_state = self.get_encoder_output(layers, x)
        decoder_state = self.get_decoder_output(layers, y, encoder_state)
        return self.propagateThroughLayer(ConcernIdentificationEnDe.get_output_layer(layers), decoder_state,
                                          apply_activation=True)

    def propagateThroughLayer(self, layer, x_t=None, apply_activation=True, mask=None, initial_state=None):
        c_t = None
        if layer.type == LayerType.RNN:
            layer.hidden_state = self.feedback_loop_rnn(layer, x_t,
                                                        apply_activation=apply_activation,
                                                        mask=mask, initial_state=initial_state)
            # if not layer.unrolled and not layer.return_sequence:
            #     updateNodeConcernForRnn(layer)

        elif layer.type == LayerType.LSTM:
            layer.hidden_state, c_t = self.feedback_loop_lstm(layer, x_t, mask=mask, initial_state=initial_state)

        elif layer.type == LayerType.GRU:
            layer.hidden_state = self.feedback_loop_gru(layer, x_t, mask=mask, initial_state=initial_state)

        elif layer.type == LayerType.Dense:

            layer.hidden_state = self.propagateThroughDense(layer, x_t=x_t,
                                                            apply_activation=apply_activation)

            # if layer.last_layer:
            #     updateNodeConcernForOutputLayer(layer)
            # else:
            #     updateNodeConcernForRegular(layer)

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

        embed = np.asarray(embed)

        mask = self.compute_mask(layer, x_t)

        return embed, mask

    def propagateThroughTimeDistributed(self, layer, x_t=None, apply_activation=True):
        output = []
        for ts in range(len(x_t)):
            h_t = self.propagateThroughDense(layer, x_t[ts], apply_activation=apply_activation)
            output.append(h_t)

        return np.asarray(output)

    def propagateThroughRNN(self, layer, x_t=None, h_t_previous=None, apply_activation=True):
        x_t = (x_t.dot(layer.W) +
               h_t_previous.dot(layer.U) +
               layer.B)

        return self.propagateThroughActivation(layer, x_t, apply_activation)

    def propagateThroughLSTM(self, layer, x_t=None, h_t_previous=None, c_t_previous=None):

        warr, uarr, barr = layer.W, layer.U, layer.B
        s_t = (x_t.dot(warr) + h_t_previous.dot(uarr) + barr)
        hunit = layer.num_node
        i = sigmoid(s_t[:, :hunit])
        f = sigmoid(s_t[:, 1 * hunit:2 * hunit])
        _c = np.tanh(s_t[:, 2 * hunit:3 * hunit])
        o = sigmoid(s_t[:, 3 * hunit:])
        c_t = i * _c + f * c_t_previous
        h_t = o * np.tanh(c_t)
        return h_t, c_t, np.concatenate([i, f, _c, o], axis=1)

    def propagateThroughGRU(self, layer, x_t=None, h_t_previous=None):

        warr, uarr, barr = layer.W, layer.U, layer.B
        hunit = layer.num_node

        z = sigmoid(x_t.dot(warr[:, :hunit:]) + h_t_previous.dot(uarr[:, :hunit]) + barr[:hunit])
        r = sigmoid(x_t.dot(warr[:, 1 * hunit:2 * hunit]) + h_t_previous.dot(uarr[:, 1 * hunit:2 * hunit]) + barr[
                                                                                                             1 * hunit:2 * hunit])
        h = tanh(x_t.dot(warr[:, 2 * hunit:]) + (r * h_t_previous.dot(uarr[:, 2 * hunit:])) + barr[2 * hunit:])

        h_t = z * h_t_previous + (1 - z) * h
        return h_t, np.concatenate([z, r, h], axis=1)

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
