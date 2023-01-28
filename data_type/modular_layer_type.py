from data_type.constants import Constants
from data_type.enums import LayerType, getLayerType, getActivationType
import numpy as np


class ModularLayer:
    type = LayerType.NotRecognized
    W = None
    U = None
    B = None
    num_node = None
    activation = None
    layer_serial = None
    first_layer = False
    last_layer = False
    number_samples = None
    number_features = None
    timestep = None
    hidden_state = None
    full_hidden_state = None
    cell_state = None
    lstm_node_val = None
    x_t = None
    return_sequence = True

    active_count = None
    inactive_count = None
    node_sum = None
    median_node_val = None

    unrolled = False
    DW = None
    DU = None
    DB = None

    name = None
    mask_zero = False

    next_layer = None

    def __init__(self, layer, timestep=None):
        self.name = layer.name
        self.type = getLayerType(layer)
        self.num_node = layer.output_shape[len(layer.output_shape) - 1]
        if type(self.num_node) is tuple:
            self.num_node = self.num_node[1]

        self.activation = getActivationType(layer)
        self.setInputShape(layer, timestep=timestep)

        if self.type != LayerType.Activation:

            if self.type == LayerType.RNN or self.type == LayerType.LSTM or self.type == LayerType.GRU:
                self.W, self.U, self.B = layer.get_weights()
                self.return_sequence = layer.return_sequences
                self.setHiddenState()
                self.initRnnModularWeights()

                numNode = self.num_node
                if self.type == LayerType.LSTM:
                    numNode = self.num_node * 4
                if self.type == LayerType.GRU:
                    numNode = self.num_node * 3

                if not Constants.UNROLL_RNN:
                    self.median_node_val = np.array([0.0] * numNode).reshape(1, numNode)
                else:
                    self.median_node_val = []
                    for ts in range(self.timestep):
                        self.median_node_val.append(np.array([0.0] * numNode).reshape(1, numNode))

                if self.type == LayerType.LSTM or self.type == LayerType.GRU:
                    self.lstm_node_val = []
                    for ts in range(self.timestep):
                        self.lstm_node_val.append(np.array([0] * numNode).reshape(1, numNode))

            if self.type == LayerType.Dense or self.type == LayerType.TimeDistributed:
                self.W, self.B = layer.get_weights()
                self.initRegularModularWeights()
                # self.node_sum = np.array([0.0] * self.num_node)

            if self.type == LayerType.Embedding:
                self.W = layer.get_weights()[0]
                self.initOpLayerModularWeights()
                self.mask_zero = layer.mask_zero

            if self.type == LayerType.RepeatVector:
                self.timestep = layer.n

    def setInputShape(self, layer, timestep=None):

        if self.type == LayerType.RNN or self.type == LayerType.LSTM or self.type == LayerType.GRU:
            lis = layer.input_shape
            if type(layer.input_shape) == list:
                lis = layer.input_shape[0]

            self.number_samples = lis[0]
            if len(lis) > 1:
                self.timestep = lis[1]
            if self.timestep is None:
                self.timestep = timestep

            if 'decoder' in self.name:
                self.timestep += 1

            if len(lis) > 2:
                self.number_features = lis[2]
        else:
            if type(layer.input_shape) == list:
                self.number_samples = layer.input_shape[0][0]
                self.number_features = layer.input_shape[0][1]
            else:
                self.number_samples = layer.input_shape[0]
                self.number_features = layer.input_shape[1]

    def initHiddenState(self):

        if not Constants.UNROLL_RNN and not self.return_sequence:
            self.hidden_state = np.array([0] * self.num_node).reshape(1, self.num_node)

        else:
            self.hidden_state = []

            for ts in range(self.timestep):
                self.hidden_state.append(np.array([0] * self.num_node).reshape(1, self.num_node))

    def setHiddenState(self, h_t_previous=None, which_timestep=None):

        if not Constants.UNROLL_RNN and not self.return_sequence:
            if h_t_previous is None:
                self.hidden_state = np.array([0] * self.num_node).reshape(1, self.num_node)

                self.full_hidden_state = []

                for ts in range(self.timestep):
                    self.full_hidden_state.append(np.array([0] * self.num_node).reshape(1, self.num_node))

            else:
                self.hidden_state = h_t_previous

                self.full_hidden_state[which_timestep] = h_t_previous
        else:
            if h_t_previous is None and self.timestep is not None:
                self.hidden_state = []

                for ts in range(self.timestep):
                    self.hidden_state.append(np.array([0] * self.num_node).reshape(1, self.num_node))

            else:
                self.hidden_state[which_timestep] = h_t_previous

    def getHiddenState(self, which_timestep=None):
        if type(self.hidden_state) is not list:
            return self.hidden_state
        # import random
        # self.hidden_state[which_timestep][:,0]=random.randint(3, 9)
        return self.hidden_state[which_timestep]

    def initRnnModularWeights(self):
        if not Constants.UNROLL_RNN:
            self.DW = np.zeros_like(self.W)
            self.DU = np.zeros_like(self.U)
            self.DB = np.zeros_like(self.B)
            self.active_count = np.array([0] * self.num_node)
            self.inactive_count = np.array([0] * self.num_node)
        elif self.timestep is not None:
            self.unrolled = True
            self.DW = []
            self.DU = []
            self.DB = []
            self.active_count = []
            self.inactive_count = []
            for ts in range(self.timestep):
                self.DW.append(np.zeros_like(self.W))
                self.DU.append(np.zeros_like(self.U))
                self.DB.append(np.zeros_like(self.B))
                self.active_count.append(np.array([0] * self.num_node))
                self.inactive_count.append(np.array([0] * self.num_node))

    def initRegularModularWeights(self):
        self.DW = np.zeros_like(self.W)
        self.DB = np.zeros_like(self.B)
        self.active_count = np.array([0] * self.num_node)
        self.inactive_count = np.array([0] * self.num_node)

    def initTimeDistributedWeights(self, timestep):
        self.timestep = timestep
        if not Constants.UNROLL_RNN:
            self.DW = np.zeros_like(self.W)
            self.DB = np.zeros_like(self.B)
            self.active_count = np.array([0] * self.num_node)
            self.inactive_count = np.array([0] * self.num_node)
        else:
            self.unrolled = True
            self.DW = []
            self.DB = []
            self.active_count = []
            self.inactive_count = []
            for ts in range(self.timestep):
                self.DW.append(np.zeros_like(self.W))
                self.DB.append(np.zeros_like(self.B))
                self.active_count.append(np.array([0] * self.num_node))
                self.inactive_count.append(np.array([0] * self.num_node))

    def initOpLayerModularWeights(self):
        self.DW = np.zeros_like(self.W)

    def copyHiddenState(self):
        if type(self.hidden_state) is not list:
            return self.hidden_state
        hs = []
        for i in range(len(self.hidden_state)):
            tmp = np.array([])
            for j in range(self.num_node):
                tmp = np.append(tmp, self.hidden_state[i][:, j])
            tmp = tmp.reshape(1, self.num_node)
            hs.append(tmp)
        return hs

    def copyLSTMNodeVal(self):
        if type(self.lstm_node_val) is not list:
            return self.lstm_node_val
        numNode = 0.0
        if self.type == LayerType.LSTM:
            numNode = self.num_node * 4
        if self.type == LayerType.GRU:
            numNode = self.num_node * 3
        hs = []
        for i in range(len(self.lstm_node_val)):
            tmp = np.array([])
            for j in range(numNode):
                tmp = np.append(tmp, self.lstm_node_val[i][:, j])
            tmp = tmp.reshape(1, numNode)
            hs.append(tmp)
        return hs
