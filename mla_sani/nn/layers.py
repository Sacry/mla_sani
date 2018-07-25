
from abc import ABC, abstractmethod
from copy import copy

import numpy as np

from ..activations import Identity

class Layer(ABC):
    """Super class for layers."""
    def __init__(self):
        """Declare variables which may be needed later."""
        # data (layer input of ith layer is just the layer output of i-1th layer)
        self.layer_input = None
        self.z = None
        self.layer_output = None

        # weights
        self.kernel = None
        self.kernel_optimizer = None

        # bias
        self.bias = None
        self.bias_optimizer = None

        # dimensionality of output
        self.units = None

        # activation function
        self.activations = None

        # unused
        self.built = False

    def build(self, input_shape, optimizer=None):
        self.built = True

    @abstractmethod
    def forward(self, X, train=True):
        pass

    @abstractmethod
    def backprop(self, grad):
        pass

class Input(Layer):
    """Input layer."""
    def __init__(self, units):
        super().__init__()
        self.units = units

    def forward(self, X, train=True):
        if train:
            self.layer_output = X
        return X

    def backprop(self, grad):
        raise NotImplementedError()

class Activation(Layer):
    """Activation layer."""
    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def build(self, input_shape, optimizer=None):
        self.units = input_shape
        super().build(input_shape)

    def forward(self, X, train=True):
        layer_output = self.activation(X)
        if train:
            self.layer_input = X
            self.layer_output = layer_output
        return layer_output

    def backprop(self, grad):
        return grad * self.activation.derivative(self.layer_input)

class Dummy(Activation):
    """Dummy layer, for test purpose."""
    def __init__(self):
        super().__init__(Identity())

class Dense(Layer):
    """Fully-connected nn layer."""
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape, optimizer=None):
        self.kernel = np.random.randn(input_shape, self.units)
        self.bias = np.random.randn(self.units)

        self.kernel_optimizer = copy(optimizer)
        self.bias_optimizer = copy(optimizer)

        super().build(input_shape)

    def forward(self, X, train=True):
        layer_output = X.dot(self.kernel) + self.bias
        if train:
            self.layer_input = X
            self.layer_output = layer_output
        return layer_output

    def backprop(self, grad):
        grad_kernel = self.layer_input.T.dot(grad) / self.layer_input.shape[0]
        grad_bias = grad.mean(axis=0)

        self.kernel = self.kernel_optimizer.get_updates(self.kernel, grad_kernel)
        self.bias = self.bias_optimizer.get_updates(self.bias, grad_bias)

        return grad.dot(self.kernel.T)
