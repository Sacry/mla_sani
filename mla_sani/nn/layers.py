
from abc import ABC, abstractmethod
from copy import copy

import numpy as np

from ..activations import *

class Layer(ABC):
    """Super class for layers."""
    def __init__(self):
        """Declare variables which may be used later."""
        # data
        self.layer_input = None
        # self.layer_output = None

        # dimensionality (excludes n_samples)
        self.input_shape = None
        self.output_shape = None

        # weights and bias
        self.kernel = None
        self.bias = None

        # optimizer
        self.kernel_optimizer = None
        self.bias_optimizer = None

        # unused
        self.built = False

    def build(self, input_shape, optimizer=None):
        self.input_shape = input_shape

        self.kernel_optimizer = copy(optimizer)
        self.bias_optimizer = copy(optimizer)

        self.built = True

    @abstractmethod
    def forward(self, X, training=True):
        pass

    @abstractmethod
    def backprop(self, grad):
        pass


class Input(Layer):
    """Input layer."""
    def __init__(self, input_shape):
        super().__init__()
        self.output_shape = input_shape

    def forward(self, X, training=True):
        if training:
            self.layer_input = X
        return X

    def backprop(self, grad):
        raise NotImplementedError()


class Dense(Layer):
    """Fully-connected nn layer."""
    def __init__(self, units):
        super().__init__()
        self.output_shape = units

    def build(self, input_shape, optimizer=None):
        # initialize weights
        self.kernel = np.random.randn(input_shape, self.output_shape) * np.sqrt(2 / self.output_shape)
        self.bias = np.zeros(self.output_shape)

        super().build(input_shape, optimizer)

    def forward(self, X, training=True):
        if training:
            self.layer_input = X

        return X.dot(self.kernel) + self.bias

    def backprop(self, grad):
        grad_kernel = self.layer_input.T.dot(grad) / self.layer_input.shape[0]
        grad_bias = grad.mean(axis=0)

        # grad for previous layer, calculate before update weights
        grad = grad.dot(self.kernel.T)

        # update weights
        self.kernel = self.kernel_optimizer.get_updates(self.kernel, grad_kernel)
        self.bias = self.bias_optimizer.get_updates(self.bias, grad_bias)

        return grad


ACTIVATIONS = {
    'identity': Identity(),
    'sigmoid': Sigmoid(),
    'softmax': Softmax(),
    'tanh': TanH(),
    'relu': ReLU(),
    'elu': ELU(),
    'softplus': SoftPlus()
}

class Activation(Layer):
    """Activation layer."""
    def __init__(self, activation):
        super().__init__()

        if isinstance(activation, str):
            self.activation = ACTIVATIONS[activation]
        else:
            self.activation = activation

    def build(self, input_shape, optimizer=None):
        self.output_shape = input_shape

        super().build(input_shape, optimizer)

    def forward(self, X, training=True):
        if training:
            self.layer_input = X

        return self.activation(X)

    def backprop(self, grad):
        return grad * self.activation.derivative(self.layer_input)


class Dummy(Activation):
    """Dummy layer, for test purpose."""
    def __init__(self):
        super().__init__(Identity())


class Conv2D(Layer):
    """2D convolutional layer.

    The process is shown as below.
        + Dense layer
            - forward: calculate wx + b
            - backprop: calculate gradients -> update weights
        + Conv2D layer
            - forward: im2col -> calculate wx + b
            - backprop: calculate gradients -> update weights -> col2im

    It's almost the same as Dense layer except the im2col/col2im transformation.  However, the shape
    thing is fairly complex.

    Args:
        filters (int): Number of filters.
        kernel_size (tuple[int, int]): Height and width of the convolution window.
        strides (tuple[int, int]): Convolution stride along the height and width.
        padding (str): Padding rule, 'same' or 'valid'.

    Note:
        * This implements assumes `channel_last` image format. Shape of each data is shown as below:
            input: (n_samples, n_rows, n_cols, n_channels)
            col: (n_channels * kernel_size[0] * kernel_size[1], n_new_rows * n_new_cols * n_samples)
            kernel: (n_filters, n_channels, kernel_size[0], kernel_size[1])
            bias: (n_filter,)
            output: (n_samples, n_new_rows, n_new_cols, n_filters)

    Reference:
        http://cs231n.github.io/convolutional-networks/
    """
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid'):
        super().__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

        self.col = None

    def build(self, input_shape, optimizer=None):
        n_rows, n_cols, n_channels = input_shape

        # initialize weights
        self.kernel = np.random.randn(self.filters, n_channels, self.kernel_size[0], self.kernel_size[1])
        self.kernel *= np.sqrt(2 / self.kernel.size)
        self.bias = np.zeros(self.filters)

        self.output_shape = self.compute_output_shape(input_shape)
        super().build(input_shape, optimizer)

    def forward(self, X, training=True):
        if training:
            self.layer_input = X

        # (n_channels * kernel_size[0] * kernel_size[1], n_new_rows * n_new_cols * n_samples)
        self.col = im2col(X, self.kernel_size, self.strides, self.padding)

        # perform wx + b
        #
        # (n_filters, n_channels * kernel_size[0] * kernel_size[1])
        #   dot (n_channels * kernel_size[0] * kernel_size[1], n_new_rows * n_new_cols * n_samples)
        # = (n_filters, n_new_rows * n_new_cols * n_samples)
        layer_output = self.kernel.reshape(self.filters, -1).dot(self.col) + self.bias.reshape(-1, 1)

        # reshape to (n_samples, n_new_rows, n_new_cols, n_filters)
        n_new_rows, n_new_cols, _ = self.output_shape
        layer_output = layer_output.reshape(self.filters, n_new_rows, n_new_cols, -1)
        layer_output = layer_output.transpose(3, 1, 2, 0)

        return layer_output

    def backprop(self, grad):
        # (n_samples, n_new_rows, n_new_cols, n_filters)
        #   -> (n_filters, n_new_rows, n_new_cols, n_samples)
        #   -> (n_filters, n_new_rows * n_new_cols * n_samples)
        grad = grad.transpose(3, 1, 2, 0).reshape(self.filters, -1)

        # (n_filters, n_new_rows * n_new_cols * n_samples)
        #   dot (n_new_rows * n_new_cols * n_samples, n_channels * kernel_size[0] * kernel_size[1])
        # = (n_filters, n_channels * kernel_size[0] * kernel_size[1])
        grad_kernel = grad.dot(self.col.T) / self.layer_input.shape[0]
        grad_kernel = grad_kernel.reshape(self.filters, -1, self.kernel_size[0], self.kernel_size[1])
        grad_bias = grad.mean(axis=1)

        # (n_channels * kernel_size[0] * kernel_size[1], n_filters)
        #   dot (n_filters, n_new_rows * n_new_cols * n_samples)
        # = (n_channels * kernel_size[0] * kernel_size[1], n_filters, n_new_rows * n_new_cols * n_samples)
        grad = self.kernel.T.dot(grad)

        # update weights
        self.kernel = self.kernel_optimizer.get_updates(self.kernel, grad_kernel)
        self.bias = self.bias_optimizer.get_updates(self.bias, grad_bias)

        return col2im(grad, self.layer_input.shape, self.kernel_size, self.strides, self.padding)

    def compute_output_shape(self, input_shape):
        n_rows, n_cols, n_channels = input_shape

        # n_new_rows = self._compute_new_length(n_rows, self.kernel_size[0], self.strides[0], self.padding)
        # n_new_cols = self._compute_new_length(n_cols, self.kernel_size[1], self.strides[1], self.padding)

        pr1, pr2, pc1, pc2 = get_pad_width(self.kernel_size, self.padding)
        n_new_rows = (n_rows - self.kernel_size[0] + pr1 + pr2) // self.strides[0] + 1
        n_new_cols = (n_cols - self.kernel_size[1] + pc1 + pc2) // self.strides[1] + 1

        return (n_new_rows, n_new_cols, self.filters)

    def _compute_new_length(self, length, kernel_length, stride, padding):
        if padding == 'same':
            length = length + kernel_length - 1

        # new length is the length of `range(start, stop, step)`
        start, stop, step = 0, length - kernel_length + 1, stride
        return (stop - start) // step + ((stop - start) % step != 0)

def im2col(image, kernel_size, strides, padding='same'):
    """im2col.

    Args:
        image (np.ndarray): shape = (n_samples, n_rows, n_cols, n_channels)
        kernel_size (tuple[int, int]): Height and width of the convolution window.
        stride (tuple[int, int]): Convolution stride along the height and width.
        padding (str): Padding rule, 'same' or 'valid'.

    Returns:
        np.ndarray: shape = (n_channels * kernel_size[0] * kernel_size[1], n_new_rows * n_new_cols * n_samples)
    """
    n_samples, n_rows, n_cols, n_channels = image.shape

    # pad image
    pr1, pr2, pc1, pc2 = get_pad_width(kernel_size, padding)
    image = np.pad(image,
                   pad_width=((0, 0), (pr1, pr2), (pc1, pc2), (0, 0)),
                   mode='constant',
                   constant_values=0)

    fancy_row, fancy_col = get_fancy_indices((image.shape[1], image.shape[2]), kernel_size, strides)

    # (n_samples, n_rows, n_cols, n_channels)
    #   -> (n_samples, n_new_rows, n_new_cols, kernel_size[0], kernel_size[1], n_channels)
    col = image[:, fancy_row, fancy_col, :]

    # (n_samples, n_new_rows, n_new_cols, kernel_size[0], kernel_size[1], n_channels)
    #   -> (n_channels * kernel_size[0] * kernel_size[1], n_new_rows * n_new_cols * n_samples)
    return col.transpose(5, 3, 4, 1, 2, 0).reshape(n_channels * kernel_size[0] * kernel_size[1], -1)

def col2im(col, image_shape, kernel_size, strides, padding='same'):
    '''col2im.

    Args:
        col (np.ndarray): shape = (n_channels * kernel_size[0] * kernel_size[1], n_new_rows * n_new_cols * n_samples)
        image_shape (tuple[int, int, int, int]): Image shape.
        kernel_size (tuple[int, int]): Height and width of the convolution window.
        stride (tuple[int, int]): Convolution stride along the height and width.
        padding (str): Padding rule, 'same' or 'valid'.

    Returns:
        np.ndarray, shape = (n_samples, n_rows, n_cols, n_channels).
    '''
    n_samples, n_rows, n_cols, n_channels = image_shape

    # initialize (padded) image
    pr1, pr2, pc1, pc2 = get_pad_width(kernel_size, padding)
    image = np.zeros((n_samples, n_rows + pr1 + pr2, n_cols + pc1 + pc2, n_channels))

    fancy_row, fancy_col = get_fancy_indices((image.shape[1], image.shape[2]), kernel_size, strides)
    n_new_rows, n_new_cols, _, _ = fancy_row.shape

    # reshape back to (n_samples, n_new_rows, n_new_cols, kernel_size[0], kernel_size[1], n_channels)
    col = col.reshape(n_channels, kernel_size[0], kernel_size[1], n_new_rows, n_new_cols, n_samples)
    col = col.transpose(5, 3, 4, 1, 2, 0)

    # add back to image
    # Note: col2im(im2col(iamge)) != image
    np.add.at(image, [slice(None), fancy_row, fancy_col, slice(None)], col)

    return image[:, pr1:image.shape[1]-pr2, pc1:image.shape[2]-pc2, :]

def get_pad_width(kernel_size, padding='same'):
    if padding != 'same':
        return 0, 0, 0, 0

    # padded_length = length + kernel_length - 1
    #
    # * odd kernel length
    #   + + - - - - - - - - + +
    #   0 1 2 3 4
    #                 0 1 2 3 4
    #
    #   pad_left = pad_right = (kernel_length - 1) // 2
    #
    # * even kernel length
    #   + - - - - - - - - + +
    #   0 1 2 3
    #                 0 1 2 3
    #
    #   pad_left = (kernel_length - 1) // 2
    #   pad_right = (kernel_length - 1) // 2 + 1
    k1, k2 = kernel_size
    return k1 // 2 - (k1 % 2 != 1), k1 // 2, k2 // 2 - (k2 % 2 != 1), k2 // 2

def get_fancy_indices(image_size, kernel_size, strides):
    # The generated fancy indices have shape (n_new_rows, n_new_cols, kernel_size[0], kernel_size[1]).
    # It could be used to solve 'extract contiguous block from matrix' problem.

    # local index within block
    indices = np.indices(kernel_size)
    # offset along row
    i = np.arange(0, image_size[0] - kernel_size[0] + 1, strides[0])
    # offset along column
    j = np.arange(0, image_size[1] - kernel_size[1] + 1, strides[1])
    # offset along row and column
    ii, jj = np.meshgrid(i, j, indexing='ij')

    return indices[0] + ii[..., np.newaxis, np.newaxis], indices[1] + jj[..., np.newaxis, np.newaxis]

class Flatten(Layer):
    """Flatten layer."""
    def build(self, input_shape, optimizer=None):
        self.output_shape = np.prod(input_shape)

        super().build(input_shape, optimizer)

    def forward(self, X, training=True):
        if training:
            self.layer_input = X
        return X.reshape(-1, self.output_shape)

    def backprop(self, grad):
        return grad.reshape(-1, *np.atleast_1d(self.input_shape))


class MaxPooling2D(Layer):
    """2D max pooling layer.

    Args:
        pool_size (tuple[int, int]): Height and width of pooling window.
        strides (optional(tuple[int, int])): Stride along the height and width.
    """
    def __init__(self, pool_size=(2, 2), strides=None):
        super().__init__()

        self.pool_size = pool_size

        # strides defaults to pool_size
        if strides is None:
            self.strides = pool_size
        else:
            self.strides = strides

        # from cs231n:
        # > Note that it is not common to use zero-padding for Pooling layers
        self.padding = 'valid'

        self.col = None

        # keep track of the index of max value with each pool for back propagation
        # shape = (n_channels * n_new_rows * n_new_cols * n_samples,)
        self.argmax = None

    def build(self, input_shape, optimizer=None):
        self.output_shape = self.compute_output_shape(input_shape)

        super().build(input_shape, optimizer)

    def forward(self, X, training=True):
        # |---|---|---|---|
        # | 1 | 1 | 2 | 4 |      |---|---|
        # | 5 | 6 | 7 | 8 |      | 6 | 8 |
        # |---|---|---|---|  ->  |---|---|
        # | 3 | 2 | 1 | 0 |      | 3 | 4 |
        # | 1 | 2 | 3 | 4 |      |---|---|
        # |---|---|---|---|

        if training:
            self.layer_input = X

        n_new_rows, n_new_cols, n_channels = self.output_shape

        # (n_channels * pool_size[0] * pool_size[1], n_new_rows * n_new_cols * n_samples)
        self.col = im2col(X, self.pool_size, self.strides, self.padding)

        # (n_channels * pool_size[0] * pool_size[1], n_new_rows * n_new_cols * n_samples)
        #   -> (pool_size[0] * pool_size[1], n_channels * n_new_rows * n_new_cols * n_samples)
        layer_output = self.col.reshape(n_channels, self.pool_size[0] * self.pool_size[1], -1)
        layer_output = layer_output.transpose(1, 0, 2)
        layer_output = layer_output.reshape(self.pool_size[0] * self.pool_size[1], -1)

        # get index of max value within each pool
        self.argmax = layer_output.argmax(axis=0)

        # (n_channels * n_new_rows * n_new_cols * n_samples,)
        layer_output = layer_output[self.argmax, np.arange(self.argmax.size)]

        # (n_channels * n_new_rows * n_new_cols * n_samples,)
        #   -> (n_samples, n_new_rows, n_new_cols, n_channels)
        layer_output = layer_output.reshape(n_channels, n_new_rows, n_new_cols, -1)
        layer_output = layer_output.transpose(3, 1, 2, 0)

        return layer_output

    def backprop(self, grad):
        #                |---|---|---|---|
        # |---|---|      | 0 | 0 | 0 | 0 |
        # | 6 | 8 |      | 0 | 6 | 0 | 8 |
        # |---|---|  ->  |---|---|---|---|
        # | 3 | 4 |      | 3 | 0 | 0 | 0 |
        # |---|---|      | 0 | 0 | 0 | 4 |
        #                |---|---|---|---|
        n_new_rows, n_new_cols, n_channels = self.output_shape

        # (n_samples, n_new_rows, n_new_cols, n_channels)
        #   -> (n_channels * n_new_rows * n_new_cols * n_samples,)
        grad = grad.transpose(3, 1, 2, 0).ravel()

        # (pool_size[0] * pool_size[1], n_channels * n_new_rows * n_new_cols * n_samples)
        col_grad = np.zeros((self.pool_size[0] * self.pool_size[1], grad.size))
        col_grad[self.argmax, np.arange(self.argmax.size)] = grad

        # (pool_size[0] * pool_size[1], n_channels * n_new_rows * n_new_cols * n_samples)
        #   -> (n_channels * pool_size[0] * pool_size[1], n_new_rows * n_new_cols * n_samples)
        col_grad = col_grad.reshape(self.pool_size[0] * self.pool_size[1], n_channels, -1)
        col_grad = col_grad.transpose(1, 0, 2)
        col_grad = col_grad.reshape(n_channels * self.pool_size[0] * self.pool_size[1], -1)

        return col2im(col_grad, self.layer_input.shape, self.pool_size, self.strides, self.padding)

    def compute_output_shape(self, input_shape):
        n_rows, n_cols, n_channels = input_shape

        n_new_rows = (n_rows - self.pool_size[0]) // self.strides[0] + 1
        n_new_cols = (n_cols - self.pool_size[1]) // self.strides[1] + 1

        return (n_new_rows, n_new_cols, n_channels)


class Dropout(Layer):
    """Dropout layer.

    Args:
        rate (float): Fraction of the input units to drop.
    """
    def __init__(self, rate):
        super().__init__()

        self.rate = rate

        self.p = None  # drop threshold
        self.drop_mask = None

    def build(self, input_shape, optimizer=None):
        self.output_shape = input_shape
        super().build(input_shape, optimizer)

    def forward(self, X, training=True):
        if not training:
            return X * self.p

        self.layer_input = X

        # numbers of input units to drop
        drop_units = int(np.prod(X.shape) * self.rate)
        probs = np.random.rand(*X.shape)
        self.p = np.partition(probs.ravel(), drop_units)[drop_units]
        self.drop_mask = probs < self.p

        return X * (~self.drop_mask)


    def backprop(self, grad):
        return grad * (~self.drop_mask)
