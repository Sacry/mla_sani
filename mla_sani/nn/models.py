import numpy as np

from math import ceil

class Sequential(object):
    def __init__(self):
        self.layers = []

        self.optimizer = None
        self.loss = None

        self.loss_curve = []

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss

        for i in range(1, len(self.layers)):
            self.layers[i].build(self.layers[i-1].output_shape, optimizer)

    def fit(self, X, y, batch_size=None, epochs=1):
        for _ in range(epochs):
            batch_losses = []
            for batch_indices in self._split(X.shape[0], batch_size):
                batch_losses.append(self._train_on_batch(X[batch_indices], y[batch_indices]))
            self.loss_curve.append(np.sum(batch_losses) / X.shape[0])

    def predict(self, X, batch_size=None):
        y_pred = []
        for batch_indices in self._split(X.shape[0], batch_size, shuffle=False):
            y_pred.append(self._predict_on_batch(X[batch_indices]))
        return np.concatenate(y_pred)

    def _split(self, n_samples, batch_size, shuffle=True):
        if batch_size is None:
            batch_size = n_samples
        n_batches = ceil(n_samples / batch_size)

        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)

        for batch_indices in np.array_split(indices, n_batches):
            yield batch_indices

    def _train_on_batch(self, X, y):
        # forward
        layer_output = self.layers[0].forward(X)
        for i in range(1, len(self.layers)):
            layer_output = self.layers[i].forward(layer_output)

        # the parameter of the loss function was supposed to be an 1d array (except for
        # CategoricalCrossEntropy), but the output of an nn layer will probably yield a 2d result
        # with shape (n_samples, n_units), so squeeze the `layer_output` before pass to loss
        # function
        batch_loss = self.loss.loss(y, np.squeeze(layer_output)).sum()
        grad = self.loss.dloss(y, np.squeeze(layer_output)).reshape(layer_output.shape)

        # backprop
        for i in range(len(self.layers)-1, 0, -1):
            grad = self.layers[i].backprop(grad)

        return batch_loss

    def _predict_on_batch(self, X):
        for i in range(1, len(self.layers)):
            X = self.layers[i].forward(X, train=False)
        return np.squeeze(X)
