import numpy as np


class Sequential(object):
    """

    Examples:
        >>> reg = Sequential()

        >>> reg.add(Input(X_train.shape[1]))
        >>> reg.add(Dense(10))
        >>> reg.add(Dense(1))

        >>> reg.compile(optimizer=SGD(lr=0.001), loss=SquareLoss())

        >>> reg.fit(X_train, y_train, epochs=1000, batch_size=100)

        >>> y_pred = reg.predict(X_test)
    """
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss

        for i in range(1, len(self.layers)):
            self.layers[i].build(self.layers[i-1].units, optimizer)

    def fit(self, X, y, batch_size=None, epochs=1):
        self.loss_curve = []
        for _ in range(epochs):
            batch_losses = []
            for batch_indices in self._next_batch_indices(X.shape[0], batch_size):
                batch_losses.append(self._train_on_batch(X[batch_indices], y[batch_indices]))
            self.loss_curve.append(np.sum(batch_losses))

    def predict(self, X, batch_size=None):
        y_pred = []
        for batch_indices in self._next_batch_indices(X.shape[0], batch_size, shuffle=False):
            y_pred.append(self._predict_on_batch(X[batch_indices]))
        return np.concatenate(y_pred).ravel()

    def _next_batch_indices(self, n_samples, batch_size, shuffle=True):
        if batch_size is None:
            batch_size = n_samples
        n_batches = n_samples // batch_size + (n_samples % batch_size != 0)

        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)

        for batch_indices in np.array_split(indices, n_batches):
            yield batch_indices

    def _train_on_batch(self, X, y):
        # forward
        self.layers[0].forward(X)
        for i in range(1, len(self.layers)):
            self.layers[i].forward(self.layers[i-1].layer_output)

        batch_loss = self.loss.loss(y[:, np.newaxis], self.layers[-1].layer_output).sum()

        # backprop
        grad = self.loss.dloss(y[:, np.newaxis], self.layers[-1].layer_output)
        for i in range(len(self.layers)-1, 0, -1):
            grad = self.layers[i].backprop(grad)

        return batch_loss

    def _predict_on_batch(self, X):
        for i in range(1, len(self.layers)):
            X = self.layers[i].forward(X, train=False)
        return X.ravel()
