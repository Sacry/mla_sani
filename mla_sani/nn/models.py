import numpy as np

from math import ceil

class Sequential(object):
    """Sequential model.

    Example:
        >>> data = load_digits()
        >>> X, y = data.data, data.target
        >>> X = MinMaxScaler().fit_transform(X).reshape(-1, 8, 8, 1)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y)

        >>> cnn = Sequential()
        >>> cnn.add(Input(X.shape[1:]))
        >>> cnn.add(Conv2D(16, (3, 3), padding='same'))
        >>> cnn.add(Activation('relu'))
        >>> cnn.add(Dropout(rate=0.1))
        >>> cnn.add(Flatten())
        >>> cnn.add(Dense(10))
        >>> cnn.add(Activation('softmax'))

        >>> cnn.compile(optimizer=Adam(), loss=CategoricalCrossEntropy(labels=np.unique(y)))
        >>> cnn.fit(X_train, y_train, epochs=30, batch_size=128)

        >>> y_pred = cnn.predict(X_test).argmax(axis=1)
    """
    def __init__(self):
        self.layers = []

        self.optimizer = None
        self.loss = None

        # loss of each epoch
        self.loss_curve = []

    def add(self, layer):
        """Add layer."""
        self.layers.append(layer)

    def compile(self, optimizer, loss):
        """Configure model.

        Args:
            optimizer: Optimizer instance.
            loss: Loss function instance.
        """
        self.optimizer = optimizer
        self.loss = loss

        for i in range(1, len(self.layers)):
            self.layers[i].build(self.layers[i-1].output_shape, optimizer)

    def fit(self, X, y, batch_size=None, epochs=1):
        """Train model.

        Args:
            X (np.ndarray): Data to train.
            y (np.ndarray): Target data.
            batch_size (optional[int]): Batch size.
            epochs (optional[int]): Number of epochs to train the model.
        """
        for _ in range(epochs):
            batch_losses = []
            for batch_indices in self._split(X.shape[0], batch_size):
                batch_losses.append(self._train_on_batch(X[batch_indices], y[batch_indices]))
            self.loss_curve.append(np.sum(batch_losses) / X.shape[0])

    def predict(self, X, batch_size=None):
        """ Perform predict.

        Args:
            X (np.ndarray): Data to predict.
            batch_size (optional[int]): Batch size.            
        """
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
        # CategoricalCrossEntropy), but nn layer will probably yield a 2d output with shape
        # (n_samples, n_units), so squeeze the `layer_output` before pass to loss function
        batch_loss = self.loss.loss(y, np.squeeze(layer_output)).sum()
        grad = self.loss.dloss(y, np.squeeze(layer_output)).reshape(layer_output.shape)

        # backprop
        for i in range(len(self.layers)-1, 0, -1):
            grad = self.layers[i].backprop(grad)

        return batch_loss

    def _predict_on_batch(self, X):
        for i in range(1, len(self.layers)):
            X = self.layers[i].forward(X, training=False)
        return np.squeeze(X)
