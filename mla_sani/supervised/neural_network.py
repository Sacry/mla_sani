import numpy as np

from ..losses import CrossEntropy
from ..activations import Sigmoid

class MLPClassifier(object):
    """A SaNI of multilayer perceptron classifier.

    Note:
        This impelementation supports binary classification only.

    Args:
        hidden_layer_sizes (tuple): The number of neurons in each hidden layer.
        learning_rate_init (float): Learning rate.
        max_iter (int): Maximum number of iterations.
    """

    def __init__(self, hidden_layer_sizes=(100,), learning_rate_init=0.1, max_iter=200):
        self.hidden_layer_sizes = hidden_layer_sizes

        self.loss = CrossEntropy()
        self.activation = Sigmoid()

        self.learning_rate = learning_rate_init
        self.max_iter = max_iter

    def fit(self, X, y):
        """Fit model.

        Args:
            X (np.ndarray): shape = (n_samples, n_features). Training data.
            y (np.ndarray): shape = (n_samples,). Target values.
        """
        n_samples, n_features = X.shape

        # number of outputs
        setattr(self, "n_outputs_", 1)

        # If I'm not wrong, multi-label classification could be achieved by
        #   * set n_outputs_ to number of unique labels
        #   * choose multi-category cross entropy as loss function
        #   * choose softmax as last layer activation

        if self.hidden_layer_sizes is not None:
            layers = [n_features] + list(self.hidden_layer_sizes) + [self.n_outputs_]
        else:
            layers = [n_features, self.n_outputs_]
        setattr(self, "n_layers_", len(layers))

        # shape of ith coef: (layers[i], layers[i + 1])
        coefs = [np.random.randn(layers[i], layers[i + 1]) for i in range(self.n_layers_ - 1)]

        # shape of ith itercept: (layers[i + 1],)
        intercepts = [np.random.randn(layers[i + 1]) for i in range(self.n_layers_ - 1)]

        # shape of ith layer input/outpt: (n_samples, layers[i])
        # layer input, not actually necessary
        z = [np.empty((n_samples, n_layer)) for n_layer in layers]
        # layer output
        a = [np.empty((n_samples, n_layer)) for n_layer in layers]
        a[0] = X

        # shape of ith delta: (n_samples, layers[i+1])
        deltas = [np.empty((n_samples, n_layer)) for n_layer in layers[1:]]

        # index and "position" of each variable:
        #
        # | layers      | 0 |   | 1 |   | ... | n-2 |     | n-1 |
        # |-------------|---|---|---|---|-----|-----|-----|-----|
        # | coefs       |   | 0 |   | 1 | ... |     | n-2 |     |
        # | intercepts  |   | 0 |   | 1 | ... |     | n-2 |     |
        # | a           | 0 |   | 1 |   |     | n-2 |     | n-1 |
        # | z           | 0 |   | 1 |   |     | n-2 |     | n-1 |
        # | deltas      |   |   | 0 |   |     | n-3 |     | n-2 |

        loss_curve = []

        for n_iter in range(self.max_iter):
            # (1) forward
            for i in range(len(coefs)):
                z[i + 1] = a[i].dot(coefs[i]) + intercepts[i]
                a[i + 1] = self.activation(z[i + 1])

            loss_curve.append(self.loss.loss(y[:, np.newaxis], a[-1]).mean())

            # (2) back propagation
            # last layer delta: gradient of loss function with respect to z, the famous y_hat - y.
            deltas[-1] = self.loss.dloss(y[:, np.newaxis], a[-1]) * self.activation.derivative(z[-1])

            for i in range(len(deltas)-2, -1, -1):
                deltas[i] = deltas[i+1].dot(coefs[i+1].T) * self.activation.derivative(z[i+1])

            # (3) update weight
            for i in range(len(coefs)):
                coefs[i] -= self.learning_rate * a[i].T.dot(deltas[i]) / n_samples
                intercepts[i] -= self.learning_rate * np.mean(deltas[i], axis=0)

            # convergence test

        setattr(self, "n_iter_", n_iter)
        setattr(self, "coefs_", coefs)
        setattr(self, "intercepts_", intercepts)
        setattr(self, "loss_curve_", loss_curve)
        return self

    def predict(self, X):
        """Perform predict on X.

        Args:
            X (np.ndarray): shape = (n_samples, n_features). Data to predict.

        Returns:
            np.array: shape = (n_samples,). Predicted result.
        """
        a = X
        for i in range(self.n_layers_ - 1):
            a = self.activation(a.dot(self.coefs_[i]) + self.intercepts_[i])
        return (a >= 0.5).astype(int).ravel()
