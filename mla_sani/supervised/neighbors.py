import numpy as np

from ..utils import pairwise_euclidean_distance, most_frequent

class KNeighborsClassifier(object):
    """A SaNI of k-nearest neighbors classifier.

    Args:
        n_neighbors (int): Number of neighbors to be used.
    """

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fit model.

        Args:
            X (np.ndarray): shape = (n_samples, n_features). Training data.
            y (np.ndarray): shape = (n_samples,). Target values.
        """
        setattr(self, "_X", X)
        setattr(self, "_y", y)
        return self

    def predict(self, X):
        """Perform predict on X.

        Args:
            X (np.ndarray): shape = (n_samples, n_features). Data to predict.

        Returns:
            np.ndarray: shape = (n_samples,). Predicted class labels.
        """
        distance = pairwise_euclidean_distance(X, self._X)

        # k-nearest neighbors
        k_nearest = self._y[distance.argsort(axis=1)[:, :self.n_neighbors]]

        return np.apply_along_axis(most_frequent, axis=1, arr=k_nearest)
