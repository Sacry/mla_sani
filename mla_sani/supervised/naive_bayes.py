import numpy as np

from ..utils import NormalDistribution

class GaussianNB(object):
    """A SaNI of Gaussian Naive Bayes classifier."""

    def fit(self, X, y):
        """Fit model.

        Args:
            X (np.ndarray): shape = (n_samples, n_features). Training data.
            y (np.ndarray): shape = (n_samples,). Target values.
        """
        # unique class labels
        setattr(self, "classes_", np.unique(y))

        # number of tranning samples per class
        setattr(self, "class_count_", np.array([(y == c).sum() for c in self.classes_]))

        # prior probability (probability of each class)
        setattr(self, "class_prior_", self.class_count_ / y.size)

        # mean of each feature per class
        setattr(self, "theta_", np.array([X[y == c].mean(axis=0) for c in self.classes_]))

        # variance of each feature per class
        setattr(self, "sigma_", np.array([X[y == c].var(axis=0) for c in self.classes_]))

        return self

    def predict(self, X):
        """Perform predict on X.

        Args:
            X (np.ndarray): shape = (n_samples, n_features). Data to predict.

        Returns:
            np.array: shape = (n_samples,). Predicted class labels.
        """
        # Naive bayes assumes the independence of features
        # P(X|Y) = product of P(x_i|Y)
        likelihood = np.column_stack([
            NormalDistribution(mean, var).pdf(X).prod(-1)
            for mean, var in zip(self.theta_, self.sigma_)
        ])

        # P(Y|X) = P(X|Y) * P(Y) / P(X)
        # P(X) is omitted here as it doesn't affect the comparing result
        posterior = likelihood * self.class_prior_

        # class with the largest P(Y|X)
        return self.classes_.take(posterior.argmax(axis=1))
