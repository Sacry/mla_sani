import numpy as np

from ..utils import MultivariateNormalDistribution
from .cluster import KMeans

class GaussianMixture(object):
    """A SaNI of gaussian mixture model.

    Args:
        n_components (int): Number of compnents.
        tol (float): Convergence threshold.
        max_iter (int): Maximum number of EM iterations.
    """
    def __init__(self, n_components=1, tol=0.001, max_iter=100):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X):
        """Fit model.

        Args:
            X (np.ndarray): shape = (n_samples, n_features). Training data.
        """
        # # random initialization
        # resp = np.random.rand(X.shape[0], self.n_components)
        # resp /= resp.sum(axis=1, keepdims=True)

        # kmeans initialization
        resp = np.zeros((X.shape[0], self.n_components))
        label = KMeans(n_clusters=self.n_components).fit(X).labels_
        resp[np.arange(X.shape[0]), label] = 1

        weights, means, covariances = self._m_step(X, resp)

        llhp = None  # previous log-likelihood
        for n_iter in range(self.max_iter):
            # expectation
            resp = self._e_step(X, weights, means, covariances)

            # maximization
            weights, means, covariances = self._m_step(X, resp)

            # convergence test
            llh = np.sum(np.log(resp).max(axis=1))  # is it right?
            if llhp is not None and abs(llh - llhp) < self.tol:
                break
            llhp = llh

        setattr(self, "n_iter_", n_iter)
        setattr(self, "converged_", abs(llh - llhp) < self.tol)
        setattr(self, "lower_bound_", llh)

        setattr(self, "weights_", np.array(weights))
        setattr(self, "means_", np.array(means))
        setattr(self, "covariances_", np.array(covariances))

        return self

    def _e_step(self, X, weights, means, covariances):
        resp = np.column_stack([
            weight * MultivariateNormalDistribution(mean, cov).pdf(X)
            for weight, mean, cov in zip(weights, means, covariances)
        ])
        resp /= resp.sum(axis=1, keepdims=True)
        return resp

    def _m_step(self, X, resp):
        weights, means, covariances = [], [], []
        for i in range(self.n_components):
            resp_i = np.expand_dims(resp[:, i], axis=1)

            weights.append(resp_i.mean())
            means.append((resp_i * X).sum(axis=0) / resp_i.sum())
            covariances.append((resp_i * (X - means[i])).T.dot(X - means[i]) / resp_i.sum())
        return weights, means, covariances

    def predict(self, X):
        """Perform predict on X.

        Args:
            X (np.ndarray): shape = (n_samples, n_features). Data to predict.

        Returns:
            np.ndarray: shape = (n_samples,). Predicted result.
        """
        return self._e_step(X, self.weights_, self.means_, self.covariances_).argmax(axis=1)

    def aic(self, X):
        pass

    def bic(self, X):
        pass
