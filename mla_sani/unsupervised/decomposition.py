import numpy as np

from ..utils import covariance_matrix

class PCA:
    """A SaNI of principal component analysis.

    Args:
        n_components (int): number of components to keep
    """

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        # covariance matrix
        cov = covariance_matrix(X.T)

        # U, _, _ = np.linalg.svd(cov)
        eig_vals, eig_vecs = np.linalg.eig(cov)

        # sort eigenvectors
        eig_vecs = eig_vecs[:, eig_vals.argsort()[::-1]]

        setattr(self, "components_", eig_vecs[:, :self.n_components])
        setattr(self, "explained_variance_", eig_vals[:self.n_components])
        return self

    def transform(self, X):
        """Apply dimensionality reduction.

        Args:
            X (np.ndarray): shape = (n_samples, n_features). Data to transform.

        Returns:
            np.ndarray: shape = (n_samples, n_components). Reduced data.
        """
        # apply transformation
        return X.dot(self.components_)
