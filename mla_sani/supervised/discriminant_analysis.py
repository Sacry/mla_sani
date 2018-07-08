import numpy as np

from ..utils import MultivariateNormalDistribution

class LinearDiscriminantAnalysis(object):
    """A SaNI of linear discriminant analysis.

    LDA assumes features as multivariate normal distributed. It can be used to perform
    classification, as well as supervised demensionality reduction. When used to do demensionality
    reduction, it tries to project data onto the max separating direction.

    Args:
        n_components (int, optional): Number of components for dimensionality reduction.
    """
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y):
        """Fit model.

        Args:
            X (np.ndarray): shape = (n_samples, n_features). Training data.
            y (np.ndarray): shape = (n_samples,). Target values.
        """
        # unique class labels, shape = (n_classes,)
        setattr(self, "classes_", np.unique(y))

        # n_samples per class
        counts = np.array([(y == c).sum() for c in self.classes_])

        # class prior, shape = (n_classes,)
        setattr(self, "priors_", counts / y.size)

        # class means, shape = (n_classes, n_features)
        setattr(self, "means_", np.array([X[y == c].mean(axis=0) for c in self.classes_]))

        # overall mean
        setattr(self, "xbar_", np.dot(self.priors_, self.means_))

        # overall covariance
        setattr(self, "covariance_", np.cov(X.T))

        # for convenience
        def tdot(a):
            a = np.atleast_2d(a)
            return a.T.dot(a)

        # https://en.wikipedia.org/wiki/Linear_discriminant_analysis#Multiclass_LDA

        # between-class scatter:
        #   Sb = ∑ N_i (μ_i - μ) (μ_i - μ)^T
        # where
        #   N_i is the number of samples of the ith class
        #   μ_i is the mean of the ith class
        #   μ is the overall mean
        Sb = np.sum([
            count * tdot(mean - self.xbar_) for count, mean in zip(counts, self.means_)
        ], axis=0)

        # within-class scatter:
        #   Sw = ∑∑ (X_ij - μ_i) (X_ij - μ_i)^T
        # where
        #   X_ij is the jth sample of the ith class
        #   μ_i is the mean of the ith class
        Sw = np.sum([
            tdot(X[y == c] - mean) for c, mean in zip(self.classes_, self.means_)
        ], axis=0)

        # class separation is defined as: S = Sb / Sw
        W = np.linalg.pinv(Sw).dot(Sb)
        eig_vals, eig_vecs = np.linalg.eig(W)

        # sort eigenvectors
        eig_vecs = eig_vecs[:, eig_vals.argsort()[::-1]]

        if self.n_components is None:
            self.n_components = X.shape[1]
        setattr(self, "scalings_", eig_vecs[:, :self.n_components])
        return self

    # This is not how it works...
    # 
    # def predict(self, X):
    #     """Perform predict on X.

    #     Args:
    #         X (np.ndarray): shape = (n_samples, n_features). Data to predict.

    #     Returns:
    #         np.ndarray: shape = (n_samples,). Predicted result.
    #     """
    #     likelihood = np.column_stack([
    #         MultivariateNormalDistribution(mean, self.covariance_).pdf(X)
    #         for mean in self.means_
    #     ])
    #     posterior = likelihood * self.priors_
    #     return self.classes_[posterior.argmax(axis=1)]

    def transform(self, X):
        return X.dot(self.scalings_)
