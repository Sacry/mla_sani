import numpy as np

from .base import BaseTransformer
from .utils import most_frequent

class StandardScaler(BaseTransformer):
    def fit(self, X):
        setattr(self, "mean_", X.mean(axis=0))
        setattr(self, "var_", X.var(axis=0))

        scaling = X.std(axis=0)
        scaling[scaling == 0] = 1
        setattr(self, "scaling_", scaling)

        setattr(self, "n_samples_seen_ ", X.shape[0])

        return self

    def transform(self, X):
        return (X - self.mean_) / self.scaling_


class MinMaxScaler(BaseTransformer):
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range

    def fit(self, X):
        setattr(self, "data_min_", X.min(axis=0))
        setattr(self, "data_max_", X.max(axis=0))
        setattr(self, "data_range_", self.data_max_ - self.data_min_)

        numerator = self.feature_range[1] - self.feature_range[0]
        denominator = np.where(self.data_range_ == 0, 1, self.data_range_)
        setattr(self, "scale_", numerator / denominator)
        setattr(self, "min_", self.feature_range[0] - self.data_min_ * self.scale_)

        return self

    def transform(self, X):
        return X * self.scale_ + self.min_

    def inverse_transform(self, X):
        return (X - self.min_) / self.scale_


class MaxAbsScaler(BaseTransformer):
    def fit(self, X):
        setattr(self, "max_abs_", np.abs(X).max(axis=0))
        setattr(self, "scale_", np.where(self.max_abs_ == 0, 1, self.max_abs_))
        setattr(self, "n_samples_seen_", X.shape[0])
        return self

    def transform(self, X):
        return X / self.scale_


class Normalizer(BaseTransformer):
    NORM_TO_ORD = {"l1": 1, "l2": 2, "max": np.inf}

    def __init__(self, norm="l2"):
        self.ord = Normalizer.NORM_TO_ORD[norm]

    def transform(self, X):
        # Note: sklearn's max norm does not take absolute value first
        return X / np.linalg.norm(X, ord=self.ord, axis=1, keepdims=True)


class Binarizer(BaseTransformer):
    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def transform(self, X):
        return (X > self.threshold).astype(int)


class OneHotEncoder(BaseTransformer):
    def fit(self, X):
        # number of unique values per feature
        setattr(self, "n_values_", np.max(X, axis=0) + 1)

        # indices to feature ranges
        setattr(self, "feature_indices_", np.r_[0, self.n_values_.cumsum()])

        # indices of active features
        active_features = np.concatenate([
            np.unique(X[:, i]) + self.feature_indices_[i]
            for i in range(X.shape[1])
        ])
        setattr(self, "active_features_", active_features)

        return self

    def transform(self, X):
        n_samples, n_features = X.shape
        X_transform = np.zeros((n_samples, self.n_values_.sum()))

        row_indices = np.repeat(np.arange(n_samples), n_features)
        column_indices = np.ravel(X + self.feature_indices_[:-1])
        X_transform[row_indices, column_indices] = 1

        return X_transform[:, self.active_features_]


class LabelEncoder(BaseTransformer):
    def fit(self, y):
        setattr(self, "classes_", np.unique(y))
        return self

    def transform(self, y):
        return np.searchsorted(self.classes_, y)


class LabelBinarizer(BaseTransformer):
    def fit(self, y):
        setattr(self, "classes_", np.unique(y))
        return self

    def transform(self, y):
        n_samples = y.shape[0]

        y_transform = np.zeros((n_samples, len(self.classes_)), dtype=np.int32)
        row_index = np.arange(n_samples)
        col_index = np.searchsorted(self.classes_, y)
        y_transform[row_index, col_index] = 1

        return y_transform

    def inverse_transform(self, y):
        return self.classes_[np.where(y == 1)[1]]


class Imputer(BaseTransformer):
    def __init__(self, strategy="mean", axis=0):
        self.strategy = strategy
        self.axis = axis

    def fit(self, X):
        if self.strategy == "mean":
            statistics = np.nanmean(X, axis=self.axis)
        elif self.strategy == "median":
            statistics = np.nanmedian(X, axis=self.axis)
        elif self.strategy == "most_frequent":
            statistics = np.apply_along_axis(most_frequent, self.axis, X)
        else:
            raise ValueError()

        setattr(self, "statistics_", statistics)
        return self

    def transform(self, X):
        indices = np.where(np.isnan(X))
        X[indices] = self.statistics_.take(indices[self.axis])
        return X

def gen_powers(degree, cur_degree, n_feature, i_feature, power):
    if degree == cur_degree or i_feature == n_feature:
        yield power.copy()
        return

    for d in range(degree - cur_degree + 1):
        power[i_feature] = d
        yield from gen_powers(degree, cur_degree+d, n_feature, i_feature+1, power)
        power[i_feature] = 0

class PolynomialFeatures(BaseTransformer):
    def __init__(self, degree=2):
        self.degree = degree

    def transform(self, X):
        n_features = X.shape[1]
        # It does not have the same order as sklearn, and at least n_feature times slower, lame
        powers = gen_powers(self.degree, 0, n_features, 0, np.zeros(n_features))
        return np.column_stack([(X ** power).prod(axis=1) for power in powers])
