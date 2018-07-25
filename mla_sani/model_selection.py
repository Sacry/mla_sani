from itertools import combinations

import numpy as np

from .base import BaseFold


def cross_val_score(estimator, X, y, scoring, cv):
    return [
        scoring(estimator.fit(X[train], y[train]), X[test], y[test])
        for train, test in cv.split(X, y)
    ]

class KFold(BaseFold):
    def __init__(self, n_splits):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        indices = np.arange(X.shape[0])
        splits = np.array_split(indices, self.n_splits)

        for i in range(self.n_splits):
            yield np.concatenate(splits[:i] + splits[i+1:]), splits[i]

class LeaveOneOut(BaseFold):
    def split(self, X, y=None, groups=None):
        yield from KFold(n_splits=X.shape[0]).split(X)

class LeavePOut(BaseFold):
    def __init__(self, p):
        self.p = p

    def split(self, X, y=None, groups=None):
        indices = np.arange(X.shape[0])
        for test_indices in combinations(indices, self.p):
            yield np.delete(indices, test_indices), np.array(test_indices)

class ShuffleSplit(BaseFold):
    def __init__(self, n_splits=10, test_size=0.1, random_state=None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        rand_state = np.random.RandomState(self.random_state)
        for i in range(self.n_splits):
            rand_state.shuffle(indices)
            test_end = np.ceil(n_samples * self.test_size).astype(int)
            yield indices[test_end:], indices[:test_end]

class StratifiedKFold(BaseFold):
    def __init__(self, n_splits):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        kf = KFold(n_splits=self.n_splits)

        class_indices = [np.argwhere(y==c).ravel() for c in np.unique(y)]
        # convert per-class split result (local index within each calss) to global index
        def convert_indices(class_split_result):
            return [ci[i] for ci, i in zip(class_indices, class_split_result)]

        # when did I start writing code like this and considering it 'recallable'
        for ith_split in zip(*(kf.split(ci) for ci in class_indices)):
            train_indices, test_indices = map(convert_indices, zip(*ith_split))
            yield np.concatenate(train_indices), np.concatenate(test_indices)

class GroupKFold(BaseFold):
    def __init__(self, n_splits):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        kf = KFold(n_splits=self.n_splits)

        indices = np.arange(X.shape[0])
        unique_groups = np.unique(groups)

        for train, test in kf.split(unique_groups):
            train = indices[np.in1d(groups, unique_groups[train])]
            test = indices[np.in1d(groups, unique_groups[test])]
            yield train, test

class TimeSeriesSplit(BaseFold):
    def __init__(self, n_splits=3):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        k = n_samples // (self.n_splits + 1)
        for i in range(self.n_splits):
            train_end = (i+1) * k + n_samples % (self.n_splits + 1)
            yield indices[:train_end], indices[train_end:train_end+k]
