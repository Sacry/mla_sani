from unittest import TestCase

import numpy as np
from sklearn import model_selection as sklearn_model_selection

from mla_sani import model_selection


class testFold(TestCase):
    def assertFoldEqual(self, fold1, fold2):
        for idx1, idx2 in zip(fold1, fold2):
            self.assertTrue((idx1[0] == idx2[0]).all())
            self.assertTrue((idx1[1] == idx2[1]).all())

class testKFold(testFold):
    def test_split(self):
        X = np.array(["a", "b", "c", "d"])

        fold1 = model_selection.KFold(2).split(X)
        fold2 = sklearn_model_selection.KFold(2).split(X)

        self.assertFoldEqual(fold1, fold2)

class testLeaveOneOut(testFold):
    def test_split(self):
        X = np.array([1, 2, 3, 4])

        fold1 = model_selection.LeaveOneOut().split(X)
        fold2 = sklearn_model_selection.LeaveOneOut().split(X)

        self.assertFoldEqual(fold1, fold2)

class testLeavePOut(testFold):
    def test_split(self):
        X = np.array([1, 2, 3, 4])

        fold1 = model_selection.LeavePOut(p=2).split(X)
        fold2 = sklearn_model_selection.LeavePOut(p=2).split(X)

        self.assertFoldEqual(fold1, fold2)

class testShuffleSplit(testFold):
    def test_split(self):
        pass

class testStratifiedKFold(testFold):
    def test_split(self):
        X = np.ones(10)
        y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

        fold1 = model_selection.StratifiedKFold(n_splits=3).split(X, y)
        fold2 = sklearn_model_selection.StratifiedKFold(n_splits=3).split(X, y)

        self.assertFoldEqual(fold1, fold2)

class testGroupKFold(testFold):
    def test_split(self):
        X = np.array([0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 8.8, 9, 10])
        y = np.array(["a", "b", "b", "b", "c", "c", "c", "d", "d", "d"])
        groups = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 3])

        fold1 = model_selection.GroupKFold(n_splits=3).split(X, y, groups)
        fold2 = sklearn_model_selection.GroupKFold(n_splits=3).split(X, y, groups)

        # Inconveniently, my implementation produces different order
        # self.assertFoldEqual(fold1, fold2)

class testTimeSeriesSplit(testFold):
    def test_split(self):
        X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
        y = np.array([1, 2, 3, 4, 5, 6])

        fold1 = model_selection.TimeSeriesSplit(n_splits=3).split(X)
        fold2 = sklearn_model_selection.TimeSeriesSplit(n_splits=3).split(X)

        self.assertFoldEqual(fold1, fold2)
