"""
Testing for preprocessing.
"""

from unittest import TestCase

import numpy as np
from sklearn.datasets import make_classification
from sklearn import preprocessing as sklearn_preprocessing

from mla_sani import preprocessing


class testStandardScaler(TestCase):
    def test_transform(self):
        X, _ = make_classification()
        self.assertTrue(np.allclose(preprocessing.StandardScaler().fit_transform(X),
                                    sklearn_preprocessing.StandardScaler().fit_transform(X)))

class testMinMaxScaler(TestCase):
    def test_transform(self):
        X, _ = make_classification()
        self.assertTrue(np.allclose(preprocessing.MinMaxScaler().fit_transform(X),
                                    sklearn_preprocessing.MinMaxScaler().fit_transform(X)))

        self.assertTrue(np.allclose(preprocessing.MinMaxScaler(feature_range=(0, 10)).fit_transform(X),
                                    sklearn_preprocessing.MinMaxScaler(feature_range=(0, 10)).fit_transform(X)))

class testMaxAbsScaler(TestCase):
    def test_transform(self):
        X, _ = make_classification()
        self.assertTrue(np.allclose(preprocessing.MaxAbsScaler().fit_transform(X),
                                    sklearn_preprocessing.MaxAbsScaler().fit_transform(X)))


class testNormalizer(TestCase):
    def test_transform(self):
        X, _ = make_classification()
        self.assertTrue(np.allclose(preprocessing.Normalizer(norm="l1").fit_transform(X),
                                    sklearn_preprocessing.Normalizer(norm="l1").fit_transform(X)))

        self.assertTrue(np.allclose(preprocessing.Normalizer(norm="l2").fit_transform(X),
                                    sklearn_preprocessing.Normalizer(norm="l2").fit_transform(X)))

        # self.assertTrue(np.allclose(preprocessing.Normalizer(norm="max").fit_transform(X),
        #                             sklearn_preprocessing.Normalizer(norm="max").fit_transform(X)))

class testBinarizer(TestCase):
    def test_transform(self):
        X = np.random.randn(50, 50)
        self.assertTrue(np.allclose(preprocessing.Binarizer().fit_transform(X),
                                    sklearn_preprocessing.Binarizer().fit_transform(X)))

        self.assertTrue(np.allclose(preprocessing.Binarizer(threshold=0.3).fit_transform(X),
                                    sklearn_preprocessing.Binarizer(threshold=0.3).fit_transform(X)))

class testOneHostEncoder(TestCase):
    def test_transform(self):
        X = np.column_stack([
            np.random.randint(0, 10, 100),
            np.random.randint(0, 3, 100),
            np.random.randint(0, 5, 100),
        ])

        X_trans1 = preprocessing.OneHotEncoder().fit_transform(X)
        X_trans2 = sklearn_preprocessing.OneHotEncoder().fit_transform(X)
        self.assertTrue((X_trans1 == X_trans2).all())

class testLabelEncoder(TestCase):
    def test_transform(self):
        y = np.random.choice(list("abcd"), 100)
        X_trans1 = preprocessing.LabelEncoder().fit_transform(y)
        X_trans2 = sklearn_preprocessing.LabelEncoder().fit_transform(y)
        self.assertTrue((X_trans1 == X_trans2).all())

class testLabelBinarizer(TestCase):
    def test_transform(self):
        y = np.random.randint(0, 5, 100)
        X_trans1 = preprocessing.LabelBinarizer().fit_transform(y)
        X_trans2 = sklearn_preprocessing.LabelBinarizer().fit_transform(y)
        self.assertTrue((X_trans1 == X_trans2).all())

class testImputer(TestCase):
    def test_transform(self):
        X = np.random.randint(0, 100, (30, 30)).astype(np.float)

        i = np.random.choice(np.arange(30), 50)
        j = np.random.choice(np.arange(30), 50)
        X[i, j] = np.nan

        self.assertTrue(np.allclose(preprocessing.Imputer(strategy="mean", axis=0).fit_transform(X),
                                    sklearn_preprocessing.Imputer(strategy="mean", axis=0).fit_transform(X)))

        self.assertTrue(np.allclose(preprocessing.Imputer(strategy="median", axis=0).fit_transform(X),
                                    sklearn_preprocessing.Imputer(strategy="median", axis=0).fit_transform(X)))

        self.assertTrue(np.allclose(preprocessing.Imputer(strategy="most_frequent", axis=0).fit_transform(X),
                                    sklearn_preprocessing.Imputer(strategy="most_frequent", axis=0).fit_transform(X)))

        self.assertTrue(np.allclose(preprocessing.Imputer(strategy="mean", axis=1).fit_transform(X),
                                    sklearn_preprocessing.Imputer(strategy="mean", axis=1).fit_transform(X)))

        self.assertTrue(np.allclose(preprocessing.Imputer(strategy="median", axis=1).fit_transform(X),
                                    sklearn_preprocessing.Imputer(strategy="median", axis=1).fit_transform(X)))

        self.assertTrue(np.allclose(preprocessing.Imputer(strategy="most_frequent", axis=1).fit_transform(X),
                                    sklearn_preprocessing.Imputer(strategy="most_frequent", axis=1).fit_transform(X)))
