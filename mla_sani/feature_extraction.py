import numbers

import numpy as np

from .base import BaseTransformer

class DictVectorizer(BaseTransformer):
    def __init__(self, separator="="):
        self.separator = separator

    def fit(self, X):
        """
        Args:
            X (List[Dict]):
        """
        feature_names = set()
        for x in X:
            for k, v in x.items():
                feature_names.add(self._get_feature_name(k, v))

        feature_names = list(sorted(feature_names))
        vocabulary = {f: i for i, f in enumerate(feature_names)}

        setattr(self, "feature_names_", feature_names)
        setattr(self, "vocabulary_", vocabulary)
        return self

    def transform(self, X):
        X_transform = np.zeros((len(X), len(self.feature_names_)))

        for i, x in enumerate(X):
            for k, v in x.items():
                j = self.vocabulary_[self._get_feature_name(k, v)]
                if isinstance(v, numbers.Number):
                    X_transform[i, j] = v
                else:
                    X_transform[i, j] = 1

        return X_transform

    def _get_feature_name(self, feature_name, feature_value):
        if isinstance(feature_value, numbers.Number):
            return feature_name
        else:
            return "{}{}{}".format(feature_name, self.separator, feature_value)


class TfidfTransformer(BaseTransformer):
    NORM_TO_ORD = {"l1": 1, "l2": 2}

    def __init__(self, norm="l2", use_idf=True, smooth_idf=True):
        self.ord = TfidfTransformer.NORM_TO_ORD[norm]
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf

    def fit(self, X):
        """
        Args:
            X (np.ndarray): shape = (n_documents, n_terms)
        """
        if self.use_idf:
            n_documents = X.shape[0]

            df = (X != 0).sum(axis=0)
            if self.smooth_idf:
                idf = np.log((1 + n_documents) / (1 + df)) + 1
            else:
                idf = np.log(n_documents / df) + 1

            setattr(self, "idf_", idf)
        return self

    def transform(self, X):
        if self.use_idf:
            X = X * self.idf_

        norm = np.linalg.norm(X, ord=self.ord, axis=1,keepdims=True)
        return X / norm
