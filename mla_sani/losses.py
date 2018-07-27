import numpy as np

from .preprocessing import LabelBinarizer

class CrossEntropy(object):
    def loss(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    def dloss(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p - (1 - y) / (1 - p))

class CategoricalCrossEntropy(object):
    def __init__(self, labels=None):
        self.labels = labels

    def _transform_yp(self, y, p):
        transformer = LabelBinarizer()
        if self.labels is None:
            transformer.fit(y)
        else:
            transformer.fit(self.labels)
        y = transformer.transform(y)

        p = np.clip(p, 1e-15, 1 - 1e-15)
        p /= p.sum(axis=1, keepdims=True)
        return y, p

    def loss(self, y, p):
        y, p = self._transform_yp(y, p)
        return -np.sum(y * np.log(p), axis=1)

    def dloss(self, y, p):
        y, p = self._transform_yp(y, p)
        return - (y / p - (1 - y) / (1 - p))

class SquareLoss(object):
    def loss(self, y, p):
        return (y - p) ** 2 / 2

    def dloss(self, y, p):
        return p - y
