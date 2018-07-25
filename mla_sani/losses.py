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
    def loss(self, y, p, labels=None):
        transformer = LabelBinarizer()
        if labels is None:
            transformer.fit(y)
        else:
            transformer.fit(labels)
        y = transformer.transform(y)

        if len(transformer.classes_) == 2:
            y = np.hstack([1 - y, y])

        p = np.clip(p, 1e-15, 1 - 1e-15)
        p /= p.sum(axis=1, keepdims=True)
        return -np.sum(y * np.log(p), axis=1)

    def dloss(self):
        raise NotImplementedError()

class SquareLoss(object):
    def loss(self, y, p):
        return (y - p) ** 2 / 2

    def dloss(self, y, p):
        return p - y
