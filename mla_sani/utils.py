""""""

import numpy as np
from numpy import exp, sqrt, pi, power, log
from numpy.linalg import det, inv



class CrossEntropy(object):
    def loss(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    
    def dloss(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p - (1 - y) / (1 - p))
    
# class GeneralizedCrossEntropy(object):
#     def loss(self, y, p, labels=None):
#         transformer = LabelBinarizer()
#         if labels is None:
#             transformer.fit(y)
#         else:
#             transformer.fit(labels)
#         y = transformer.transform(y)

#         if len(transformer.classes_) == 2:
#             y = np.hstack([1 - y, y])

#         p = np.clip(p, 1e-15, 1 - 1e-15)
#         p /= p.sum(axis=1, keepdims=True)
#         return -np.sum(y * np.log(p), axis=1)
    
#     def dloss(self):
#         raise NotImplementedError()
 
class SquareLoss(object):
    def loss(self, y, p):
        return (y - p) ** 2 / 2
    
    def dloss(self, y, p):
        return p - y

class Sigmoid(object):
    def __call__(self, x):
        # avoid overflow or underflow
        x = np.clip(x, -100, 100)
        return 1 / (1 + np.exp(-x))
    
    def derivative(self, ｘ):
        return self(x) * (1 - self(x))

sigmoid = Sigmoid()
    
class Softmax(object):
    def __call__(self, z, axis=-1):
        return np.exp(z) / np.sum(np.exp(z), axis=-1)
    
    def derivative(self, x):
        return self(x) * (1 - self(x))
    
class Identity(object):
    def __call__(self, x):
        return x
    
    def derivative(self, x):
        return 1

class NormalDistribution(object):
    def __init__(self, mean ,var):
        self.mean = mean
        self.var = var
        
    def pdf(self, X):
        return exp(-(X - self.mean)**2 / (2 * self.var**2)) / sqrt(2 * pi * self.var**2)
    
class MultivariateNormalDistribution(object):
    def __init__(self, mean, cov):
        self.mean = mean 
        self.cov = cov
        
    def pdf(self, X):
        X = X - self.mean
        n = len(self.mean)

        return exp((X.dot(inv(self.cov)) * X).sum(-1) / -2) / (power(2 * pi, n / 2) * sqrt(det(self.cov)))
    
def pairwise_euclidean_distance(X, Y):
    """Compute pairwise euclidean distance."""
    return np.sqrt(np.power(X[:, None, :] - Y, 2).sum(axis=-1))

def covariance_matrix(X):
    """Compute convariance matrix."""
    X = X - X.mean(axis=1, keepdims=True)
    return X.dot(X.T) / (X.shape[1] - 1)

def most_frequent(arr):
    unique, count = np.unique(arr, return_counts=True)
    return unique[count.argmax()]
