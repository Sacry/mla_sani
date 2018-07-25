""""""

import numpy as np
from numpy import exp, sqrt, pi, power, log
from numpy.linalg import det, inv

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
