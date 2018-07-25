import numpy as np

class Identity(object):
    def __call__(self, x):
        return x

    def derivative(self, x):
        return 1

class Sigmoid(object):
    def __call__(self, x):
        x = np.clip(x, -100, 100)
        return 1 / (1 + np.exp(-x))

    def derivative(self, ｘ):
        return self(x) * (1 - self(x))

sigmoid = Sigmoid()

class Softmax(object):
    def __call__(self, x, axis=-1):
        return np.exp(x) / np.sum(np.exp(x), axis=-1)

    def derivative(self, x):
        return self(x) * (1 - self(x))

class TanH(object):
    def __call__(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - self(x) ** 2

class ELU(object):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def __call__(self, x):
        x = np.clip(x, -100, 100)
        return np.where(x >= 0, x, self.alpha * (np.exp(x) - 1))

    def derivative(self, x):
        x = np.clip(x, -100, 100)
        return np.where(x >= 0, 1, self.alpha * np.exp(x))

class SoftPlus(object):
    def __call__(self, x):
        x = np.clip(x, -100, 100)
        return np.log(1 + np.exp(x))

    def derivative(self, x):
        x = np.clip(x, -100, 100)
        return sigmoid(x)
