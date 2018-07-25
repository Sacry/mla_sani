"""
Reference:
    http://ruder.io/optimizing-gradient-descent/
"""

import numpy as np

def decaying_average(alpha, acc, val):
    return alpha * acc + (1 - alpha) * val


class Optimizer(object):
    pass


class SGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.0):
        self.lr = lr
        self.momentum = momentum

        # previous velocity
        self.v = None

    def get_updates(self, params, gradients):
        if self.v is None:
            self.v = np.zeros(params.shape)

        self.v = self.momentum * self.v + self.lr * gradients
        return params - self.v


class Adagrad(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr
        self.eps = 1e-08

        # accumulated past squared gradients
        self.G = None

    def get_updates(self, params, gradients):
        if self.G is None:
            self.G = np.zeros(gradients.shape)

        self.G += gradients ** 2
        return params - self.lr * gradients / np.sqrt(self.G + self.eps)


class AdaDelta(Optimizer):
    def __init__(self, lr=1.0, rho=0.95):
        self.lr = lr  # not actually necessarily
        self.rho = rho
        self.eps = 1e-08

        # decaying average of past squared gradients
        self.Eg = None

        # decaying average of params updates
        self.Edp = None

    def get_updates(self, params, gradients):
        if self.Eg is None:
            self.Eg = np.zeros(gradients.shape)

        if self.Edp is None:
            self.Edp = np.zeros(params.shape)

        self.Eg = decaying_average(self.rho, self.Eg, gradients ** 2)
        dp = np.sqrt(self.Edp + self.eps) / np.sqrt(self.Eg + self.eps) * gradients
        self.Edp = decaying_average(self.rho, self.Edp, dp ** 2)

        return params - self.lr * dp


class RMSprop(Optimizer):
    def __init__(self, lr=0.001, rho=0.9):
        self.lr = lr
        self.rho = rho
        self.eps = 1e-08

        # decaying average of past squared gradients
        self.Eg = None

    def get_updates(self, params, gradients):
        if self.Eg is None:
            self.Eg = np.zeros(gradients.shape)

        self.Eg = decaying_average(self.rho, self.Eg, gradients ** 2)
        return params - self.lr * gradients / np.sqrt(self.Eg + self.eps)


class Adam(Optimizer):
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = 1e-08
        self.amsgrad = amsgrad

        # decaying average of past gradients
        self.m = None
        # decaying average of past squared gradients
        self.v = None
        if self.amsgrad:
            self.v_hat = None

    def get_updates(self, params, gradients):
        if self.m is None:
            self.m = np.zeros(gradients.shape)

        if self.v is None:
            self.v = np.zeros(gradients.shape)

        if self.amsgrad and self.v_hat is None:
            self.v_hat = np.zeros(gradients.shape)

        self.m = decaying_average(self.beta_1, self.m, gradients)
        m_hat = self.m / (1 - self.beta_1)

        self.v = decaying_average(self.beta_2, self.v, gradients ** 2)
        if self.amsgrad:
            self.v_hat = v_hat = np.maximum(self.v_hat, self.v)
        else:
            v_hat = self.v / (1 - self.beta_2)

        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class Adamax(Optimizer):
    def __init__(self, lr=0.002, beta_1=0.9, beta_2=0.999):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = 1e-08

        # decaying average of past gradients
        self.m = None
        #
        self.u = None

    def get_updates(self, params, gradients):
        if self.m is None:
            self.m = np.zeros(gradients.shape)

        if self.u is None:
            self.u = np.zeros(gradients.shape)

        self.m = decaying_average(self.beta_1, self.m, gradients)
        m_hat = self.m / (1 - self.beta_1)

        self.u = np.maximum(self.beta_2 * self.u, np.abs(gradients))

        return params - self.lr * m_hat / self.u
