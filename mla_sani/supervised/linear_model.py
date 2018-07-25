from abc import ABC, abstractmethod

import numpy as np

from ..activations import sigmoid

class LinearRegressionLinalg(object):
    """A SaNI linear regression using linear algebra approach.

    Args:
        fit_intercept (boolean): Whether to calculate intercept.
    """
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        """Fit model.

        Args:
            X (np.ndarray): shape = (n_samples, n_features). Training data.
            y (np.ndarray): shape = (n_samples,). Target values.
        """        
        if self.fit_intercept:
            X = np.column_stack([X, np.ones(X.shape[0])])
        else:
            X = np.column_stack([X, np.zeros(X.shape[0])])

        coef = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
        setattr(self, "coef_", coef[:-1])
        setattr(self, "intercept_", coef[-1])
        
        return self

    def predict(self, X):
        return X.dot(self.coef_) + self.intercept_


class LinearRegression(object):
    """A SaNI of lienar regression.

    Args:
        fit_intercept (boolean): Whether to calculate intercept.
        learning_rate (float): Learning rate.
        max_iter (int): Maximum number of iterations.
    """    
    def __init__(self, fit_intercept=True, learning_rate=0.01, max_iter=100):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
    
    def _center_data(self, X, y):
        self._X_mean = X.mean(axis=0)
        self._y_mean = y.mean()
        
        X = X - self._X_mean[np.newaxis, :]
        y = y - self._y_mean
        return X, y
    
    def _calc_intercept(self, intecept):
        if self.fit_intercept:
            intecept += self._y_mean
        
        return intecept
        
    def regularize(self, coef):
        return 0

    def fit(self, X, y):
        """Fit model.

        Args:
            X (np.ndarray): shape = (n_samples, n_features). Training data.
            y (np.ndarray): shape = (n_samples,). Target values.
        """
        if self.fit_intercept:
            X, y = self._center_data(X, y)
        n_samples, n_features = X.shape

        coef = np.zeros(n_features)
        intercept = 0
        for n_iter in range(self.max_iter):
            y_pred = self._predict(X, coef, intercept)

            grad_coef = (y_pred - y).dot(X) / n_samples + self.regularize(coef)
            coef -= self.learning_rate * grad_coef

            if self.fit_intercept:
                grad_intercept = (y_pred - y).mean()
                intercept -= self.learning_rate * grad_intercept
            
        setattr(self, "coef_", coef)
        setattr(self, "intercept_", self._calc_intercept(intercept))
        setattr(self, "n_iter_", n_iter)
        return self     
    
    def predict(self, X):
        """Perform predict on X.

        Args:
            X (np.ndarray): shape = (n_samples, n_features). Data to predict.

        Returns:
            np.ndarray: shape = (n_samples,). Predicted result.
        """
        return self._predict(X, self.coef_, self.intercept_)
    
    def _predict(self, X, coef, intercept):
        return X.dot(coef) + intercept
        
class Ridge(LinearRegression):
    """A SaNI of ridge regression (linear regression with l2 regularization)."""
    def __init__(self, fit_intercept=True, learning_rate=0.01, max_iter=100, alpha=1.0):
        self.alpha = alpha
        super(Ridge, self).__init__(fit_intercept, learning_rate, max_iter)
    
    def regularize(self, coef):
        return self.alpha * coef

class Lasso(LinearRegression):
    """A SaNI of lasso regression (linear regression with l1 regularization)."""    
    def __init__(self, fit_intercept=True, learning_rate=0.01, max_iter=100, alpha=1.0):
        self.alpha = alpha
        super(Lasso, self).__init__(fit_intercept, learning_rate, max_iter)
    
    def regularize(self, coef):
        return self.alpha * np.sign(coef)

class ElasticNet(LinearRegression):
    """A SaNI of lasso regression (linear regression with both l1 and l2 regularization)."""        
    def __init__(self, fit_intercept=True, learning_rate=0.01, max_iter=100, alpha=1.0, l1_ratio=0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        super(ElasticNet, self).__init__(fit_intercept, learning_rate, max_iter)
    
    def regularize(self, coef):
        return self.alpha * (self.l1_ratio * np.sign(coef) + 0.5 * (1 - self.l1_ratio) * coef)
    

class LogisticRegression(LinearRegression):
    """A SaNI of logistic regression.

    Coincidently (or may be not), the cost function of logisitic regression and linear regression
    share the same gradient. So code below simply works.
    """
    def _predict(self, X, coef, intercept):
        return (sigmoid(X.dot(coef) + intercept) >= 0.5).astype(int)

    def predict_proba(self, X):
        return sigmoid(X.dot(self.coef_) + self.intercept_)
