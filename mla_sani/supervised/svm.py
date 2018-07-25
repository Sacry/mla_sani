import numpy as np
from scipy.optimize import minimize

from ..metrics import rbf_kernel

class SVCPrimal(object):
    """A SaNI of svm classification tying to solve the primal problem.

    Note:
        This implementation only supports binary classification with y belongs to {-1, 1}.
        No kernel support, no C support.
    """
    def __init__(self):
        pass

    def fit(self, X, y):
        """Fit model.

        Args:
            X (np.ndarray): shape = (n_samples, n_features). Training data.
            y (np.ndarray): shape = (n_samples,). Target values. y: {-1, 1}
        """
        n_samples, n_features = X.shape

        # primal problem:
        #
        #   min (1/2) * ||w|| ^ 2
        #       s.t. y*(w.dot(x) + b) >= 1
        #
        # matrix form:
        #
        #   min (1/2) * w.T.dot(P).dot(w) + q.dot(w)
        #       s.t. y*(w.dot(x) + b) >= 1

        # =====Objective=====
        P = np.eye(n_features + 1)
        P[-1, -1] = 0

        q = np.zeros(n_features + 1)

        def objective_func(x):
            return 0.5 * x.dot(P).dot(x) + q.dot(x)

        def objective_grad(x):
            return x.dot(P) + q
        # ====================

        # =====Constraint=====
        G = -y[:, np.newaxis] * np.column_stack([X, np.ones(n_samples)])

        h = np.full(n_samples, -1, dtype=np.float64)

        def constraint_func(x):
            return h - G.dot(x)

        def constraint_grad(x):
            return -G
        # ====================

        x0 = np.random.rand(n_features + 1)
        cons = {'type': 'ineq', 'fun': constraint_func, 'jac': constraint_grad}
        res = minimize(objective_func, x0, jac=objective_grad, constraints=cons)
        if not res.success:
            raise ValueError("I don't know what to do next, try some perfect linear-seperable data.")

        setattr(self, "coef_", res.x[:-1])
        setattr(self, "intercept_", res.x[-1])

        # find support vector: data lie on w.dot(x) + b = -1 or w.dot(x) + b = 1
        support = []
        for i, x in enumerate(X):
            y_pred = x.T.dot(self.coef_) + self.intercept_
            if np.isclose(y_pred, -1) or np.isclose(y_pred, 1):
                support.append(i)

        setattr(self, "support_", np.array(support))
        setattr(self, "support_vectors_", X[support])
        setattr(self, "n_support_", len(support))

        return self

    def predict(self, X):
        """Perform predict on X.

        Args:
            X (np.ndarray): shape = (n_samples, n_features). Data to predict.

        Returns:
            np.ndarray: shape = (n_samples,). Predicted result.
        """
        return np.sign(X.dot(self.coef_) + self.intercept_)

class SVCDual(object):
    """A SaNI of svm classification trying to solve the dual problem.

    Note:
        This implementation only supports binary classification with y belongs to {-1, 1}.

    Args:
        kernel (Callable): Kernel function.
        C (float): Penalty term.
    """

    def __init__(self, kernel=rbf_kernel, C=1.0):
        self.kernel = kernel
        self.C = C

    def fit(self, X, y):
        """Fit model.

        Args:
            X (np.ndarray): shape = (n_samples, n_features). Training data.
            y (np.ndarray): shape = (n_samples,). Target values. y: {-1, 1}
        """
        n_samples, n_features = X.shape

        # http://scikit-learn.org/stable/modules/svm.html#svc
        #
        # dual problem:
        #
        #   min (1/2) * a.dot(Q).dot(a) - e.dot(a)
        #       s.t. y.dot(a) = 0
        #            0 <= a <= C

        # =====Objective=====
        P = Q = np.outer(y, y) * self.kernel(X, X)

        e = np.ones(n_samples)
        q = -e

        def objective_func(x):
            return 0.5 * x.T.dot(P).dot(x) + q.dot(x)

        def objective_grad(x):
            return x.T.dot(P) + q
        # ====================

        # =====Constraint(1)=====
        #   0 <= a <= C
        bounds = [(0, self.C) for _ in range(n_samples)]
        # =======================

        # =====Constraint(2)=====
        #  y.dot(a) = 0
        A = y

        b = 0

        def constraint_02_func(x):
            return A.dot(x) + b

        def constraint_02_grad(x):
            return A
        # =======================

        x0 = np.random.rand(n_samples)
        cons = {'type': 'eq', 'fun': constraint_02_func, 'jac': constraint_02_grad}
        res = minimize(objective_func, x0, jac=objective_grad, constraints=cons, bounds=bounds)
        if not res.success:
            raise ValueError("I don't know what to do next, just try some other data.")

        # many of the `a`s are zero, non-zero `a`s define the support vector.
        setattr(self, "support_", np.argwhere(~np.isclose(res.x, 0)).ravel())
        setattr(self, "support_vectors_", X[self.support_])
        setattr(self, "n_support_", len(self.support_))

        setattr(self, "dual_coef_", y[self.support_] * res.x[self.support_])

        # didn't find out how to calculate `b`, taking the mean looks good to me
        y_pred = self._predict(self.dual_coef_, self.support_vectors_, self.support_vectors_)
        setattr(self, "intercept_", (y[self.support_] - y_pred).mean())

        return self

    def predict(self, X):
        """Perform predict on X.

        Args:
            X (np.ndarray): shape = (n_samples, n_features). Data to predict.

        Returns:
            np.ndarray: shape = (n_samples,). Predicted result.
        """
        return np.sign(self._predict(self.dual_coef_, self.support_vectors_, X, self.intercept_))

    def _predict(self, dual_coef, support_vector, X, b=0):
        return np.sum(dual_coef[:, np.newaxis] * self.kernel(support_vector, X), axis=0) + b

SVC = SVCDual
