from abc import ABC, abstractmethod

import numpy as np

from ..utils import most_frequent

class BaseDecisionTree(ABC):
    """Super class of DecisionTreeClassifire and DecisionTreeRegressor.

    Intuition:
        * There are several ways to measure how messy the data is.
        * If you split data into different parts, each part could become much less messy. For example,
            - split [0, 1, 0, 1, 0, 1] into [0, 0, 0], [1, 1, 1] looks pretty neat
            - split [0, 1, 0, 1, 0, 1] into [0, 0, 1], [1, 1, 0] looks not bad but not as good as above.
          Decision tree tries to find the best split by measuring the decrease of mess.
        * One data point per partition is the purest split you could get, but it doesn't make much sense.

    Args:
        criterion (Criterion): The way to measure the uncertainty.
        max_depth (int): Maximum depth of tree.
        min_samples_split (int): Minimum number of samples required to preform split.
        min_impurity_decrease (float): Minimum impurity decrease required to preform split.
    """
    def __init__(self, criterion, max_depth=None, min_samples_split=2, min_impurity_decrease=0.0):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease

    def fit(self, X, y):
        """Fit model.

        Args:
            X (np.ndarray): shape = (n_samples, n_features). Training data.
            y (np.ndarray): shape = (n_samples,). Target values.
        """
        setattr(self, "tree_", self._build_tree(X, y))
        return self

    def _build_tree(self, X, y, depth=0):
        n_node_samples = X.shape[0]

        # current impurity
        impurity = self.criterion(y)

        # value that node will predict if it's leaf
        value = self.criterion.node_value(y)

        if (n_node_samples < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)):
            return DecisionTreeNode(impurity, n_node_samples, value)

        best_impurity_decrease = 0
        best_decsition = None

        # loop through all features and all unique values of each feature to find the best split
        for i_feature in range(X.shape[1]):
            for threshold in np.unique(X[:, i_feature]):
                decision = Decision(i_feature, threshold)
                idx1, idx2 = decision.split(X)
                if idx1.size == 0 or idx2.size == 0:
                    continue

                impurity_decrease = self.criterion.impurity_decrease(impurity, y[idx1], y[idx2])
                if impurity_decrease > best_impurity_decrease:
                    best_impurity_decrease = impurity_decrease
                    best_decsition = decision

        if best_impurity_decrease <= self.min_impurity_decrease:
            return DecisionTreeNode(impurity, n_node_samples, value)

        idx1, idx2 = best_decsition.split(X)
        true_branch = self._build_tree(X[idx1], y[idx1], depth + 1)
        false_branch = self._build_tree(X[idx2], y[idx2], depth + 1)
        return DecisionTreeNode(impurity, n_node_samples, value,
                                decision=best_decsition,
                                true_branch=true_branch,
                                false_branch=false_branch)

    def predict(self, X):
        """Perform predict on X.

        Args:
            X (np.ndarray): shape = (n_samples, n_features). Data to predict.

        Returns:
            np.ndarray: shape = (n_samples,). Predicted result.
        """
        return np.array([self.tree_.predict(X[i]) for i in range(len(X))])

class DecisionTreeClassifier(BaseDecisionTree):
    """A SaNI of decision tree classifier."""
    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2, min_impurity_decrease=0.0):
        if criterion == "gini":
            criterion = Gini()
        elif criterion == "entropy":
            criterion = Entropy()
        else:
            raise ValueError()
        super(DecisionTreeClassifier, self).__init__(criterion, max_depth, min_samples_split, min_impurity_decrease)


class DecisionTreeRegressor(BaseDecisionTree):
    """A SaNI of decision tree regressor."""
    def __init__(self, criterion="mse", max_depth=None, min_samples_split=2, min_impurity_decrease=0.0):
        if criterion == "mse":
            criterion = MSE()
        else:
            raise ValueError()

        super(DecisionTreeRegressor, self).__init__(criterion, max_depth, min_samples_split, min_impurity_decrease)


class DecisionTreeNode(object):
    """Class represents a decision node.

    Args:
        impurity (float): Impurity of current node.
        n_node_samples (int): Number of samples of current node.
        value (float): Value that node will predict if it's leaf
        decision (Decision): Decision of current node.
        true_branch (DecisionTreeNode):
        false_branch (DecisionTreeNode):
    """
    def __init__(self,
                 impurity,
                 n_node_samples,
                 value,
                 decision=None,
                 true_branch=None,
                 false_branch=None):
        self.impurity = impurity
        self.n_node_samples = n_node_samples
        self.value = value

        # decision node only
        self.decision = decision
        self.true_branch = true_branch
        self.false_branch = false_branch

    def is_decision_node(self):
        return self.decision is not None

    def predict(self, X):
        if not self.is_decision_node():
            return self.value

        if self.decision.is_true(X):
            return self.true_branch.predict(X)
        else:
            return self.false_branch.predict(X)

class Decision(object):
    def __init__(self, i_feature, threshold):
        self.i_feature = i_feature
        self.threshold = threshold

    def split(self, X):
        mask = X[:, self.i_feature] <= self.threshold
        return np.argwhere(mask).ravel(), np.argwhere(~mask).ravel()

    def is_true(self, X):
        return X[self.i_feature] <= self.threshold

    def __str__(self):
        return "X[{}] <= {}".format(self.i_feature, self.threshold)


class Criterion(ABC):
    @abstractmethod
    def __call__(self, y): pass

    def impurity_decrease(self, impurity, y1, y2):
        """Compute the impurity descrease/information gain/variance decrease.

        Coincidently (or may be not), gini/entropy/mse share the same formula.
        """
        p = len(y1) / (len(y1) + len(y2))
        return impurity - p * self(y1) - (1 - p) * self(y2)

    @abstractmethod
    def node_value(self, y): pass

class Gini(Criterion):
    def __call__(self, y):
        p = np.array([(y == c).sum() / y.size for c in np.unique(y)])
        return np.sum(p * (1 - p))

    def node_value(self, y):
        return most_frequent(y)

class Entropy(Criterion):
    def __call__(self, y):
        p = np.array([(y == c).sum() / y.size for c in np.unique(y)])
        return -np.sum(p * np.log2(p))

    def node_value(self, y):
        return most_frequent(y)

class MSE(Criterion):
    def __call__(self, y):
        return np.mean(np.power(y - y.mean(), 2))

    def node_value(self, y):
        return y.mean()
