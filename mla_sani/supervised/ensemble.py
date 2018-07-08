from copy import deepcopy

import numpy as np
# my own tree doesn't support sample weights
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class BaggingClassifier(object):
    """A SaNI of bagging classifier.

    Args:
        base_estimator: Base estimator. If none, DecisionTreeClassifier is used.
        n_estimators (int): Number of base estimators.
        max_samples (float): Max percentage of samples to train each estimator.
        max_features (flaot): Max percentage of features to train each estimator.
        bootstrap (boolean): Whether samples are drawn with replacement.
        bootstrap_features (boolean): Whether features are drawn with replacement.
    """
    def __init__(self, 
                 base_estimator=None, 
                 n_estimators=10, 
                 max_samples=1, 
                 max_features=1, 
                 bootstrap=True,
                 bootstrap_features=False):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features

    def fit(self, X, y):
        """Fit model.

        Args:
            X (np.ndarray): shape = (n_samples, n_features). Training data.
            y (np.ndarray): shape = (n_samples,). Target values.
        """
        n_samples, n_features = X.shape

        setattr(self, "classes_", np.unique(y))

        setattr(self, "n_classes_", len(self.classes_))

        if self.base_estimator is None:
            self.base_estimator = DecisionTreeClassifier()
        setattr(self, "base_estimator_", self.base_estimator)

        estimators = [deepcopy(self.base_estimator) for _ in range(self.n_estimators)]
        setattr(self, "estimators_", estimators)
        
        # draw samples for each estimator
        n_estimator_samples = int(n_samples * self.max_samples)
        estimators_samples = np.array([
            self._draw(self.bootstrap, n_samples, n_estimator_samples)
            for _ in range(self.n_estimators)
        ])
        setattr(self, "estimators_samples_", np.array(estimators_samples))
        
        # draw features for each estimator
        n_estimator_features = int(n_features * self.max_features)
        estimators_features = np.array([
            self._draw(self.bootstrap_features, n_features, n_estimator_features)
            for _ in range(self.n_estimators)
        ])
        setattr(self, "estimators_features_", estimators_features)
        
        # fit estimators
        for i in range(self.n_estimators):
            Xi = X[estimators_samples[i]]
            yi = y[estimators_samples[i]]
            estimators[i].fit(Xi[:, estimators_features[i]], yi)
            
        return self
    
    def _draw(self, bootstrap, n_population, n_sample):
        if bootstrap:
            indices = np.random.randint(0, n_population, n_sample)
        else:
            population_indices = np.arange(n_population)
            np.random.shuffle(population_indices)
            indices = population_indices[:n_sample]
        return indices

    def predict(self, X):
        """Perform predict on X.

        Args:
            X (np.ndarray): shape = (n_samples, n_features). Data to predict.

        Returns:
            np.ndarray: shape = (n_samples,). Predicted result.
        """
        votes = np.zeros((X.shape[0], self.n_classes_))
        for i in range(self.n_estimators):
            y_pred = self.estimators_[i].predict(X[:, self.estimators_features_[i]])
            votes[np.arange(X.shape[0]), y_pred] += 1
        return self.classes_.take(votes.argmax(axis=1))


class AdaBoostClassifier(object):
    """A SaNI of adaptive boosting classfier.

    Args:
        base_estimator: Base estimator. If none, DecisionTreeClassifier(max_depth=1) is used.
        n_estimators (int): Number of base estimators.
        learning_rate (flowt): Learning rate.
    """
    def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1.0):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

    def fit(self, X, y):
        """Fit model.

        Args:
            X (np.ndarray): shape = (n_samples, n_features). Training data.
            y (np.ndarray): shape = (n_samples,). Target values.
        """
        n_samples, _ = X.shape

        setattr(self, "classes_", np.unique(y))

        setattr(self, "n_classes_", len(self.classes_))

        if self.base_estimator is None:
            self.base_estimator = DecisionTreeClassifier(max_depth=1)
        setattr(self, "base_estimator_", self.base_estimator)

        estimators = [deepcopy(self.base_estimator) for _ in range(self.n_estimators)]
        setattr(self, "estimators_", estimators)

        estimator_weights = np.ones(self.n_estimators)
        estimator_errors = np.zeros(self.n_estimators)
        sample_weight = np.full(n_samples, 1 / n_samples)

        for i, clf in enumerate(self.estimators_):
            # adaboost for multi-class classification
            #   https://web.stanford.edu/~hastie/Papers/samme.pdf
            # Algorithm 2 SAMME

            # (a) fit a estimator
            y_pred = clf.fit(X, y, sample_weight=sample_weight).predict(X)

            # (b) compute error
            error = np.average(y != y_pred, weights=sample_weight)

            # (c) compute alpha (estimator weights)
            alpha = self.learning_rate * np.log(1 - error) / error + np.log(self.n_classes_ - 1)

            # (d) update weight
            sample_weight *= np.exp(alpha * (y != y_pred))

            # (e) re-normalize
            sample_weight /= sample_weight.sum()

            estimator_errors[i] = error
            estimator_weights[i] = alpha

        setattr(self, "estimator_weights_", estimator_weights)
        setattr(self, "estimator_errors_", estimator_errors)
        return self

    def predict(self, X):
        """Perform predict on X.

        Args:
            X (np.ndarray): shape = (n_samples, n_features). Data to predict.

        Returns:
            np.ndarray: shape = (n_samples,). Predicted result.
        """
        y_pred = np.sum([
            (clf.predict(X)[:, np.newaxis] == self.classes_) * alpha
            for clf, alpha in zip(self.estimators_, self.estimator_weights_)
        ], axis=0)

        return self.classes_[y_pred.argmax(axis=1)]

class CrossEntropy(object):
    def __call__(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    
    def negative_gradient(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return y / p - (1 - y) / (1 - p)

class GradientBoostingClassifier(object):
    """A SaNI of gradient boosting classfier.

    Note:
        This implementation supports binary classification only.

    Args:
        n_estimators (int): Number of base estimators.
        learning_rate (flowt): Learning rate.
    """
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        
        # self.loss = CrossEntropy()
        from sklearn import ensemble
        self.loss = ensemble.gradient_boosting.BinomialDeviance(2)

    def fit(self, X, y):
        """Fit model.

        Args:
            X (np.ndarray): shape = (n_samples, n_features). Training data.
            y (np.ndarray): shape = (n_samples,). Target values.
        """
        estimators = [deepcopy(DecisionTreeRegressor(max_depth=1)) for _ in range(self.n_estimators)]
        setattr(self, "estimators_", estimators)
        
        train_score = []
        
        y_pred = self.estimators_[0].fit(X, y).predict(X)
        train_score.append(self.loss(y, y_pred).mean())
    
        for i in range(1, len(self.estimators_)):
            gradient = self.loss.negative_gradient(y, y_pred)
            y_pred += self.learning_rate * self.estimators_[i].fit(X, gradient).predict(X)
            train_score.append(self.loss(y, y_pred).mean())
                   
        setattr(self, "train_score_",train_score)
        return self
    
    def predict(self, X):
        """Perform predict on X.

        Args:
            X (np.ndarray): shape = (n_samples, n_features). Data to predict.

        Returns:
            np.ndarray: shape = (n_samples,). Predicted result.
        """
        y_pred = self.estimators_[0].predict(X)
        for i in range(1, len(self.estimators_)):
            y_pred += self.learning_rate * self.estimators_[i].predict(X)        
        return (y_pred >= 0.5).astype(int)


# The following implementation is wrong.
# Google tells me that the difference between RF and bagging DT is that RF searches over a random
# subset of features when selecting a split point, while the best bagging DT can do is to feed only
# a random subset of feature to each DT.
    
# class RandomForestClassifier(object):
#     """A SaNI of random forest classifier."""
#     def __init__(self, 
#                  n_estimators=10, 
#                  max_features=1, 
#                  bootstrap=True,
#                  criterion="gini", 
#                  max_depth=None, 
#                  min_samples_split=2, 
#                  min_impurity_decrease=0.0):
#         self.n_estimators = n_estimators
#         self.max_features = max_features
#         self.bootstrap = bootstrap
        
#         self.criterion = criterion
#         self.max_depth = max_depth
#         self.min_samples_split = min_samples_split
#         self.min_impurity_decrease = min_impurity_decrease
        
#         self._bagging = None
        
#     def fit(self, X, y):
#         base_estimator = DecisionTreeClassifier(criterion=self.criterion,
#                                                 max_depth=self.max_depth,
#                                                 min_samples_split=self.min_samples_split,
#                                                 min_impurity_decrease=self.min_impurity_decrease)
        
#         self._bagging = BaggingClassifier(base_estimator=base_estimator,
#                                           n_estimators=self.n_estimators,
#                                           max_features=self.max_features,
#                                           bootstrap=self.bootstrap)
#         self._bagging.fit(X, y)
#         return self
        
#     def predict(self, X):
#         return self._bagging.predict(X)    
