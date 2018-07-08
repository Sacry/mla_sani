from abc import ABC, abstractmethod


class BaseTransformer(ABC):
    def fit(self, X):
        return self

    @abstractmethod
    def transform(self, X):
        pass

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        raise NotImplementedError()

class BaseFold(ABC):
    @abstractmethod
    def split(self, X, y=None, groups=None):
        pass

