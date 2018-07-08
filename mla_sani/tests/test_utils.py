from unittest import TestCase

import numpy as np
from sklearn import metrics as sklearn_metrics

from mla_sani import utils

class TestUtils(TestCase):
    def test_pairwise_euclidean_distance(self):
        for _ in range(10):
            X = np.random.random((100, 10))
            Y = np.random.random((20, 10))
            self.assertTrue(np.allclose(utils.pairwise_euclidean_distance(X, Y),
                                        sklearn_metrics.pairwise.euclidean_distances(X, Y)))
