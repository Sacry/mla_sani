from unittest import TestCase

import numpy as np
from sklearn import metrics as sklearn_metrics
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification

from mla_sani import metrics


class TestClassificationMetrics(TestCase):
    def _test_score(self, func_test, func_sklearn):
        for _ in range(10):
            y_true = np.random.randint(0, 10, 100)
            y_pred = np.random.randint(0, 10, 100)
            self.assertAlmostEqual(func_test(y_true, y_pred),
                                   func_sklearn(y_true, y_pred))

    def _test_binary(self, func_test, func_sklearn):
        for _ in range(10):
            y_true = np.random.randint(0, 2, 100)
            y_pred = np.random.randint(0, 2, 100)
            self.assertAlmostEqual(func_test(y_true, y_pred),
                                   func_sklearn(y_true, y_pred))

    def test_accuracy_score(self):
        self._test_score(metrics.accuracy_score, sklearn_metrics.accuracy_score)

    def test_hamming_loss(self):
        self._test_score(metrics.hamming_loss, sklearn_metrics.hamming_loss)

    def test_confusion_matrix(self):
        for _ in range(10):
            y_true = np.random.randint(0, 10, 100)
            y_pred = np.random.randint(0, 10, 100)
            self.assertTrue(np.allclose(metrics.confusion_matrix(y_true, y_pred),
                                        sklearn_metrics.confusion_matrix(y_true, y_pred)))

    def test_confusion_matrix_with_labels(self):
        for _ in range(10):
            labels = np.array(list("abcd"))
            y_true = np.random.choice(labels, 100)
            y_pred = np.random.choice(labels, 100)
            self.assertTrue(np.allclose(metrics.confusion_matrix(y_true, y_pred, labels=labels),
                                        sklearn_metrics.confusion_matrix(y_true, y_pred, labels=labels)))

    def test_precision_score(self):
        self._test_binary(metrics.precision_score, sklearn_metrics.precision_score)

    def test_recall_score(self):
        self._test_binary(metrics.recall_score, sklearn_metrics.recall_score)

    def test_f1_score(self):
        self._test_binary(metrics.f1_score, sklearn_metrics.f1_score)

    def test_fbeta_score(self):
        for _ in range(10):
            y_true = np.random.randint(0, 2, 100)
            y_pred = np.random.randint(0, 2, 100)
            beta = np.random.random()
            self.assertAlmostEqual(metrics.fbeta_score(y_true, y_pred, beta),
                                   sklearn_metrics.fbeta_score(y_true, y_pred, beta))

    def test_roc_curve(self):
        for _ in range(10):
            y_true = np.random.randint(0, 2, 100)
            y_pred = np.random.rand(100)
            fpr, tpr, thr = metrics.roc_curve(y_true, y_pred)
            fpr2, tpr2, thr2 = sklearn_metrics.roc_curve(y_true, y_pred, drop_intermediate=False)
            self.assertTrue(np.allclose(fpr, fpr2))
            self.assertTrue(np.allclose(tpr, tpr2))
            self.assertTrue(np.allclose(thr, thr2))

    def test_roc_curve(self):
        for _ in range(10):
            y_true = np.random.randint(0, 2, 100)
            y_pred = np.random.rand(100)
            self.assertAlmostEqual(metrics.roc_auc_score(y_true, y_pred),
                                   sklearn_metrics.roc_auc_score(y_true, y_pred))


class TestRegressionMetrics(TestCase):
    def _test(self, func_test, func_sklearn):
        for _ in range(10):
            y_true = np.random.random(100)
            y_pred = np.random.random(100)
            self.assertAlmostEqual(func_test(y_true, y_pred),
                                   func_sklearn(y_true, y_pred))

    def test_explained_variance_score(self):
        self._test(metrics.explained_variance_score, sklearn_metrics.explained_variance_score)

    def test_mean_absolute_error(self):
        self._test(metrics.mean_absolute_error, sklearn_metrics.mean_absolute_error)

    def test_mean_squared_error(self):
        self._test(metrics.mean_squared_error, sklearn_metrics.mean_squared_error)

    def test_mean_squared_log_error(self):
        self._test(metrics.mean_squared_log_error, sklearn_metrics.mean_squared_log_error)

    def test_median_absolute_error(self):
        self._test(metrics.median_absolute_error, sklearn_metrics.median_absolute_error)

    def test_r2_score(self):
        self._test(metrics.r2_score, sklearn_metrics.r2_score)


class TestClusteringMetrics(TestCase):
    def _test(self, func_test, func_sklearn):
        for _ in range(10):
            label_true = np.random.choice(list("abcd"), 100)
            label_pred = np.random.choice(list("abcef"), 100)
            self.assertTrue(np.allclose(func_test(label_true, label_pred), func_sklearn(label_true, label_pred)))

    def test_contigency_matrix(self):
        self._test(metrics.contingency_matrix, sklearn_metrics.cluster.supervised.contingency_matrix)

    def test_adjusted_rand_score(self):
        self._test(metrics.adjusted_rand_score, sklearn_metrics.adjusted_rand_score)

    def test_adjusted_mutual_info_score(self):
        self._test(metrics.adjusted_mutual_info_score, sklearn_metrics.adjusted_mutual_info_score)

    def test_homogeneity_score(self):
        self._test(metrics.homogeneity_score, sklearn_metrics.homogeneity_score)

    def test_homogeneity_score(self):
        self._test(metrics.homogeneity_score, sklearn_metrics.homogeneity_score)

    def test_v_meature_score(self):
        self._test(metrics.v_measure_score, sklearn_metrics.v_measure_score)

    def test_completeness__score(self):
        self._test(metrics.completeness_score, sklearn_metrics.completeness_score)

    def test_silhouette_score(self):
        X, y = make_classification(n_classes=3, n_informative=10)
        y_pred = KMeans(n_clusters=3).fit_predict(X)
        self.assertAlmostEqual(metrics.silhouette_score(X, y),
                               sklearn_metrics.silhouette_score(X, y))
        self.assertAlmostEqual(metrics.silhouette_score(X, y_pred),
                               sklearn_metrics.silhouette_score(X, y_pred))

    def test_fowlkes_mallows_score(self):
        self._test(metrics.fowlkes_mallows_score, sklearn_metrics.fowlkes_mallows_score)

    def test_calinski_harabaz_score(self):
        X, y = make_classification(n_classes=3, n_informative=10)
        y_pred = KMeans(n_clusters=3).fit_predict(X)
        self.assertAlmostEqual(metrics.calinski_harabaz_score(X, y),
                               sklearn_metrics.calinski_harabaz_score(X, y))
        self.assertAlmostEqual(metrics.calinski_harabaz_score(X, y_pred),
                               sklearn_metrics.calinski_harabaz_score(X, y_pred))


class TestKernelMetrics(TestCase):
    def _test(self, func_test, func_sklearn):
        for _ in range(10):
            X = np.random.random((100, 10))
            Y = np.random.random((20, 10))
            self.assertTrue(np.allclose(func_test(X, Y), func_sklearn(X, Y)))

    def test_cosine_similarity(self):
        self._test(metrics.cosine_similarity, sklearn_metrics.pairwise.cosine_similarity)

    def test_linear_kernel(self):
        self._test(metrics.linear_kernel, sklearn_metrics.pairwise.linear_kernel)

    def test_polynomial_kernel(self):
        self._test(metrics.polynomial_kernel, sklearn_metrics.pairwise.polynomial_kernel)

    def test_sigmoid_kernel(self):
        self._test(metrics.sigmoid_kernel, sklearn_metrics.pairwise.sigmoid_kernel)

    def test_rbf_kernel(self):
        self._test(metrics.rbf_kernel, sklearn_metrics.pairwise.rbf_kernel)

    def test_chi2_kernel(self):
        self._test(metrics.chi2_kernel, sklearn_metrics.pairwise.chi2_kernel)
