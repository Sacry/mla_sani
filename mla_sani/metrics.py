import numpy as np

from .utils import pairwise_euclidean_distance
from .preprocessing import LabelEncoder


def make_scorer(score_func):
    def scorer(esitmator, X, y):
        return score_func(y, esitmator.predict(X))
    return scorer

# ------------------------------ Classification metrics ------------------------------

def accuracy_score(y_true, y_pred):
    return (y_true == y_pred).sum() / y_true.size

def hamming_loss(y_true, y_pred):
    return 1 - accuracy_score(y_true, y_pred)

def confusion_matrix(y_true, y_pred, labels=None):
    if labels is None:
        labels = np.union1d(y_true, y_pred)

    enc = LabelEncoder().fit(labels)
    y_true = enc.transform(y_true)
    y_pred = enc.transform(y_pred)

    matrix = np.zeros((labels.size, labels.size), dtype=np.int64)
    np.add.at(matrix, [y_true, y_pred], 1)
    return matrix

def precision_score(y_true, y_pred):
    return y_pred[y_true == y_pred].sum() / y_pred.sum()

def recall_score(y_true, y_pred):
    return y_pred[y_true == y_pred].sum() / y_true.sum()

def fbeta_score(y_true, y_pred, beta):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

def f1_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=1)

def roc_curve(y_true, y_score):
    sorted_index = y_score.argsort()[::-1]
    y_score = y_score[sorted_index]
    y_true = y_true[sorted_index]

    tps = np.cumsum(y_true)
    fps = np.arange(1, len(y_true) + 1) - tps

    return fps / fps[-1], tps / tps[-1], y_score

def roc_auc_score(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return np.trapz(tpr, x=fpr)

# ------------------------------ Regression metrics ------------------------------

def mean_absolute_error(y_true, y_pred):
    return np.abs(y_true - y_pred).mean()

def mean_squared_error(y_true, y_pred):
    return np.power(y_true - y_pred, 2).mean()

def mean_squared_log_error(y_true, y_pred):
    return np.power(np.log(1 + y_true) - np.log(1 + y_pred), 2).mean()

def median_absolute_error(y_true, y_pred):
    return np.median(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    """coefficient of determination"""
    return 1 - np.power(y_true - y_pred, 2).sum() / np.power(y_true - y_true.mean(), 2).sum()

def explained_variance_score(y_true, y_pred):
    """
    Mathematically, this produces the same result as `r2_score` if the
    mean of the error is zero. But I don't know it intuitively.
    """
    return 1 - np.var(y_true - y_pred) / np.var(y_true)

# ------------------------------ Clustering metrics ------------------------------

def contingency_matrix(labels_true, labels_pred):
    """Compute contingency table.

    Returns:
        np.ndarray: shape = (n_classes, n_clusters)

        |      | Y_1  | Y_2  | ... | Y_s  |
        |------|------|------|-----|------|
        | X_1  | n_11 | n_12 | ... | n_1s |
        | X_2  | n_21 | n_22 | ... | n_2s |
        | ...  | ...  | ...  | ... | ...  |
        | X_r  | n_r1 | n_r2 | ... | n_rs |
    """
    classes = np.unique(labels_true)
    labels_true = LabelEncoder().fit(classes).transform(labels_true)

    clusters = np.unique(labels_pred)
    labels_pred = LabelEncoder().fit(clusters).transform(labels_pred)

    matrix = np.zeros((classes.size, clusters.size), dtype=np.int64)
    np.add.at(matrix, [labels_true, labels_pred], 1)
    return matrix

def comb2(a):
    return a * (a-1) / 2

def adjusted_rand_score(labels_true, labels_pred):
    """Compute adjusted rand index.

    Rand score is like `accuracy_score` for classification, except
    that it does not check on one single data point but two at a
    time. For all combinations of two data points, check if they're
    correctly assigned to a single cluster or two different clusters.

    Adjusted rand score further ensures that a random result get a score of 0.
    """
    # |      | Y_1  | Y_2  | ... | Y_s  | Sums |
    # |------|------|------|-----|------|------|
    # | X_1  | n_11 | n_12 | ... | n_1s | a_1  |
    # | X_2  | n_21 | n_22 | ... | n_2s | a_2  |
    # | ...  | ...  | ...  | ... | ...  | ...  |
    # | X_r  | n_r1 | n_r2 | ... | n_rs | a_r  |
    # |------|------|------|-----|------|------|
    # | Sums | b1   | b2   | ... | b_s  |      |
    contingency = contingency_matrix(labels_true, labels_pred)
    suma = contingency.sum(axis=0)
    sumb = contingency.sum(axis=1)

    # patch
    if contingency.shape[0] == contingency.shape[1] == labels_true.size:
        return 1.0

    RI = comb2(contingency).sum()

    suma_comb2 = comb2(suma).sum()
    sumb_comb2 = comb2(sumb).sum()
    Expected_RI = suma_comb2 * sumb_comb2 / comb2(labels_true.size)

    max_RI = (suma_comb2 + sumb_comb2) / 2

    return (RI - Expected_RI) / (max_RI - Expected_RI)

def safe_log(p):
    """Avoid log(0)."""
    return np.log(p + 1e-15)

def mutual_info_score(labels_true, labels_pred, contingency=None):
    """Compute mutual information."""
    n_samples = labels_true.size
    if contingency is None:
        contingency = contingency_matrix(labels_true, labels_pred)

    pi = contingency.sum(axis=0) / n_samples
    pj = contingency.sum(axis=1, keepdims=True) / n_samples
    pij = contingency / n_samples

    return np.sum(pij * (safe_log(pij) - safe_log(pi) - safe_log(pj)))

def normalized_mutual_info_score(labels_true, labels_pred):
    n_samples = labels_true.size
    contingency = contingency_matrix(labels_true, labels_pred)

    pi = contingency.sum(axis=0) / n_samples
    Hi = -np.sum(pi * safe_log(pi))

    pj = contingency.sum(axis=1) / n_samples
    Hj = -np.sum(pj * safe_log(pj))

    return mutual_info_score(labels_true, labels_pred, contingency) / np.sqrt(Hi * Hj)

def expected_mutual_info(contingency):
    N = contingency.sum()
    a = contingency.sum(axis=0)
    b = contingency.sum(axis=1, keepdims=True)

    nij = np.arange(1, max(a.max(), b.max())+1).reshape(-1, 1, 1)
    mask = nij <= np.minimum(a, b)

    nij = nij.clip(max=np.minimum(a, b))
    term1 = nij / N
    term2 = safe_log(nij * N / a / b)

    # from math import factorial
    # f = np.vectorize(factorial)
    # term3 = (f(a) * f(b) * f(N-a) * f(N-b)) / (f(N) * f(nij) * f(a-nij) * f(b-nij) * f(N-a-b+nij))

    # code above is a direct translation of the formular from scikit-learn documents,
    # but factorial blowup really fast, so calculate in log space instead
    f = np.vectorize(lambda e: np.log(np.arange(e) + 1).sum())
    term3 = np.exp(f(a) + f(b) + f(N-a) + f(N-b) - f(N) - f(nij) - f(a-nij) - f(b-nij) - f(N-a-b+nij))

    term_mul = term1 * term2 * term3
    return term_mul[mask].sum()

def adjusted_mutual_info_score(labels_true, labels_pred):
    n_samples = labels_true.size
    contingency = contingency_matrix(labels_true, labels_pred)

    pi = contingency.sum(axis=0) / n_samples
    Hi = -np.sum(pi * safe_log(pi))

    pj = contingency.sum(axis=1) / n_samples
    Hj = -np.sum(pj * safe_log(pj))

    MI = mutual_info_score(labels_true, labels_pred, contingency)

    EMI = expected_mutual_info(contingency)

    return (MI - EMI) / (max(Hi, Hj) - EMI)

def homogeneity_completeness_v_measure(labels_true, labels_pred):
    """Compute homogeneity, completeness and V-measure.

    Intuition:
        * homogeneity: does setosa cluster contains virginica or versicolor?
        * completeness: are all setosas clustered to a single setosa cluster?
        * V-measure: harmonic mean of homogeneity and completeness
    """
    n_samples = labels_true.shape[0]
    contingency = contingency_matrix(labels_true, labels_pred)

    pk = contingency.sum(axis=0) / n_samples
    pc = contingency.sum(axis=1, keepdims=True) / n_samples
    pck = contingency / n_samples

    # conditional entropy of the classes given the cluster assignments
    Hck = -np.sum(pck * (safe_log(pck) - safe_log(pk)))
    # entropy of the classes
    Hc = -np.sum(pc * safe_log(pc))

    # conditional entropy of clusters given class
    Hkc = -np.sum(pck * (safe_log(pck) - safe_log(pc)))
    # entropy of the clusters
    Hk = -np.sum(pk * safe_log(pk))

    h = 1 - Hck / Hc
    c = 1 - Hkc / Hk
    v = 2 * h * c / (h + c)
    return h, c, v

def homogeneity_score(labels_true, labels_pred):
    h, _, _ = homogeneity_completeness_v_measure(labels_true, labels_pred)
    return h

def completeness_score(labels_true, labels_pred):
    _, c, _ = homogeneity_completeness_v_measure(labels_true, labels_pred)
    return c

def v_measure_score(labels_true, labels_pred):
    _, _, v = homogeneity_completeness_v_measure(labels_true, labels_pred)
    return v

def silhouette_score(X, labels):
    """Compute silhouette coefficient.

    Intuition: If a data point is assigned to clsuter A, but is
    "coloser" to cluster B, then it might be assigned to a wrong
    cluster an will have a low silhouette score.
    """
    distances = pairwise_euclidean_distance(X, X)
    clusters = np.unique(labels)

    scores = []
    for cur_cluster in clusters:
        cur_distances = distances[labels == cur_cluster]

        if cur_distances.shape[0] == 1:
            scores.append(0)
            continue

        # a: The mean distance between a sample and all other points in the same class.
        a = cur_distances[:, labels == cur_cluster].sum(axis=1) / (cur_distances.shape[0] - 1)

        # bs: The mean distance between a sample and all other points of all other clusters.
        bs = []
        for other_cluster in clusters:
            if cur_cluster != other_cluster:
                bs.append(cur_distances[:, labels == other_cluster].mean(axis=1))

        # b: The mean distance between a sample and all other points in the next nearest cluster.
        b = np.column_stack(bs).min(axis=1)

        # silhouette score per sample
        scores.append((b - a) / np.maximum(a, b))

    # overall silhouette score
    return np.concatenate(scores).mean()

def fowlkes_mallows_score(labels_true, labels_pred):
    """Compute fowlkes mallows score.

        FM = TP / sqrt((TP + FP) * (TP + FN))
    """
    n_samples = labels_true.shape[0]
    contingency = contingency_matrix(labels_true, labels_pred)

    Tk = (contingency**2).sum() - n_samples
    Pk = (contingency.sum(axis=0)**2).sum() - n_samples
    Qk = (contingency.sum(axis=1)**2).sum() - n_samples
    return Tk / np.sqrt(Pk * Qk)

def calinski_harabaz_score(X, labels):
    """Compute calinski harabaz score.

    Intuition: It measures the "between-cluster scatter" over
    "within-cluster scatter". If all cluster are dense and far
    away from each other, the score would be high.
    """
    overall_mean = X.mean(axis=0)
    clusters = np.unique(labels)

    Wk, Bk = 0, 0
    for cluster in clusters:
        X_cluster = X[labels == cluster]
        cluster_mean = X_cluster.mean(axis=0)
        Wk += np.power(X_cluster - cluster_mean, 2).sum()
        Bk += np.power(cluster_mean - overall_mean, 2).sum() * X_cluster.shape[0]

    if Bk == 0:
        return 1

    N, k = X.shape[0], clusters.shape[0]
    return Bk * (N - k) / (Wk * (k - 1))

# ------------------------------ kernels ------------------------------

def cosine_similarity(X, Y):
    X_norm = np.linalg.norm(X, axis=1)
    Y_norm = np.linalg.norm(Y, axis=1)
    return np.inner(X, Y) /  (X_norm[:, None] * Y_norm)

def linear_kernel(X, Y):
    return np.inner(X, Y)

def polynomial_kernel(X, Y, degree=3, gamma=None, coef0=1):
    if gamma is None:
        gamma = 1 / X.shape[1]

    return (gamma * np.inner(X, Y) + coef0) ** degree

def sigmoid_kernel(X, Y, gamma=None, coef0=1):
    if gamma is None:
        gamma = 1 / X.shape[1]

    return np.tanh(gamma * np.inner(X, Y) + coef0)

def rbf_kernel(X, Y, gamma=None):
    if gamma is None:
        gamma = 1 / X.shape[1]

    X = X[:, np.newaxis, :]
    return np.exp(-gamma * np.linalg.norm(X - Y, axis=-1) ** 2)

def chi2_kernel(X, Y, gamma=1.0):
    X = X[:, np.newaxis, :]
    return np.exp(-gamma * np.sum((X - Y) ** 2 / (X + Y), axis=-1))
