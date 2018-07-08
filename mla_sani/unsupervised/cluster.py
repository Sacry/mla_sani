import numpy as np

from ..utils import pairwise_euclidean_distance

class KMeans(object):
    """A SaNI k-means clustering implementation.

    Args:
        n_clusters (int): Number of clusters.
        max_iter (int): Maximum number of iteration.
    """

    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        """Fit model.

        Args:
            X (np.ndarray): shape = (n_samples, n_features). Data to cluster.
        """
        # initialize centroids by choosing k samples at random
        centroids = X[np.random.choice(np.arange(X.shape[0]), size=self.n_clusters, replace=False)]

        for n_iter in range(self.max_iter):
            # calculate new centroids
            distances = pairwise_euclidean_distance(X, centroids)
            new_cluster = distances.argmin(axis=1)
            new_centroids = np.array([X[new_cluster == i].mean(axis=0) for i in range(self.n_clusters)])

            # converged
            if np.allclose(centroids, new_centroids):
                break

            centroids = new_centroids

        setattr(self, "inertia_", distances.sum())
        setattr(self, "n_iter_", n_iter)
        setattr(self, "cluster_centers_", centroids)
        setattr(self, "labels_", new_cluster)
        return self

    def predict(self, X):
        """Assign data to closed cluster centroids.

        Args:
            X (np.ndarray): shape = (n_samples, n_features). Data to Predict.
        """
        return pairwise_euclidean_distance(X, self.cluster_centers_).argmin(axis=1)


class DBSCAN(object):
    """A SaNI of density-based spatial clustering of applications with noise.

    Args:
        eps (float): The maximum distance that is considered "reachable".
        min_samples(int): The minimum reachable samples required to be a core point.
    """
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        """Fit model.

        Args:
            X (np.ndarray): shape = (n_samples, n_features). Data to cluster.
        """
        n_samples = X.shape[0]

        distances = pairwise_euclidean_distance(X, X)

        setattr(self, "core_sample_indices_", [])

        # -2 for unvisited, -1 for noise
        cluster = [-2] * n_samples
        cluster_id = 0
        for i in range(n_samples):
            # i is already visited
            if cluster[i] != -2:
                continue

            # mark i as visited (noise)
            cluster[i] = -1

            # i is core sample
            if self._is_core(distances, i):
                self.core_sample_indices_.append(i)
                self._expand_cluster(i, cluster, cluster_id, distances)
                cluster_id += 1

        self.core_sample_indices_ = np.sort(self.core_sample_indices_)
        setattr(self, "components_", X[self.core_sample_indices_])
        setattr(self, "labels_", cluster)
        return self

    def _is_core(self, distances, i):
        return (distances[i] <= self.eps).sum() >= self.min_samples

    def _expand_cluster(self, i, cluster, cluster_id, distances):
        neighbors = np.argwhere(distances[i] <= self.eps).ravel()
        for j in neighbors:
            # unvisited neighbor
            if cluster[j] == -2:
                # mark as visited
                cluster[j] = -1

                # expand recursively
                if self._is_core(distances, j):
                    self.core_sample_indices_.append(j)
                    self._expand_cluster(j, cluster, cluster_id, distances)

            # assign to clsuter
            if cluster[j] == -1:
                cluster[j] = cluster_id
