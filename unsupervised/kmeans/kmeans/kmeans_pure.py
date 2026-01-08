"""
Pure Python implementation of the K-Means clustering algorithm.

This module provides a minimal, from-scratch K-Means implementation using
only the Python standard library. It is intended for educational purposes
and as a baseline for performance comparison against optimized and
scikit-learn-based implementations.
"""

from math import sqrt
from random import sample, seed
from typing import Optional, Sequence, Tuple, List


class KMeans:
    """
    A pure Python implementation of the K-Means clustering algorithm.

    This implementation uses only Python standard library utilities
    and is intended for educational purposes and benchmarking.
    """

    def __init__(self, n_clusters: int, max_iter: int = 300, tol: float = 1e-4,
                 random_state: Optional[int] = None) -> None:
        """
        Initialize the K-Means clustering model.

        Args:
            n_clusters: Number of clusters to form.
            max_iter: Maximum number of iterations.
            tol: Convergence tolerance based on centroid movement.
            random_state: Optional random seed for reproducibility.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self._centroids: Optional[List[Tuple[float, ...]]] = None

    @staticmethod
    def _euclidean_distance(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
        """
        Compute the Euclidean distance between two points.

        Args:
            a: First point.
            b: Second point.

        Returns:
            The Euclidean distance between a and b.
        """
        # Explicit loop-based distance for pure Python baseline
        return sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    
    def _closest_centroid(self, point: Tuple[float, ...]) -> int:
        """
        Find the index of the closest centroid to a given point.

        Args:
            point: Input data point.

        Returns:
            Index of the nearest centroid.
        """

        distances = [self._euclidean_distance(point, centroid)
                     for centroid in self._centroids]
        
        return distances.index(min(distances))
    
    def fit(self, X: Sequence[Tuple[float, ...]]) -> None:
        """
        Fit the K-Means model on the given dataset.

        Args:
            X: A sequence of feature vectors.

        Raises:
            ValueError: If input data is empty or cluster count is invalid.
        """

        if len(X) == 0:
            raise ValueError("Input data is empty.")    
        
        if self.n_clusters <= 0 or self.n_clusters > len(X):
            raise ValueError("Invalid number of clusters.")

        if self.random_state is not None:
            seed(self.random_state)

        n_features = len(X[0])

        # Initialize centroids by random sampling
        self._centroids = sample(list(X), self.n_clusters)

        for _ in range(self.max_iter):
            clusters = [[] for _ in range(self.n_clusters)]

            # Assignment step
            for point in X:
                idx = self._closest_centroid(point)
                clusters[idx].append(point)
            
            new_centroids = []

            # Update step
            for i, cluster in enumerate(clusters):
                if not cluster:
                    # Reinitialize empty cluster
                    new_centroids.append(self._centroids[i])
                    continue

                centroid = tuple(
                    sum(point[d] for point in cluster) / len(cluster)
                    for d in range(n_features)
                )

                new_centroids.append(centroid)

            # Check convergence
            shift = sum(
                self._euclidean_distance(self._centroids[i], new_centroids[i])
                for i in range(self.n_clusters)
            )
            
            self._centroids = new_centroids

            if shift < self.tol:
                break
            
    def predict(self, X: Sequence[Tuple[float, ...]]) -> List[int]:
        """
        Assign cluster labels to input data points.

        Args:
            X: A sequence of feature vectors.

        Returns:
            A list of cluster indices.

        Raises:
            RuntimeError: If the model has not been fitted.
        """

        if self._centroids is None:
            raise RuntimeError("Model must be fitted before prediction.")

        return [self._closest_centroid(point) for point in X]

    def predict_batch(self, X: Sequence[Tuple[float, ...]]) -> List[int]:
        """
        Alias for predict, kept for benchmark API consistency.

        Args:
            X: Input data points.

        Returns:
            Cluster indices for each data point.
        """
        return self.predict(X)

    @property
    def centroids(self) -> List[Tuple[float, ...]]:
        """
        Get the learned cluster centroids.

        Returns:
            Centroid array of shape (n_clusters, n_features).

        Raises:
            RuntimeError: If the model has not been fitted.
        """

        if self._centroids is None:
            raise RuntimeError("Model has not been fitted yet.")
        
        return self._centroids
