"""
Optimized K-Means clustering implementation.

This module provides a NumPy-based implementation of the K-Means algorithm,
designed for high performance through vectorized distance computation and
efficient centroid updates.

It follows the standard K-Means procedure while exposing a minimal API
(`fit`, `predict`, `predict_batch`) for fair benchmarking against pure
Python and scikit-learn implementations.
"""

import numpy as np
from typing import Optional, Sequence


class KMeans:
    """
    An optimized implementation of the K-Means clustering algorithm.

    This version leverages NumPy vectorization to efficiently compute
    distances and update centroids, significantly improving performance
    over a pure-Python implementation.
    """

    def __init__(self, n_clusters: int, max_iter: int = 300,
                  tol: float = 1e-4, random_state: Optional[int] = None) -> None:
        """
        Initialize the K-Means clustering model.

        Args:
            n_clusters: Number of clusters to form.
            max_iter: Maximum number of iterations.
            tol: Convergence tolerance based on centroid movement.
            random_state: Seed for centroid initialization.
        """

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self._centroids: Optional[np.ndarray] = None

    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize centroids by randomly sampling points from the dataset.

        Args:
            X: Input data of shape (n_samples, n_features).

        Returns:
            Initialized centroids array.
        """
        
        if self.random_state is not None:
            np.random.seed(self.random_state)

        indices = np.random.choice(len(X), self.n_clusters, replace=False)
        return X[indices].copy()

    @staticmethod
    def _compute_distances(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Compute squared Euclidean distances between samples and centroids.

        Args:
            X: Data points of shape (n_samples, n_features).
            centroids: Centroid array of shape (n_clusters, n_features).

        Returns:
            Distance matrix of shape (n_samples, n_clusters).
        """

        # ||x - c||^2 = ||x||^2 + ||c||^2 - 2x·c
        x_norm = np.sum(X ** 2, axis=1, keepdims=True)
        c_norm = np.sum(centroids ** 2, axis=1)

        return x_norm + c_norm - 2 * X @ centroids.T

    def fit(self, X: Sequence[Sequence[float]]) -> None:
        """
        Fit the K-Means model on the given dataset.

        Args:
            X: Input data as a sequence of feature vectors.

        Raises:
            ValueError: If the dataset is empty or invalid.
        """

        X = np.asarray(X, dtype=float)
    
        if X.ndim != 2 or len(X) == 0:
            raise ValueError("Input data must be a non-empty 2D array.")

        centroids = self._initialize_centroids(X)

        for _ in range(self.max_iter):
            # Assign points to nearest centroid
            distances = self._compute_distances(X, centroids)
            labels = np.argmin(distances, axis=1)

            # Recompute centroids
            new_centroids = np.zeros_like(centroids)
            for k in range(self.n_clusters):
                cluster_points = X[labels == k]

                # Handle empty clusters by keeping previous centroid
                if len(cluster_points) == 0:
                    new_centroids[k] = centroids[k]

                else:
                    new_centroids[k] = cluster_points.mean(axis=0)

            # Check convergence
            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids

            if shift <= self.tol:
                break

        self._centroids = centroids

    def predict(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        """
        Assign each data point to the nearest centroid.

        Args:
            X: Input data points.

        Returns:
            Cluster indices for each data point.

        Raises:
            RuntimeError: If the model has not been fitted.
        """

        if self._centroids is None:
            raise RuntimeError("Model must be fitted before prediction.")
        
        X = np.asarray(X, dtype=float)
        distances = self._compute_distances(X, self._centroids)

        return np.argmin(distances, axis=1)

    def predict_batch(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        """
        Alias for predict, kept for benchmark API consistency.

        Args:
            X: Input data points.

        Returns:
            Cluster indices for each data point.
        """
        return self.predict(X)
    
    @property
    def centroids(self) -> np.ndarray:
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
