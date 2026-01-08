"""
Scikit-learn based K-Means clustering implementation.

This module provides a thin wrapper around sklearn's `KMeans` class,
exposing a unified API (`fit`, `predict`, `predict_batch`, `centroids`)
consistent with the pure Python and optimized implementations in this
repository.

It is intended for benchmarking and reference comparison rather than
reimplementing the algorithm from scratch.
"""

import numpy as np
from typing import Optional, Sequence
from sklearn.cluster import KMeans as SKMeans


class KMeans:
    """
    Scikit-learn based K-Means clustering wrapper.

    This class wraps sklearn's `KMeans` implementation to provide a unified
    interface compatible with other K-Means implementations in this repository
    (pure Python and optimized versions).

    It is primarily intended for benchmarking and reference comparison.
    """

    def __init__(self, n_clusters: int, init: str = "k-means++", max_iter: int = 300,
                 tol: float = 1e-4, random_state: Optional[int] = None) -> None:
        """
        Initialize the K-Means model.

        Parameters
        ----------
        n_clusters : int
            Number of clusters to form.
        init : str, default="k-means++"
            Method for initializing centroids.
        max_iter : int, default=300
            Maximum number of iterations for a single run.
        tol : float, default=1e-4
            Relative tolerance to declare convergence.
        random_state : Optional[int], default=None
            Seed for random number generation (for reproducibility).
        """ 

        self.model = SKMeans(
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            n_init="auto"
        )
    
    def fit(self, X: Sequence[Sequence[float]]) -> None:
        """
        Compute K-Means clustering on the given dataset.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features).
        """
        self.model.fit(np.asarray(X, dtype=float))
    
    def predict(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        """
        Predict the closest cluster for each sample in X.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Cluster labels for each sample.
        """
        return self.model.predict(np.asarray(X, dtype=float))
    
    def predict_batch(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        """
        Alias for predict, kept for benchmark API consistency.
        """
        return self.predict(X)
    
    @property
    def centroids(self) -> np.ndarray:
        """
        Return the coordinates of cluster centroids.

        Returns
        -------
        np.ndarray
            Array of shape (n_clusters, n_features).
        """
        return self.model.cluster_centers_
