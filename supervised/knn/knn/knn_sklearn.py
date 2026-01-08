"""
Scikit-learn based implementation of the K-Nearest Neighbors (KNN) classifier.

This module wraps `sklearn.neighbors.KNeighborsClassifier` to provide a
consistent API with the pure and optimized KNN implementations, and serves
as a highly optimized reference baseline.
"""

import numpy as np
from typing import Sequence, Tuple, List
from sklearn.neighbors import KNeighborsClassifier


class KNN:
    """
    A wrapper implementation of the k-Nearest Neighbors (k-NN) classifier
    using scikit-learn.

    This class provides a simplified interface around
    `sklearn.neighbors.KNeighborsClassifier`, supporting configurable
    distance metrics and optional distance-weighted voting.
    """

    def __init__(self, k: int = 3, metric: str = 'euclidean', weighted: bool = True) -> None:
        """
        Initialize the KNN classifier.

        Args:
            k: Number of nearest neighbors to consider.
            metric: Distance metric to use (e.g., 'euclidean', 'manhattan').
            weighted: Whether to weight votes by inverse distance.
        """
        
        # Choose voting strategy for scikit-learn
        weights = 'distance' if weighted else 'uniform'

        # Initialize the underlying scikit-learn KNN model
        self.model = KNeighborsClassifier(
            n_neighbors = k,
            metric = metric,
            weights = weights
        )
    
    def fit(self, X_train: Sequence[Tuple[float, ...]], y_train: Sequence[str]) -> None:
        """
        Fit the KNN classifier using training data.

        Args:
            X_train: Training feature vectors.
            y_train: Corresponding class labels.
        """

        # Convert input to NumPy array for scikit-learn compatibility
        self.model.fit(np.asarray(X_train), y_train)
    
    def predict(self, X_query: Tuple[float, ...]) -> str:
        """
        Predict the class label for a single query point.

        Args:
            X_query: Feature vector of the query point.

        Returns:
            The predicted class label.
        """

        # Reshape query to 2D array as required by scikit-learn
        X_query = np.asarray(X_query).reshape(1, -1)
        return self.model.predict(X_query)[0]
    
    def predict_batch(self, X_test: Sequence[Tuple[float, ...]]) -> List[str]:
        """
        Predict class labels for multiple query points.

        Args:
            X_test: A sequence of feature vectors.

        Returns:
            A list of predicted class labels.
        """

        # Predict labels for all query points and convert to a Python list
        return self.model.predict(np.asarray(X_test)).tolist()
