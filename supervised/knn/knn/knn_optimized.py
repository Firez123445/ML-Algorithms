"""
Optimized K-Nearest Neighbors (KNN) classifier.

This implementation uses NumPy and SciPy to accelerate distance computation
and neighbor selection while preserving the standard KNN algorithm behavior.

It is designed for performance benchmarking and comparison against
pure Python and scikit-learn implementations.
"""

import numpy as np
from scipy.spatial.distance import cdist
from typing import Sequence, Optional, Tuple, List, Dict


class KNN:
    """
    An implementation of the k-Nearest Neighbors (k-NN) classifier.

    This classifier supports Euclidean and Manhattan distance metrics and
    optionally uses distance-weighted voting.
    """

    def __init__(self, k: Optional[int] = None, metric: str = 'euclidean', weighted: bool = True) -> None:
        """
        Initialize the KNN classifier.

        Args:
            k: Number of nearest neighbors to consider. If None, k is set
               automatically based on the training data size.
            weighted: Whether to weight votes by inverse distance.
            metric: Distance metric to use ('euclidean' or 'manhattan').
        """

        self.k = k
        self.metric = metric.lower()
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[Sequence[str]] = None

        # Voting function: inverse-distance weighting or uniform voting
        if weighted:
            self.vote_fn = lambda distance: 1 / (distance + 1e-9)
        else:
            self.vote_fn = lambda distance: 1.0
    
    def _compute_distances(self, X_query: np.ndarray) -> np.ndarray:
        """
        Compute distances between a query point and all training samples.

        Args:
            X_query: A single query point as a NumPy array.

        Returns:
            A 1D NumPy array of distances to each training sample.

        Raises:
            ValueError: If an unsupported distance metric is used.
        """

        if self.metric not in ('euclidean', 'manhattan'):
            raise ValueError(f"Unsupported metric: {self.metric}")

        # Compute pairwise distances and flatten the result
        return cdist(self.X_train, [X_query], self.metric).flatten()
    
    def fit(self, X_train: Sequence[Tuple[float, ...]], y_train: Sequence[str]) -> None:
        """
        Fit the KNN classifier using training data.

        Args:
            X_train: Training feature vectors.
            y_train: Corresponding class labels.

        Raises:
            ValueError: If training data is empty, mismatched, or k is invalid.
        """

        if not len(X_train):
            raise ValueError("Training data is empty!")
        
        if len(X_train) != len(y_train):
            raise ValueError("Mismatched X_train and y_train lengths!")
        
        # Store training data
        self.X_train = np.asarray(X_train)
        self.y_train = y_train

        # Automatically choose k if not provided
        if self.k is None:
            k = int(np.sqrt(len(X_train)))

            # Prefer an odd k to reduce ties
            if k % 2 == 0:
                k -= 1
            
            self.k = max(1, k) # Ensure k is at least 1
        
        if self.k > len(X_train):
            raise ValueError("K can't be larger than training data length!")
    
    def predict(self, X_query: Tuple[float, ...]) -> str:
        """
        Predict the class label for a single query point.

        Args:
            X_query: Feature vector of the query point.

        Returns:
            The predicted class label.

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If the query dimension does not match training data.
        """

        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Model must be fitted before prediction!")
        
        X_query = np.asarray(X_query)

        # Ensure query dimensionality matches training data
        if X_query.shape[0] != self.X_train.shape[1]:
            raise ValueError("Query point has different dimension than training data.")
        
        # Compute distances to all training samples
        distances = self._compute_distances(X_query)

        # Get indices of the k nearest neighbors
        k_nearest_indices = np.argpartition(distances, self.k)[:self.k]
        
        # Aggregate votes for each class
        class_votes: Dict[str, float] = {}
        for idx in k_nearest_indices:
            label = self.y_train[idx]
            class_votes[label] = class_votes.get(label, 0.0) + self.vote_fn(distances[idx])
        
        # Return the class with the highest total vote
        return max(class_votes, key=class_votes.get)
    
    def predict_batch(self, X_test: Sequence[Tuple[float, ...]]) -> List[str]:
        """
        Predict class labels for multiple query points.

        Args:
            X_test: A sequence of feature vectors.

        Returns:
            A list of predicted class labels.
        """
        return [self.predict(X_query) for X_query in X_test]
