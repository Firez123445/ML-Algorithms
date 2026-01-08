"""
Pure Python implementation of the K-Nearest Neighbors (KNN) classifier.

This version is written using only the Python standard library and focuses on
algorithmic clarity rather than performance. It serves as a baseline for
understanding KNN internals and benchmarking against optimized implementations.
"""

from math import sqrt
from heapq import nlargest
from typing import Optional, Sequence, Tuple, List, Dict


class KNN:
    """
    A simple implementation of the k-Nearest Neighbors (k-NN) classifier.

    This classifier supports Euclidean and Manhattan distance metrics and
    optionally uses distance-weighted voting.
    """

    def __init__(self, k: Optional[int] = None, metric: str = 'euclidean', weighted: bool = True) -> None:
        """
        Initialize the KNN classifier.

        Args:
            k: Number of nearest neighbors to consider. If None, k is set
               automatically based on the training data size.
            metric: Distance metric to use ('euclidean' or 'manhattan').
            weighted: Whether to weight votes by inverse distance.
        """

        self.k = k
        self.metric = metric.lower()
        self.X_train: Optional[Sequence[Tuple[float, ...]]] = None
        self.y_train: Optional[Sequence[str]] = None

        # Voting function: inverse-distance weighting or uniform voting
        if weighted:
            self.vote_fn = lambda distance: 1 / (distance + 1e-9)
        else:
            self.vote_fn = lambda distance: 1.0

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

        return sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    
    @staticmethod
    def _manhattan_distance(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
        """
        Compute the Manhattan (L1) distance between two points.

        Args:
            a: First point.
            b: Second point.

        Returns:
            The Manhattan distance between a and b.
        """

        return sum(abs(x - y) for x, y in zip(a, b))

    def _compute_distance(self, a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
        """
        Compute the distance between two points using the selected metric.

        Args:
            a: First point.
            b: Second point.

        Returns:
            The computed distance.

        Raises:
            ValueError: If an unsupported distance metric is specified.
        """

        match self.metric:
            case 'euclidean':
                return self._euclidean_distance(a, b)
            
            case 'manhattan':
                return self._manhattan_distance(a, b)
            
            case _:
                raise ValueError(f"Unsupported metric: {self.metric}.")

    def fit(self, X_train: Sequence[Tuple[float, ...]], y_train: Sequence[str]) -> None:
        """
        Fit the KNN classifier using training data.

        Args:
            X_train: Training feature vectors.
            y_train: Corresponding class labels.

        Raises:
            ValueError: If training data is empty, mismatched, or k is invalid.
        """

        if len(X_train) == 0:
            raise ValueError("Training data is empty.")
        
        if len(X_train) != len(y_train):
            raise ValueError("Mismatched X_train and y_train lengths.")
        
        # Store training data
        self.X_train = X_train
        self.y_train = y_train

        # Automatically choose k if not provided 
        if self.k is None:
            k = int(sqrt(len(X_train)))

            # Prefer an odd k to reduce ties
            if k % 2 == 0:
                k -= 1

            self.k = max(k, 1) # Ensure k is at least 1

        if self.k > len(X_train):
            raise ValueError("K can't be larger than training set length.")
    
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
            raise RuntimeError("Model must be fitted before prediction.")
        
        # Ensure query dimensionality matches training data
        if len(X_query) != len(self.X_train[0]):
            raise ValueError("Query point has different dimension than training data.")
        
        # Compute distances to all training samples
        distances = [
            (self._compute_distance(x, X_query), y)
            for x, y in zip(self.X_train, self.y_train)
        ]
        
        # Select the k nearest neighbors (smallest distances)
        k_nearest = nlargest(
            self.k,
            distances,
            key = lambda x: -x[0]
        )

        # Aggregate votes for each class
        class_votes: Dict[str, float] = {}
        for dist, label in k_nearest: 
            class_votes[label] = class_votes.get(label, 0.0) + self.vote_fn(dist)

        # Return the class with the highest total vote
        return max(class_votes, key = class_votes.get)
    
    def predict_batch(self, X_test: Sequence[Tuple[float, ...]]) -> List[str]:
        """
        Predict class labels for multiple query points.

        Args:
            X_test: A sequence of feature vectors.

        Returns:
            A list of predicted class labels.
        """
        
        return [self.predict(X_query) for X_query in X_test]
