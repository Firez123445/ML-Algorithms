"""
KNN Benchmark Script

This script benchmarks different implementations of the K-Nearest Neighbors (KNN)
algorithm under a controlled and lightweight workload.

It compares:
- Pure Python implementation
- Optimized NumPy-based implementation
- Scikit-learn reference implementation

The benchmark measures end-to-end time including both training (fit)
and inference (batch prediction).
"""

import csv
import time
import statistics 
from pathlib import Path
from typing import Tuple, Sequence, Dict, Type

from knn import PureKNN, OptimizedKNN, SklearnKNN


def load_train_csv(filepath: Path) -> Tuple[Sequence[Tuple[float, ...]], Sequence[str]]:
    """
    Load a training CSV file containing features and labels.

    The CSV file is expected to have:
    - A header row
    - Feature columns followed by a label column

    Args:
        filepath: Path to the training CSV file.

    Returns:
        A tuple (X, y) where:
        - X is a sequence of feature tuples
        - y is a sequence of labels
    """

    X, y = [], []

    with open(filepath, newline="") as f:
        reader = csv.reader(f)
        next(reader) # Skip header

        for row in reader:
            *features, label = row
            X.append(tuple(map(float, features)))
            y.append(label)
    
    return X, y

def load_test_csv(filepath: Path, limit: int) -> Sequence[Tuple[float, ...]]:
    """
    Load a test CSV file containing only feature vectors.

    The CSV file is expected to have:
    - A header row
    - Feature columns only (no labels)

    Only the first `limit` samples are loaded to keep the benchmark workload safe.

    Args:
        filepath: Path to the test CSV file.
        limit: Maximum number of samples to load.

    Returns:
        A sequence of feature tuples.
    """

    X_test = []

    with open(filepath, newline="") as f:
        reader = csv.reader(f)
        next(reader) # Skip header

        for row in reader:
            X_test.append(tuple(map(float, row)))

            if len(X_test) >= limit:
                break

    return X_test

def benckmark_model(model_cls: Type, X_train, y_train, X_test, runs: int) -> Dict[str, float]:
    """
    Benchmark a KNN model implementation.

    The benchmark measures total execution time for:
    - Model fitting
    - Batch prediction

    A warm-up run is performed before timing to reduce
    cache and initialization effects.

    Args:
        model_cls: KNN class to benchmark.
        X_train: Training feature vectors.
        y_train: Training labels.
        X_test: Test feature vectors.
        runs: Number of benchmark repetitions.

    Returns:
        A dictionary containing:
        - min: Minimum execution time
        - mean: Mean execution time
        - std: Standard deviation of execution time
    """
    
    times = []

    # Warm up
    model = model_cls()
    model.fit(X_train, y_train)
    model.predict_batch(X_test[:10])

    for _ in range(runs):
        model = model_cls()

        start = time.perf_counter()
        model.fit(X_train, y_train)
        model.predict_batch(X_test)
        end = time.perf_counter()

        times.append(end - start)

    return {
        "min": min(times),
        "mean": statistics.mean(times),
        "std": statistics.stdev(times) if runs > 1 else 0.0
    }

def main() -> None:
    """
    Run the KNN benchmark on a controlled dataset and report timing results.
    """
    
    dataset_path = Path("./supervised/datasets/classification")

    train_path = dataset_path / "classification_classification_10D_10000_train.csv"
    test_path = dataset_path / "classification_classification_10D_10000_test.csv"

    X_train, y_train = load_train_csv(train_path)
    X_test = load_test_csv(test_path, limit=300)

    RUNS = 3

    benchmarks = {
        "PureKNN": benckmark_model(PureKNN, X_train, y_train, X_test, RUNS),
        "OptimizedKNN":benckmark_model(OptimizedKNN, X_train, y_train, X_test, RUNS),
        "SklearnKNN": benckmark_model(SklearnKNN, X_train, y_train, X_test, RUNS),
    }

    print("\nKNN Benchmark Results (safe workload)")
    print("-" * 40)

    for name, stats in benchmarks.items():
        print(f"{name:<12} | "
            f"min: {stats['min']:.4f}s | "
            f"mean: {stats['mean']:.4f}s | "
            f"std: {stats['std']:.4f}s")


if __name__ == "__main__":
    main()
  