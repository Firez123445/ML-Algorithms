"""
K-Means Benchmark Script

This script benchmarks multiple implementations of the K-Means clustering
algorithm under a controlled and lightweight workload.

It compares:
- Pure Python implementation
- Optimized NumPy-based implementation
- Scikit-learn reference implementation

The benchmark focuses on clustering (fit) time, which dominates the
computational cost of K-Means.

To ensure fair and reproducible results:
- A synthetic clustering dataset is used
- Dataset size is capped to avoid excessive system load
- A warm-up run is performed before timing
- Multiple runs are aggregated using min / mean / std statistics
"""

import time
import statistics
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Type

from kmeans import PureKMeans, OptimizedKMeans, SklearnKMeans


def load_csv(filepath: Path, limit: Optional[int] = None) -> np.ndarray:
    """
    Load clustering dataset from CSV.

    Args:
        filepath: Path to dataset CSV.
        limit: Optional maximum number of samples to load.

    Returns:
        NumPy array of shape (n_samples, n_features).
    """
    
    X = np.loadtxt(filepath, delimiter=",", skiprows=1)
    
    if limit is not None:
        X = X[:limit]
    
    return X

def benchmark_model(model_cls: Type, X: np.ndarray,
                     n_clusters: int, runs: int) -> Dict[str, float]:
    """
    Benchmark a clustering model (fit-time focused).

    Timing primarily measures the cost of fitting (clustering),
    which dominates K-Means runtime.
    """
    
    times = []

    # Warm-up
    model = model_cls(n_clusters=n_clusters)
    model.fit(X)
    model.predict_batch(X[:10])

    for _ in range(runs):
        model = model_cls(n_clusters=n_clusters)

        start = time.perf_counter()
        model.fit(X)
        end = time.perf_counter()

        times.append(end - start)

    return {
        "min": min(times),
        "mean": statistics.mean(times),
        "std": statistics.stdev(times) if runs > 1 else 0.0
    }

def main() -> None:
    """
    Run the K-Means benchmark on a controlled clustering dataset.
    """

    dataset_path = Path("./unsupervised/datasets/clustering")
    
    data_file = "clustering_blobs_10D_10000.csv"
    
    X = load_csv(dataset_path / data_file, limit=1000)

    N_CLUSTERS = 3
    RUNS = 3

    benchmarks = {
        "PureKMeans": benchmark_model(PureKMeans, X, N_CLUSTERS, RUNS),
        "OptimizedKMeans": benchmark_model(OptimizedKMeans, X, N_CLUSTERS, RUNS),
        "SKlearnKMeans": benchmark_model(SklearnKMeans, X, N_CLUSTERS, RUNS)

    }

    print("\nKMeans Benchmark Results (safe workload)")
    print("-" * 45)

    for name, stats in benchmarks.items():
        print(
            f"{name:<15} | "
            f"min: {stats['min']:.4f}s | "
            f"mean: {stats['mean']:.4f}s | "
            f"std: {stats['std']:.4f}s"
        )


if __name__ == "__main__":
    main()
