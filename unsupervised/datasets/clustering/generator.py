"""
Clustering Dataset Generator

This script generates synthetic clustering datasets for unsupervised learning
experiments. Generated datasets contain feature vectors only (no labels) and
are saved as CSV files.
"""

import numpy as np
from pathlib import Path
from typing import Union, List
from sklearn.datasets import make_blobs


def save_csv(X: np.ndarray, path: Path) -> None:
    """
    Save clustering dataset to CSV format.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        path: Output file path.
    """

    np.savetxt(path, X, delimiter=",", fmt="%.6f")

def generate_blobs(
    n_samples: int,
    n_features: int,
    n_clusters: int,
    out_dir: Path,
    cluster_std: Union[float, List[float]] = 1.0,
    seed: int = 42,
    name_suffix: str = ""
) -> None:
    """
    Generate a blob-based clustering dataset.

    Args:
        n_samples: Number of samples.
        n_features: Number of feature dimensions.
        n_clusters: Number of clusters.
        out_dir: Directory to save the dataset.
        cluster_std: Standard deviation of clusters (can be per-cluster).
        seed: Random seed.
        name_suffix: Optional suffix for filename.
    """

    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=cluster_std,
        random_state=seed
    )

    name = f"clustering_blobs_{n_features}D_{n_samples}{name_suffix}.csv"
    save_csv(X, out_dir / name)

def main() -> None:
    """
    Generate a predefined set of clustering datasets.
    """

    base_dir = Path("./unsupervised/datasets/clustering")

    configs = [
        (1_000, 2, 3, 1.0),
        (1_000, 2, 3, 2.5),
        (5_000, 5, 4, 1.0),
        (10_000, 10, 5, 1.2),
        (3_000, 2, 3, [0.5, 2.5, 1]),
    ]

    for samples, dims, clusters, std in configs:
        suffix = "_varied" if isinstance(std, list) else ""

        generate_blobs(
            n_samples=samples,
            n_features=dims,
            n_clusters=clusters,
            cluster_std=std,
            out_dir=base_dir,
            name_suffix=suffix
        )

    print(f"Clustering datasets generated in {base_dir}")


if __name__ == "__main__":
    main()
