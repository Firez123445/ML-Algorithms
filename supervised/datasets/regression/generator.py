"""
Regression Dataset Generator

This script generates synthetic regression datasets for supervised learning
experiments. Data is split into training and test sets and saved as CSV files.
"""

import numpy as np
from pathlib import Path
from typing import Tuple
from sklearn.datasets import make_regression


def save_csv(X: np.ndarray, y: np.ndarray, path: Path) -> None:
    """
    Save regression dataset to CSV format.

    Features and target values are stored together, with the target
    as the last column.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Target vector of shape (n_samples,).
        path: Output file path.
    """

    data = np.hstack([X, y.reshape(-1, 1)])
    np.savetxt(path, data, delimiter=",", fmt="%.6f")

def split_data(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float,
    seed: int
) -> Tuple:
    """
    Shuffle and split dataset into training and test subsets.

    Args:
        X: Feature matrix.
        y: Target vector.
        train_ratio: Fraction of samples to use for training.
        seed: Random seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test
    """

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    split = int(len(X) * train_ratio)
    return X[:split], X[split:], y[:split], y[split:]

def generate_regression(
    n_samples: int,
    n_features: int,
    out_dir: Path,
    train_ratio: float = 0.8,
    noise: float = 0.0,
    seed: int = 42
) -> None:
    """
    Generate a synthetic regression dataset.

    Args:
        n_samples: Total number of samples.
        n_features: Number of input features.
        out_dir: Directory to save generated datasets.
        train_ratio: Fraction of data used for training.
        noise: Standard deviation of Gaussian noise added to targets.
        seed: Random seed.
    """

    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=seed
    )
    
    X = X.astype(np.float64)
    y = y.astype(np.float64)

    X_train, X_test, y_train, y_test = split_data(X, y, train_ratio, seed)

    name = f"regression_{n_features}D_{n_samples}"  

    save_csv(X_train, y_train, out_dir / f"{name}_train.csv")
    save_csv(X_test, y_test, out_dir / f"{name}_test.csv")

def main() -> None:
    """
    Generate a predefined set of regression datasets.
    """

    base_dir = Path("./supervised/datasets/regression")

    configs = [
        (1_000, 2, 0.1),
        (5_000, 5, 1.0),
    ]

    for samples, dims, noise in configs:
        generate_regression(
            n_samples=samples,
            n_features=dims,
            noise=noise,
            out_dir=base_dir
        )

    print(f"Regression datasets generated in {base_dir}")


if __name__ == "__main__":
    main()
