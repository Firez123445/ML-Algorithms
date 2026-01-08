"""
Classification Dataset Generator

This script generates synthetic classification datasets for supervised learning
experiments. It supports multiple data generation modes and automatically splits
data into training and test sets.

Generated datasets are saved as CSV files:
- Training files include feature vectors and class labels.
- Test files include feature vectors only (labels omitted).
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from sklearn.datasets import make_blobs, make_classification


def save_csv(X: np.ndarray, y: Optional[np.ndarray], path: Path) -> None:
    """
    Save dataset to CSV format.

    If labels are provided, they are appended as the last column.
    Otherwise, only feature vectors are saved.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Optional label array of shape (n_samples,).
        path: Output file path.
    """

    if y is not None:
        data = np.hstack([X, y.reshape(-1, 1)])
    else:
        data = X
    
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
        y: Label vector.
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

def generate_classification(
    n_samples: int,
    n_features: int,
    n_classes: int,
    out_dir: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
    mode: str = "blobs"
) -> None:
    """
    Generate a synthetic classification dataset.

    Supported modes:
    - 'blobs': Well-separated Gaussian clusters.
    - 'classification': More complex feature interactions using
      sklearn's make_classification.

    Args:
        n_samples: Total number of samples.
        n_features: Number of input features.
        n_classes: Number of target classes.
        out_dir: Directory to save generated datasets.
        train_ratio: Fraction of data used for training.
        seed: Random seed.
        mode: Dataset generation mode.
    """

    match mode:
        case "blobs":
            X, y = make_blobs(
                n_samples=n_samples,
                n_features=n_features,
                centers=n_classes,
                random_state=seed
            )

        case "classification":
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=max(1, n_features - 2),
                n_redundant=0,
                n_repeated=0,
                n_classes=n_classes,
                random_state=seed
            )

        case _:
            raise ValueError(f"Unknown classification mode: {mode}")

    X = X.astype(np.float64)
    y = y.astype(np.int64)
    
    X_train, X_test, y_train, y_test = split_data(X, y, train_ratio, seed)

    name = f"classification_{mode}_{n_features}D_{n_samples}"

    save_csv(X_train, y_train, out_dir / f"{name}_train.csv")
    save_csv(X_test, None, out_dir / f"{name}_test.csv")

def main() -> None:
    """
    Generate a predefined set of classification datasets.
    """

    base_dir = Path("./supervised/datasets/classification")

    configs = [
        (1_000, 2, 3, "blobs"),
        (1_000, 5, 3, "blobs"),
        (10_000, 10, 5, "classification"),
    ]

    for samples, dims, classes, mode in configs:
        generate_classification(
            n_samples=samples,
            n_features=dims,
            n_classes=classes,
            mode=mode,
            out_dir=base_dir
        )

    print(f"Classification datasets generated in {base_dir}")
    

if __name__ == "__main__":
    main()
