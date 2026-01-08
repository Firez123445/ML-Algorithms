# K-Means Clustering

This directory contains multiple implementations of the **K-Means clustering**
algorithm, designed for learning, experimentation, and performance comparison.

K-Means is an **unsupervised learning algorithm** that partitions data into `k`
clusters by minimizing intra-cluster variance.

---

## About K-Means

K-Means works by iteratively repeating the following steps:

1. Initialize `k` cluster centroids (randomly or using K-Means++)
2. Assign each data point to the nearest centroid
3. Update centroids as the mean of assigned points
4. Repeat until convergence or reaching a maximum number of iterations

Key characteristics:

* Unsupervised (no labels required)
* Sensitive to initialization
* Assumes roughly spherical, equally sized clusters
* Performance depends heavily on distance computation and vectorization

Because of its simplicity and widespread use, K-Means is an excellent algorithm
for studying **optimization techniques and numerical performance**.

---

## Implementations

This module includes three conceptually equivalent implementations:

* **Pure Python**
  A from-scratch implementation using only core Python constructs.  
  Focused on readability and algorithmic clarity, not performance.

* **Optimized (NumPy-based)**
  Uses NumPy vectorization and efficient numerical operations to significantly
  reduce Python-level overhead.

* **Scikit-learn Reference**
  A wrapper around `sklearn.cluster.KMeans`, serving as a highly optimized
  and battle-tested baseline.

All implementations expose a similar interface:

* `fit(X)`
* `predict(X)`
* `predict_batch(X)` (API consistency for benchmarks)
* `centroids` (property)

This allows fair and direct performance comparison.

---

## Dataset

Benchmarks are performed on **synthetic clustering datasets** generated using
Gaussian blobs.

Dataset characteristics:

* **Type**: Clustering (blobs)
* **Dimensions**: 10D
* **Samples**: 10,000 (limited to 1,000 for safe workload)
* **Clusters**: 3
* **Labels**: Not used (unsupervised)

Unlike supervised learning, K-Means does **not require a test set**.  
Evaluation focuses on **convergence behavior and runtime performance**, not
generalization accuracy.

The workload is intentionally capped to ensure reproducible results on typical
hardware.

---

## Benchmarking

A lightweight benchmark script compares the runtime performance of all
implementations.

### Benchmark Setup

* **Task**: Clustering
* **Input size**: 1,000 samples × 10 dimensions
* **Clusters (`k`)**: 3
* **Runs**: 3
* **Metric**: Euclidean distance
* **Timing focus**: `fit` phase only
* **Workload**: Safe / limited

The benchmark primarily measures **clustering time**, as the `fit` phase
dominates K-Means runtime. A warm-up run is performed before timing to reduce
initialization and caching effects.

---

## Benchmark Results

```

## KMeans Benchmark Results (safe workload)

PureKMeans      | min: 0.0891s | mean: 0.1034s | std: 0.0128s
OptimizedKMeans | min: 0.0016s | mean: 0.0023s | std: 0.0012s
SklearnKMeans   | min: 0.0024s | mean: 0.0032s | std: 0.0013s

```

---

## Discussion

The benchmark results highlight several important points:

* **Pure Python** implementations are significantly slower due to nested loops
  and Python-level distance computations.
* **Vectorized NumPy implementations** achieve orders-of-magnitude speedups
  while preserving algorithmic correctness.
* **Scikit-learn** provides excellent performance and numerical robustness,
  benefiting from optimized C/Cython backends.

Interestingly, for this workload the optimized NumPy implementation slightly
outperforms the scikit-learn wrapper. This is primarily due to reduced abstraction
overhead and a simplified benchmarking setup, demonstrating how performance can
vary depending on problem size and implementation details.

---

## Notes

* Accuracy comparison is intentionally omitted; all implementations converge
  to comparable cluster assignments.
* Benchmarks focus on **relative performance**, not absolute timing.
* Results may vary depending on hardware, BLAS backend, and system load.

---

This module may be extended in the future with:

* K-Means++ initialization analysis
* Convergence diagnostics
* Distance metric variants
* Larger-scale benchmarks