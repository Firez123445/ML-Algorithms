# K-Nearest Neighbors (KNN)

This directory contains multiple implementations of the **K-Nearest Neighbors (KNN)** algorithm, designed for learning, experimentation, and performance comparison.

KNN is a **lazy, instance-based supervised learning algorithm** that makes predictions based on the similarity between a query point and training samples.

---

## About KNN

KNN works by:

1. Computing the distance between a query point and all training samples
2. Selecting the `k` nearest neighbors
3. Aggregating their labels (majority vote or weighted vote)

Key characteristics:

* No explicit training phase (training = storing data)
* Prediction cost grows with dataset size
* Performance highly depends on distance computation

This makes KNN a great candidate for studying **algorithmic optimization and performance trade-offs**.

---

## Implementations

This module includes three conceptually equivalent implementations:

* **Pure Python**
  A from-scratch implementation focused on clarity and correctness. Useful for understanding the algorithm and its bottlenecks.

* **Optimized (NumPy / SciPy)**
  Uses vectorized operations and efficient distance computation to significantly improve performance.

* **Scikit-learn Reference**
  A wrapper around `sklearn.neighbors.KNeighborsClassifier`, used as a well-optimized baseline.

All implementations share a similar API (`fit`, `predict`, `predict_batch`) to allow fair comparison.

---

## Benchmarking

A lightweight benchmark script is included to compare prediction performance across implementations.

### Benchmark Setup

* **Task**: Classification
* **Training samples**: 10,000
* **Test samples**: 200
* **Dimensions**: 10D
* **k**: Automatically selected
* **Metric**: Euclidean distance
* **Workload**: Intentionally limited to avoid excessive system load

Each implementation was run multiple times, and summary statistics were recorded.

---

## Benchmark Results

```
KNN Benchmark Results (safe workload)
----------------------------------------
PureKNN      | min: 7.2406s | mean: 7.2747s | std: 0.0307s
OptimizedKNN | min: 0.0724s | mean: 0.0899s | std: 0.0205s
SklearnKNN   | min: 0.0480s | mean: 0.0519s | std: 0.0049s
```

---

## Discussion

These results highlight several important points:

* **Pure Python** implementations are easy to read but scale poorly due to Python-level loops and repeated distance computations.
* **Vectorization and optimized numerical libraries** provide orders-of-magnitude speedups without changing the algorithm itself.
* **Scikit-learn** remains the fastest and most stable, benefiting from highly optimized C/Fortran-backed routines and years of engineering effort.

This comparison demonstrates that **algorithm choice and implementation strategy are equally important** when working with real-world data sizes.

---

## Notes

* Accuracy comparison is intentionally omitted; all implementations are algorithmically equivalent.
* The benchmark focuses on **relative performance**, not absolute timing.
* Results may vary depending on hardware and environment.

---

This module will continue to evolve with additional experiments, variants (e.g. regression), and optimization techniques.
