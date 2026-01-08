# Unsupervised Learning

This section of the repository focuses on **unsupervised learning algorithms** — methods that operate on data **without labeled targets** and aim to uncover hidden structures or patterns within the data.

Unsupervised learning is fundamentally exploratory: instead of predicting known outputs, algorithms attempt to **organize, model, or represent data** based solely on its intrinsic properties.

---

## Goals & Philosophy

The unsupervised part of this repository is designed with the following goals:

- Implement unsupervised algorithms from scratch where meaningful
- Provide **synthetic dataset generators** tailored to each algorithm’s assumptions
- Analyze algorithm behavior rather than only final results
- Enable fair and reproducible benchmarking
- Highlight both **success cases and failure modes**

This section prioritizes **understanding algorithms**, not just using them.

---

## Algorithm Categories

Unsupervised algorithms are grouped conceptually, allowing the repository to evolve over time without rigid constraints.

### 1. Clustering

Clustering algorithms group samples based on similarity, distance, or density.

Typical characteristics:
- No ground-truth labels
- Sensitive to data geometry and scale
- Often rely on implicit assumptions (e.g. spherical clusters)

Examples:
- KMeans
- (Planned) Hierarchical Clustering
- (Planned) Density-based clustering (e.g. DBSCAN)

Focus areas:
- Cluster shape assumptions
- Initialization sensitivity
- Scalability with data size and dimensionality

---

### 2. Density Estimation (Planned)

Density estimation algorithms attempt to model the underlying probability distribution of the data.

Examples:
- Gaussian Mixture Models (GMM)
- Kernel Density Estimation (KDE)

Focus areas:
- Multi-modal distributions
- Probabilistic interpretation
- Overfitting vs generalization

---

### 3. Dimensionality Reduction (Planned)

Dimensionality reduction techniques aim to learn compact representations of data while preserving meaningful structure.

Examples:
- PCA
- (Planned) t-SNE, UMAP

Focus areas:
- Variance preservation
- Information loss
- Visualization vs compression trade-offs

---

## Datasets

Because unsupervised learning does not rely on labels, datasets are designed to emphasize:

- Geometric structure (clusters, manifolds)
- Distribution shape
- Noise and outliers
- Dimensionality effects

Dataset generators are preferred over static datasets to ensure:
- Reproducibility
- Controlled experiments
- Easy scaling of samples and dimensions

Both **low-dimensional datasets** (for visualization and intuition) and **high-dimensional datasets** (for performance analysis) are considered.

---

## Design Principles

- No strict dependency on a fixed directory structure
- Algorithms and datasets evolve independently
- Dataset generators are algorithm-aware
- Clarity and correctness over feature count

This section is intended to grow organically as new unsupervised methods and experiments are added.
