# Supervised Learning

## Overview

This section of the repository focuses on **Supervised Machine Learning algorithms**.

Supervised learning refers to a class of algorithms that learn a mapping from input features to target labels using **labeled data**. These algorithms are typically used for:

* **Classification** (predicting discrete labels)
* **Regression** (predicting continuous values)

The implementations in this directory are part of an **evolving codebase** and are added incrementally over time.

---

## Goals of This Section

* Implement supervised algorithms **from scratch** to understand their core mechanics
* Provide **optimized versions** using numerical libraries when appropriate
* Compare handcrafted implementations with **standard library-based approaches**
* Use shared datasets and benchmarks to evaluate behavior and performance

This is not intended to be a fixed framework, but a growing collection of experiments and references.

---

## Algorithm Categories

### Classification

Algorithms that predict discrete class labels. Examples that may appear in this section include:

* k-Nearest Neighbors (KNN)
* Logistic Regression
* Naive Bayes
* Decision Trees
* Support Vector Machines (SVM)

### Regression

Algorithms that predict continuous values. Examples include:

* Linear Regression
* Polynomial Regression
* Ridge / Lasso Regression
* kNN Regression

Not all algorithms are guaranteed to exist at all times — this section grows gradually.

---

## Datasets

The `datasets/` directory contains **synthetic and generated datasets** designed to be reusable across multiple supervised algorithms.

General principles:

* Separate datasets for **classification** and **regression**
* Support for multiple dimensionalities (e.g. 2D, 5D, 10D)
* Simple CSV-based format for transparency and portability
* Dataset generators are included to allow reproducible data creation

Datasets are shared across algorithms to allow fair comparison.

---

## Design Philosophy

* Simple and readable code
* Minimal abstraction layers
* Explicit control over algorithm behavior
* Educational value prioritized over production-level optimization

Implementations may trade strict efficiency for clarity, especially in "from scratch" versions.

---

## Notes

* Directory structure may change as new algorithms are added
* APIs are not guaranteed to be stable
* Some subdirectories may be incomplete or empty during development

---

## Status

🚧 Actively evolving — new algorithms, datasets, and benchmarks are added over time.
