# AI-Generated Text Detector

A machine learning project that classifies essays as either **AI-generated or human-written**, achieving over **90% accuracy**. The project explores a range of classical ML models and neural networks, comparing their performance on a real-world text classification problem.

---

## Overview

With the rise of AI writing tools, distinguishing between human and machine-generated content has become an increasingly relevant problem. This project tackles that challenge by training and evaluating multiple classifiers on a labeled dataset of essays, using text vectorization and dimensionality reduction to prepare features for classification.

---

## Features

- **Exploratory Data Analysis** — Text length distributions, word clouds, and label breakdowns across different essay prompts
- **Text Vectorization** — Bag-of-words using CountVectorizer with English stop word removal
- **Dimensionality Reduction** — TruncatedSVD to reduce feature space while retaining ~87% of variance
- **Multiple Classifiers** — Logistic Regression, Decision Tree, Random Forest, AdaBoost, and MLP (Neural Network)
- **Hyperparameter Tuning** — GridSearchCV used to optimize MLP activation functions, solvers, and iterations
- **Cross-Validation** — 5-fold cross-validation to evaluate model generalization

---

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core language |
| Pandas / NumPy | Data manipulation & analysis |
| Matplotlib | Data visualization |
| Scikit-learn | ML models, vectorization, evaluation |
| WordCloud | Text visualization |
| Google Colab | Development environment |

---

## Models Evaluated

| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear classifier |
| Decision Tree | Interpretable tree-based classifier (max depth 3) |
| Random Forest | Ensemble of 200 decision trees |
| AdaBoost | Boosting ensemble (100 estimators) |
| MLP — Sigmoid | Neural network with logistic activation |
| MLP — ReLU | Neural network with ReLU activation ✅ Best performer |

> Best accuracy achieved: **90%+** using MLP with ReLU activation after hyperparameter tuning.

---

## How It Works

1. Load and explore the labeled essay dataset
2. Visualize label distributions, text lengths, and word frequency
3. Vectorize text using CountVectorizer (Bag-of-Words)
4. Apply TruncatedSVD for dimensionality reduction
5. Train and evaluate multiple classifiers
6. Tune the best-performing model (MLP) using GridSearchCV
7. Validate results with 5-fold cross-validation

---

## Results

- **Best Model:** MLP Classifier with ReLU activation
- **Best Accuracy:** 90%+
- **Validation:** 5-fold cross-validation confirmed model generalization

---

## What I Learned

- Applying and comparing multiple classical ML and neural network models on a real dataset
- The impact of dimensionality reduction (SVD) on model performance and training speed
- Hyperparameter tuning with GridSearchCV for neural networks
- Evaluating models properly using cross-validation to avoid overfitting
