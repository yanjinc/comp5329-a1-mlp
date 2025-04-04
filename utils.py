# utils.py
"""
Evaluation utilities for classification models:
- Data shuffling
- Accuracy
- Confusion matrix
- Precision / Recall / F1 Score (macro average)
"""

import numpy as np


def shuffle_data(X, y):
    """
    Shuffle data and labels in unison.

    Args:
        X: np.ndarray, shape (N, D)
        y: np.ndarray, shape (N,) or (N, C)

    Returns:
        Tuple of shuffled (X, y)
    """
    assert len(X) == len(y), "X and y must have the same number of samples"
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    return X[indices], y[indices]


def accuracy(preds, labels):
    """
    Compute classification accuracy.

    Args:
        preds: np.ndarray, shape (N, C) - predicted scores (e.g., logits or softmax)
        labels: np.ndarray, shape (N,) or (N, C) - true labels

    Returns:
        Accuracy (float)
    """
    pred_class = np.argmax(preds, axis=1)
    true_class = np.argmax(labels, axis=1) if labels.ndim == 2 else labels
    return np.mean(pred_class == true_class)


def confusion_matrix(preds, labels, num_classes):
    """
    Compute the confusion matrix.

    Args:
        preds: np.ndarray, shape (N, C) - predicted scores
        labels: np.ndarray, shape (N,) or (N, C)
        num_classes: int - total number of classes

    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    """
    pred_class = np.argmax(preds, axis=1)
    true_class = np.argmax(labels, axis=1) if labels.ndim == 2 else labels
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true_class, pred_class):
        cm[t, p] += 1
    return cm


def precision_recall_f1(cm):
    """
    Compute macro-averaged precision, recall, and F1 score from confusion matrix.

    Args:
        cm: np.ndarray, shape (C, C), confusion matrix

    Returns:
        Tuple of (precision, recall, f1) macro-averaged scores
    """
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP

    precision = TP / (TP + FP + 1e-8)  # Avoid divide-by-zero
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return np.mean(precision), np.mean(recall), np.mean(f1)
