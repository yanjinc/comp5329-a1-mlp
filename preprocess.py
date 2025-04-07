# preprocess.py
"""
Data preprocessing utilities:
- normalize(): min-max scaling to [0, 1]
- standardize(): zero mean, unit variance
- one_hot_encode(): one-hot encoding for classification labels
"""

import numpy as np

def normalize(X, axis=0):
    """
    Normalize input features to [0, 1] range using min-max normalization.

    Args:
        X: np.ndarray of shape (N, D)
        axis: axis to normalize over (1 = per sample, 0 = per feature)

    Returns:
        np.ndarray: normalized input of the same shape
    """
    X = np.array(X)
    X_min = np.min(X, axis=axis, keepdims=True)
    X_max = np.max(X, axis=axis, keepdims=True)
    return (X - X_min) / (X_max - X_min + 1e-8)


def standardize(X, axis=0):
    """
    Standardize input to zero mean and unit variance (Z-score normalization).

    Args:
        X: np.ndarray of shape (N, D)
        axis: axis to compute mean/std over (0 = per feature)

    Returns:
        np.ndarray: standardized input of the same shape
    """
    X = np.array(X)
    mean = np.mean(X, axis=axis, keepdims=True)
    std = np.std(X, axis=axis, keepdims=True)
    return (X - mean) / (std + 1e-8)


def one_hot_encode(y, num_classes):
    """
    Convert integer class labels into one-hot encoded format.

    Args:
        y: array-like of shape (N,) - class indices (e.g. [0, 2, 1])
        num_classes: int, total number of classes

    Returns:
        np.ndarray of shape (N, num_classes)
    """
    y = np.array(y).astype(int).reshape(-1)  # 确保是一维向量
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot
