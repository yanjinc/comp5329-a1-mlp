# layers.py
"""
Defines core neural network layers: Linear (Fully Connected), Dropout, and Batch Normalization.
Each layer includes both forward and backward propagation logic.
"""

import numpy as np


class Linear:
    """
    Fully-connected (dense) linear layer: y = x @ W + b
    Stores gradients dW, db for backpropagation and optimizer update.
    """
    def __init__(self, in_features, out_features, use_batchnorm=False, dropout_prob=0.0):
        # Xavier initialization
        self.W = np.random.uniform(
            low = -np.sqrt(6. / (in_features + out_features)),
            high = np.sqrt(6. / (in_features + out_features)),
            size = (in_features, out_features)
        )
        self.b = np.zeros((1, out_features))

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.use_batchnorm = use_batchnorm
        self.dropout_prob = dropout_prob

    def forward(self, x, training=True):
        self.input = x
        out = x @ self.W + self.b
        self.cache = out   # Cache for activation function
        return out

    def backward(self, grad_output, training=True):
        self.dW = self.input.T @ grad_output
        self.db = np.sum(grad_output, axis=0, keepdims=True)
        return grad_output @ self.W.T
    
    def get_parameters(self):
        return {
            "W": self.W,
            "b": self.b,
            "dW": self.dW,
            "db": self.db
        }


class Dropout:
    """
    Dropout regularization layer.

    During training:
        - Randomly zeroes elements in the input tensor with probability `drop_prob`.
        - Scales remaining elements by 1 / (1 - drop_prob) to preserve expected value.

    During inference:
        - No dropout is applied (identity function).
    """
    def __init__(self, drop_prob):
        assert 0.0 <= drop_prob < 1.0, "drop_prob must be in [0, 1)"
        self.drop_prob = drop_prob
        self.mask = None

    def forward(self, x, training=True):
        """
        Args:
            x: np.ndarray, input tensor of shape (N, D)
            training: bool, whether in training mode

        Returns:
            output tensor after dropout
        """
        if training:
            self.mask = (np.random.rand(*x.shape) > self.drop_prob).astype(float)
            scale = 1.0 / (1.0 - self.drop_prob + 1e-8)  # Avoid divide-by-zero
            return x * self.mask * scale
        else:
            return x  # inference: no dropout

    def backward(self, grad_output):
        """
        Backpropagate through dropout.

        Args:
            grad_output: np.ndarray, upstream gradient of shape (N, D)

        Returns:
            gradient passed to previous layer
        """
        scale = 1.0 / (1.0 - self.drop_prob + 1e-8)
        return grad_output * self.mask * scale



class BatchNorm:
    def __init__(self, dim, eps=1e-5, momentum=0.9):
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.running_mean = np.zeros(dim)
        self.running_var = np.ones(dim)
        self.x_centered = None
        self.std_inv = None
        self.x_norm = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, training=True):
        if training:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            self.x_centered = x - mean
            self.std_inv = 1. / np.sqrt(var + self.eps)
            self.x_norm = self.x_centered * self.std_inv
            out = self.gamma * self.x_norm + self.beta
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_norm + self.beta
        return out

    def backward(self, grad_output):
        N, D = grad_output.shape
        self.dbeta = grad_output.sum(axis=0)
        self.dgamma = np.sum(grad_output * self.x_norm, axis=0)
        dx_norm = grad_output * self.gamma
        dvar = np.sum(dx_norm * self.x_centered, axis=0) * -0.5 * self.std_inv**3
        dmean = np.sum(dx_norm * -self.std_inv, axis=0) + dvar * np.mean(-2 * self.x_centered, axis=0)
        dx = dx_norm * self.std_inv + dvar * 2 * self.x_centered / N + dmean / N
        return dx
