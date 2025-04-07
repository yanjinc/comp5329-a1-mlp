# loss.py
"""
Implements the combination of Softmax activation and Cross-Entropy loss
for multi-class classification.

- Accepts logits (raw scores) as input
- Supports integer labels or one-hot encoded labels
"""

import numpy as np

class SoftmaxCrossEntropyLoss:
    def __init__(self):
        self.probs = None    # Softmax probabilities
        self.labels = None   # Ground truth class indices

    def forward(self, logits, labels):
        """
        Compute the average cross-entropy loss over a batch.

        Parameters:
            logits: np.ndarray, shape (N, C)
                Model output scores (before softmax)
            labels: np.ndarray, shape (N,) or (N, C)
                Ground truth class labels

        Returns:
            loss: float
                Mean cross-entropy loss over the batch
        """
        # Shift logits for numerical stability
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Convert one-hot labels to class indices if needed
        if labels.ndim == 2:
            self.labels = np.argmax(labels, axis=1)
        else:
            self.labels = labels

        N = logits.shape[0]
        assert N > 0, "Empty input batch"

        # Cross-entropy loss: -log(p_true)
        log_probs = -np.log(self.probs[np.arange(N), self.labels] + 1e-8)
        loss = np.mean(log_probs)
        return loss

    def backward(self):
        """
        Compute gradient of loss w.r.t. logits.

        Returns:
            grad: np.ndarray, shape (N, C)
                Gradient to propagate to previous layer
        """
        N = self.probs.shape[0]
        grad = self.probs.copy()
        grad[np.arange(N), self.labels] -= 1
        return grad / N
