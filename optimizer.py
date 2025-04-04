# optimizer.py
"""
实现带 momentum & weight decay 的 SGD
"""
import numpy as np


class SGD:
    """
    Stochastic Gradient Descent optimizer with Momentum and Weight Decay (L2 regularization).
    Each parameter is a dictionary containing:
        - 'W': weight
        - 'b': bias
        - 'dW': gradient of weight
        - 'db': gradient of bias
    """

    def __init__(self, parameters, lr=0.01, momentum=0.0, weight_decay=0.0):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        # Initialize momentum buffers
        self.velocities = []
        for param in self.parameters:
            self.velocities.append({
                'vW': np.zeros_like(param['W']),
                'vb': np.zeros_like(param['b'])
            })

    def step(self):
        """
        Perform a single parameter update.
        """
        for i, param in enumerate(self.parameters):
            W, b = param['W'], param['b']
            dW, db = param['dW'], param['db']

            # Sanity check (防止 dW = None 导致类型错误)
            if dW is None or db is None:
                continue

            # Apply L2 weight decay (do not regularize biases)
            if self.weight_decay > 0:
                dW += self.weight_decay * W

            # Momentum update
            self.velocities[i]['vW'] = self.momentum * self.velocities[i]['vW'] - self.lr * dW
            self.velocities[i]['vb'] = self.momentum * self.velocities[i]['vb'] - self.lr * db

            # Parameter update
            param['W'] += self.velocities[i]['vW']
            param['b'] += self.velocities[i]['vb']
