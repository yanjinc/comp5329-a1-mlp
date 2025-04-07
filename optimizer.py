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


class Adam:
    """
    Adam Optimizer.
    """
    def __init__(self, parameters, lr=0.001, weight_decay=0.0, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = []
        self.v = []
        self.t = 0
        for param in self.parameters:
            self.m.append({
                'mW': np.zeros_like(param['W']),
                'mb': np.zeros_like(param['b']),
            })
            self.v.append({
                'vW': np.zeros_like(param['W']),
                'vb': np.zeros_like(param['b']),
            })

    def step(self):
        self.t += 1
        for i, param in enumerate(self.parameters):
            dW, db = param['dW'], param['db']
            if dW is None or db is None:
                continue

            if self.weight_decay > 0:
                dW += self.weight_decay * param['W']

            # Update moment estimates
            self.m[i]['mW'] = self.beta1 * self.m[i]['mW'] + (1 - self.beta1) * dW
            self.m[i]['mb'] = self.beta1 * self.m[i]['mb'] + (1 - self.beta1) * db
            self.v[i]['vW'] = self.beta2 * self.v[i]['vW'] + (1 - self.beta2) * (dW ** 2)
            self.v[i]['vb'] = self.beta2 * self.v[i]['vb'] + (1 - self.beta2) * (db ** 2)

            # Bias correction
            mW_hat = self.m[i]['mW'] / (1 - self.beta1 ** self.t)
            mb_hat = self.m[i]['mb'] / (1 - self.beta1 ** self.t)
            vW_hat = self.v[i]['vW'] / (1 - self.beta2 ** self.t)
            vb_hat = self.v[i]['vb'] / (1 - self.beta2 ** self.t)

            # Update parameters
            param['W'] -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.eps)
            param['b'] -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)
