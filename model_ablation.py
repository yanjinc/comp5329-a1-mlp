"""
Modified model.py MLP class for ablation study
"""

# model_ablation.py

import numpy as np
from activations import relu, relu_derivative, tanh, tanh_derivative, softmax
from layers import Linear, Dropout, BatchNorm

class MLP:
    def __init__(self, input_dim, hidden_layers, output_dim, 
                 dropout_prob=0.0, use_batchnorm=True, activation='relu'):
        self.layers = []
        self.activations = []
        self.use_batchnorm = use_batchnorm
        self.dropout_prob = dropout_prob
        self.activation_name = activation.lower()

        # Select activation functions
        if self.activation_name == 'relu':
            self.act_fn = relu
            self.act_fn_deriv = relu_derivative
        elif self.activation_name == 'tanh':
            self.act_fn = tanh
            self.act_fn_deriv = tanh_derivative
        else:
            raise ValueError("Unsupported activation: use 'relu' or 'tanh'")

        # Construct hidden layers
        dims = [input_dim] + hidden_layers
        for i in range(len(dims) - 1):
            self.layers.append(Linear(dims[i], dims[i+1]))
            if self.use_batchnorm:
                self.layers.append(BatchNorm(dims[i+1]))
            self.layers.append(self.activation_name)  # e.g., 'relu'
            if self.dropout_prob > 0:
                self.layers.append(Dropout(self.dropout_prob))

        # Output layer (Linear only)
        self.layers.append(Linear(dims[-1], output_dim))

        self.cache = []

    def forward(self, x, training=True):
        out = x
        self.cache = []

        for layer in self.layers:
            if isinstance(layer, Linear):
                out = layer.forward(out)
                self.cache.append(out)
            elif isinstance(layer, BatchNorm):
                out = layer.forward(out, training=training)
                self.cache.append(out)
            elif isinstance(layer, Dropout):
                out = layer.forward(out, training=training)
                self.cache.append(out)
            elif layer == 'relu' or layer == 'tanh':
                out = self.act_fn(out)
                self.cache.append(out)

        return out  # logits

    def backward(self, x, grad_output, training=True):
        grad = grad_output

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            input_to_layer = x if i == 0 else self.cache[i - 1]

            if isinstance(layer, Linear):
                grad = layer.backward(input_to_layer, grad)
            elif isinstance(layer, BatchNorm):
                grad = layer.backward(grad)
            elif isinstance(layer, Dropout):
                grad = layer.backward(grad)
            elif layer == 'relu' or layer == 'tanh':
                grad = grad * self.act_fn_deriv(input_to_layer)

    def update(self, optimizer):
        for layer in self.layers:
            if isinstance(layer, Linear):
                optimizer.update_param(layer)
            elif isinstance(layer, BatchNorm):
                optimizer.update_param(layer, bn=True)

    def predict(self, x):
        out = x
        for layer in self.layers:
            if isinstance(layer, Linear):
                out = layer.forward(out)
            elif isinstance(layer, BatchNorm):
                out = layer.forward(out, training=False)
            elif isinstance(layer, Dropout):
                continue  # skip dropout during inference
            elif layer == 'relu' or layer == 'tanh':
                out = self.act_fn(out)
        return np.argmax(softmax(out), axis=1)

    def get_parameters(self):
        params = []
        for layer in self.layers:
            if isinstance(layer, Linear):
                params.append({
                    'W': layer.W,
                    'b': layer.b,
                    'dW': layer.dW,
                    'db': layer.db
                })
        return params
