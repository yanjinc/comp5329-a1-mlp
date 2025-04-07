import numpy as np
from layers import Linear
from activations import (
    relu, relu_derivative,
    tanh, tanh_derivative,
    leaky_relu, leaky_relu_derivative
)

class MLP:
    def __init__(self, input_dim, hidden_dims, output_dim, activation="relu", use_batchnorm=False, dropout_prob=0.0):
        self.layers = []
        self.activation_name = activation.lower()
        self.use_batchnorm = use_batchnorm
        self.dropout_prob = dropout_prob
        self.training = True

        layer_dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(layer_dims) - 1):
            self.layers.append(Linear(layer_dims[i], layer_dims[i+1],
                                      use_batchnorm=(use_batchnorm and i < len(hidden_dims)),
                                      dropout_prob=(dropout_prob if i < len(hidden_dims) else 0.0)))

    def forward(self, X, training=True):
        self.training = training
        out = X
        for i, layer in enumerate(self.layers):
            out = layer.forward(out, training=training)
            if i < len(self.layers) - 1:  # activation before last layer
                out = self._activate(out)
        return out

    def backward(self, X, grad, training=True):
        dout = grad
        for i in reversed(range(len(self.layers))):
            if i < len(self.layers) - 1:
                dout = self._activate_backward(self.layers[i].cache, dout)
            dout = self.layers[i].backward(dout, training=training)

    def get_parameters(self):
        params = []
        for layer in self.layers:
            params.append(layer.get_parameters())
        return params

    def _activate(self, x):
        if self.activation_name == "relu":
            return relu(x)
        elif self.activation_name == "tanh":
            return tanh(x)
        elif self.activation_name == "leaky_relu":
            return leaky_relu(x)
        else:
            raise ValueError(f"Unsupported activation: {self.activation_name}")

    def _activate_backward(self, cache, dout):
        if self.activation_name == "relu":
            return dout * relu_derivative(cache)
        elif self.activation_name == "tanh":
            return dout * tanh_derivative(cache)
        elif self.activation_name == "leaky_relu":
            return dout * leaky_relu_derivative(cache)
        else:
            raise ValueError(f"Unsupported activation: {self.activation_name}")
