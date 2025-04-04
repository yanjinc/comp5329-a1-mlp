import numpy as np
from activations import relu, relu_derivative, sigmoid, sigmoid_derivative, softmax
from layers import Linear, Dropout, BatchNorm

class MLP:
    def __init__(self, input_dim, hidden_layers, output_dim, 
                 activation='relu', dropout_prob=0.0, use_batchnorm=True):
        self.layers = []
        self.activation = activation
        self.use_batchnorm = use_batchnorm
        self.dropout_prob = dropout_prob
        layer_dims = [input_dim] + hidden_layers

        for i in range(len(hidden_layers)):
            self.layers.append(("linear", Linear(layer_dims[i], layer_dims[i+1])))
            if use_batchnorm:
                self.layers.append(("batchnorm", BatchNorm(layer_dims[i+1])))
            self.layers.append(("activation", self.activation))
            if dropout_prob > 0.0:
                self.layers.append(("dropout", Dropout(dropout_prob)))
        
        # Output layer (no activation)
        self.layers.append(("linear", Linear(layer_dims[-1], output_dim)))

    def forward(self, x, training=True):
        self.cache = []
        out = x
        for name, layer in self.layers:
            if name == "linear":
                out = layer.forward(out)
                self.cache.append(out)
            elif name == "batchnorm":
                out = layer.forward(out, training=training)
                self.cache.append(out)
            elif name == "activation":
                if self.activation == "relu":
                    out = relu(out)
                elif self.activation == "sigmoid":
                    out = sigmoid(out)
                self.cache.append(out)
            elif name == "dropout":
                out = layer.forward(out, training=training)
                self.cache.append(out)
        return out

    def backward(self, x, grad_output, training=True):
        grad = grad_output
        for i in reversed(range(len(self.layers))):
            name, layer = self.layers[i]
            cache_input = x if i == 0 else self.cache[i-1]

            if name == "activation":
                if self.activation == "relu":
                    grad = grad * relu_derivative(self.cache[i-1])
                elif self.activation == "sigmoid":
                    grad = grad * sigmoid_derivative(self.cache[i-1])
            elif name == "dropout":
                grad = layer.backward(grad)
            elif name == "batchnorm":
                grad = layer.backward(grad)
            elif name == "linear":
                grad = layer.backward(cache_input, grad)

    def update(self, optimizer):
        for _, layer in self.layers:
            if isinstance(layer, Linear) or isinstance(layer, BatchNorm):
                optimizer.update_param(layer, bn=isinstance(layer, BatchNorm))

    def predict(self, x):
        out = x
        for name, layer in self.layers:
            if name == "linear":
                out = layer.forward(out)
            elif name == "batchnorm":
                out = layer.forward(out, training=False)
            elif name == "activation":
                out = relu(out) if self.activation == "relu" else sigmoid(out)
            elif name == "dropout":
                continue  # Skip dropout in inference
        return np.argmax(softmax(out), axis=1)

    def get_parameters(self):
        params = []
        for _, layer in self.layers:
            if isinstance(layer, Linear):
                params.append({
                    'W': layer.W,
                    'b': layer.b,
                    'dW': layer.dW,
                    'db': layer.db
                })
        return params
