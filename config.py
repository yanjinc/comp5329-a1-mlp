# config.py
"""
Global configuration for hyperparameters and model architecture.
Used for training the MLP network.
"""

# # Optimization hyperparameters
# learning_rate = 1e-3           # Learning rate for optimizer
# momentum = 1e-4                # Momentum factor for SGD
# weight_decay = 0.0005          # L2 regularization (weight decay)

# # Training loop settings
# batch_size = 100               # Mini-batch size
# num_epochs = 100               # Total training epochs

# # Model architecture
# input_dim = 128                # Input dimension (e.g., feature size)
# hidden_dims = [256, 128]       # Hidden layer sizes including output layer
# dropout_prob = 0               # Dropout probability for regularization

# # Output
# output_dim = 10                # Number of output classes


# """
# Used for Ablation Study
# """

# learning_rate = 1e-3
# momentum = 1e-4              
# weight_decay = 0.0005    
# batch_size = 100
# num_epochs = 100

# input_dim = 128
# hidden_dims = [512, 256, 128]
# dropout_prob = 0.0          
# output_dim = 10


"""
based on the best model from hyperparameter tuning
"""
learning_rate = 1e-4
momentum = 0.9              
weight_decay = 0.0005    
dropout_prob = 0.2    
batch_size = 64
num_epochs = 100

input_dim = 128
hidden_dims = [512, 256, 128]      
output_dim = 10