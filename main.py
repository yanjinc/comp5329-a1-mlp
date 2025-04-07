import numpy as np
import pickle
import os

from model import MLP
from loss import SoftmaxCrossEntropyLoss
from optimizer import SGD, Adam
from preprocess import one_hot_encode, standardize
from utils import shuffle_data, accuracy, confusion_matrix, precision_recall_f1
from config import (
    learning_rate, momentum, weight_decay, batch_size, num_epochs,
    dropout_prob, hidden_dims, input_dim, output_dim
)
from visualizer import plot_metrics

import random
np.random.seed(42)
random.seed(42)

# --------------------- Load dataset ---------------------
train_data = np.load('./dataset/train_data.npy')
train_label = np.load('./dataset/train_label.npy')
test_data = np.load('./dataset/test_data.npy')
test_label = np.load('./dataset/test_label.npy')

# --------------------- Preprocess ---------------------
train_label = train_label.reshape(-1)
test_label = test_label.reshape(-1)

train_data = standardize(train_data)
test_data = standardize(test_data)

train_label = one_hot_encode(train_label, output_dim)
test_label = one_hot_encode(test_label, output_dim)

# --------------------- Initialize ---------------------
model = MLP(
    input_dim=input_dim,
    hidden_dims=hidden_dims,
    output_dim=output_dim,
    use_batchnorm=True,        
    dropout_prob=dropout_prob,
    activation="relu"
)
loss_fn = SoftmaxCrossEntropyLoss()
# optimizer = SGD(model.get_parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
optimizer = Adam(model.get_parameters(), lr=learning_rate, beta1=0.9, beta2=0.999, weight_decay=weight_decay)

# --------------------- Training Loop ---------------------
train_losses = []
train_accuracies = []
best_acc = 0.0
early_stop_patience = 5
early_stop_threshold = 0.005
no_improve_count = 0
best_f1 = 0.0

os.makedirs("saved_models", exist_ok=True)  

for epoch in range(num_epochs):
    train_data, train_label = shuffle_data(train_data, train_label)
    num_samples = train_data.shape[0]
    batch_losses = []
    batch_accs = []

    for i in range(0, num_samples, batch_size):
        X_batch = train_data[i:i+batch_size]
        y_batch = train_label[i:i+batch_size]

        logits = model.forward(X_batch, training=True)
        loss = loss_fn.forward(logits, y_batch)
        grad = loss_fn.backward()

        model.backward(X_batch, grad, training=True)
        optimizer.step()

        optimizer.parameters = model.get_parameters()

        acc = accuracy(loss_fn.probs, y_batch)
        batch_losses.append(loss)
        batch_accs.append(acc)

    epoch_loss = np.mean(batch_losses)
    epoch_acc = np.mean(batch_accs)
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    print(f"Epoch {epoch+1:03d} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")

    # Evaluate on test set for early stopping
    val_logits = model.forward(test_data, training=False)
    _ = loss_fn.forward(val_logits, test_label)
    probs = loss_fn.probs
    _, _, f1 = precision_recall_f1(confusion_matrix(probs, test_label, output_dim))

    if f1 > best_f1 + early_stop_threshold:
        best_f1 = f1
        no_improve_count = 0
        with open("saved_models/best_model.pkl", "wb") as f:
            pickle.dump(model, f)
        print(f"Saved new best model at epoch {epoch+1} with F1: {f1:.4f}")
    else:
        no_improve_count += 1

    if no_improve_count >= early_stop_patience:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break


# --------------------- Evaluation on Test Set ---------------------
logits = model.forward(test_data, training=False)
_ = loss_fn.forward(logits, test_label)
probs = loss_fn.probs

test_acc = accuracy(probs, test_label)
cm = confusion_matrix(probs, test_label, output_dim)
precision, recall, f1 = precision_recall_f1(cm)

print("\n--- Test Results ---")
print(f"Test Accuracy:  {test_acc:.4f}")
print(f"Precision:      {precision:.4f}")
print(f"Recall:         {recall:.4f}")
print(f"F1 Score:       {f1:.4f}")

# --------------------- Plot Loss / Accuracy ---------------------
plot_metrics(train_losses, train_losses, train_accuracies, train_accuracies)
