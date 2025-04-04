import numpy as np
import os
import itertools
import json
import matplotlib.pyplot as plt

from model_ablation import MLP
from loss import SoftmaxCrossEntropyLoss
from optimizer import SGD
from preprocess import standardize, one_hot_encode
from utils import shuffle_data, accuracy, confusion_matrix, precision_recall_f1
from config import input_dim, output_dim
from visualizer import plot_metrics

# --------------------- Load & Preprocess ---------------------
train_data = np.load('./dataset/train_data.npy')
train_label = np.load('./dataset/train_label.npy')
test_data = np.load('./dataset/test_data.npy')
test_label = np.load('./dataset/test_label.npy')

train_label = train_label.reshape(-1)
test_label = test_label.reshape(-1)

train_data = standardize(train_data)
test_data = standardize(test_data)

train_label = one_hot_encode(train_label, output_dim)

def run_experiment(learning_rate, dropout_prob, batch_size, hidden_dims, num_epochs=50):
    model = MLP(
        input_dim=input_dim,
        hidden_layers=hidden_dims,
        output_dim=output_dim,
        use_batchnorm=True,
        dropout_prob=dropout_prob,
        activation="relu"
    )

    loss_fn = SoftmaxCrossEntropyLoss()
    optimizer = SGD(model.get_parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    train_losses, train_accuracies = [], []
    for epoch in range(num_epochs):
        X_shuffled, y_shuffled = shuffle_data(train_data, train_label)
        batch_losses, batch_accs = [], []

        for i in range(0, X_shuffled.shape[0], batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            logits = model.forward(X_batch, training=True)
            loss = loss_fn.forward(logits, y_batch)
            grad = loss_fn.backward()

            model.backward(X_batch, grad, training=True)
            optimizer.step()

            acc = accuracy(loss_fn.probs, y_batch)
            batch_losses.append(loss)
            batch_accs.append(acc)

        train_losses.append(np.mean(batch_losses))
        train_accuracies.append(np.mean(batch_accs))
        print(f"[lr={learning_rate}, dropout={dropout_prob}, batch={batch_size}, hid={hidden_dims}] Epoch {epoch+1:03d} | Loss: {train_losses[-1]:.4f} | Acc: {train_accuracies[-1]:.4f}")

    # Evaluate
    logits = model.forward(test_data, training=False)
    _ = loss_fn.forward(logits, test_label)
    probs = loss_fn.probs

    test_acc = accuracy(probs, test_label)
    cm = confusion_matrix(probs, test_label, output_dim)
    precision, recall, f1 = precision_recall_f1(cm)

    return {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "test_acc": test_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "config": {
            "learning_rate": learning_rate,
            "dropout_prob": dropout_prob,
            "batch_size": batch_size,
            "hidden_dims": hidden_dims
        }
    }


# --------------------- Ablation Experiment ---------------------
learning_rates = [0.01, 0.001]
dropouts = [0.0, 0.3]
batches = [64, 128]
hidden_options = [[256, 128], [512, 256, 128]]

all_results = []
os.makedirs("./ablation_results", exist_ok=True)

for lr, drop, bs, hids in itertools.product(learning_rates, dropouts, batches, hidden_options):
    result = run_experiment(learning_rate=lr, dropout_prob=drop, batch_size=bs, hidden_dims=hids)
    all_results.append(result)

    # Save metrics
    config_str = f"lr{lr}_drop{drop}_bs{bs}_hid{'-'.join(map(str, hids))}"
    with open(f"./ablation_results/{config_str}.json", "w") as f:
        json.dump(result, f, indent=2)

    # Plot and save curves
    plot_metrics(
        result["train_losses"],
        result["train_losses"],
        result["train_accuracies"],
        result["train_accuracies"],
        save_path=f"./ablation_results/{config_str}.png"
    )

# Save all summary
with open("./ablation_results/all_summary.json", "w") as f:
    json.dump(all_results, f, indent=2)
