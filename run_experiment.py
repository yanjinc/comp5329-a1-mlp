import numpy as np
import os
import pickle
from model import MLP
from loss import SoftmaxCrossEntropyLoss
from optimizer import SGD, Adam
from preprocess import standardize, one_hot_encode
from utils import shuffle_data, accuracy, confusion_matrix, precision_recall_f1
from visualizer import plot_metrics



def run_experiment(config, trial_id=None):
    # --------------------- Load and preprocess dataset ---------------------
    train_data = np.load('./dataset/train_data.npy')
    train_label = np.load('./dataset/train_label.npy')
    test_data = np.load('./dataset/test_data.npy')
    test_label = np.load('./dataset/test_label.npy')

    train_label = train_label.reshape(-1)
    test_label = test_label.reshape(-1)

    train_data = standardize(train_data)
    test_data = standardize(test_data)

    train_label = one_hot_encode(train_label, config["output_dim"])
    test_label = one_hot_encode(test_label, config["output_dim"])

    # --------------------- Initialize model and optimizer ---------------------
    model = MLP(
        input_dim=config["input_dim"],
        hidden_dims=config["hidden_dims"],
        output_dim=config["output_dim"],
        dropout_prob=config["dropout_prob"],
        use_batchnorm=True,
        activation="relu"
    )
    loss_fn = SoftmaxCrossEntropyLoss()

    # Choose optimizer based on config
    if config.get("optimizer", "sgd") == "adam":
        optimizer = Adam(model.get_parameters(),
                        lr=config["learning_rate"],
                        weight_decay=config["weight_decay"])
    else:
        optimizer = SGD(model.get_parameters(),
                        lr=config["learning_rate"],
                        momentum=config["momentum"],
                        weight_decay=config["weight_decay"])

    train_losses, train_accs = [], []
    best_acc = 0.0
    os.makedirs("saved_models", exist_ok=True)

    # --------------------- Training ---------------------
    patience = config.get("early_stop_patience", None)
    threshold = config.get("early_stop_threshold", 0.0)
    no_improve_epochs = 0
    last_best_acc = 0.0

    for epoch in range(config["num_epochs"]):
        train_data, train_label = shuffle_data(train_data, train_label)
        epoch_losses, epoch_accs = [], []

        for i in range(0, len(train_data), config["batch_size"]):
            X_batch = train_data[i:i+config["batch_size"]]
            y_batch = train_label[i:i+config["batch_size"]]

            logits = model.forward(X_batch, training=True)
            loss = loss_fn.forward(logits, y_batch)
            grad = loss_fn.backward()

            model.backward(X_batch, grad, training=True)
            optimizer.step()
            optimizer.parameters = model.get_parameters()

            acc = accuracy(loss_fn.probs, y_batch)
            epoch_losses.append(loss)
            epoch_accs.append(acc)

        avg_loss = np.mean(epoch_losses)
        avg_acc = np.mean(epoch_accs)
        train_losses.append(avg_loss)
        train_accs.append(avg_acc)

        print(f"[{trial_id or 'Main'}] Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.4f}")

        # Save best model
        if avg_acc > best_acc + threshold:
            best_acc = avg_acc
            no_improve_epochs = 0
            model_path = f"saved_models/best_model_{trial_id or 'main'}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            print(f"Saved new best model at epoch {epoch+1} with accuracy {best_acc:.4f}")
        else:
            no_improve_epochs += 1

        # Early stopping
        if patience is not None and no_improve_epochs >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    # --------------------- Evaluation ---------------------
    logits = model.forward(test_data, training=False)
    loss_fn.forward(logits, test_label)
    probs = loss_fn.probs

    test_acc = accuracy(probs, test_label)
    cm = confusion_matrix(probs, test_label, config["output_dim"])
    precision, recall, f1 = precision_recall_f1(cm)

    print("\n--- Test Results ---")
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1 Score:       {f1:.4f}")

    # --------------------- Plot Training Metrics ---------------------
    plot_metrics(train_losses, train_losses, train_accs, train_accs,
                 save_path=f"saved_models/train_curve_{trial_id or 'main'}.png")

    return {
        "train_accs": train_accs,
        "train_losses": train_losses,
        "test_acc": test_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "best_acc": best_acc
    }


# --------------------- Run standalone ---------------------
if __name__ == "__main__":
    from config import (
        learning_rate, momentum, weight_decay, batch_size, num_epochs,
        dropout_prob, hidden_dims, input_dim, output_dim
    )

    config = {
        "learning_rate": learning_rate,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "dropout_prob": dropout_prob,
        "hidden_dims": hidden_dims,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "optimizer": "adam",             # or "sgd"
        "activation": "relu"             # or "tanh", "leaky_relu"
    }

