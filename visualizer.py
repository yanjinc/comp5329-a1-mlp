import matplotlib.pyplot as plt

def plot_metrics(train_losses, val_losses, train_accs=None, val_accs=None, save_path="training_curve.png"):
    """
    Plots training and validation loss and accuracy curves.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 5))

    # Subplot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    plt.plot(epochs, val_losses, label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.grid(True)
    plt.legend()

    # Subplot 2: Accuracy
    if train_accs is not None and val_accs is not None:
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accs, label="Train Accuracy", linewidth=2)
        plt.plot(epochs, val_accs, label="Val Accuracy", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Curve")
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()
    plt.close()
