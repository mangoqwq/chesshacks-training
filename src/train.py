import os
import random
import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import datetime

import matplotlib.pyplot as plt


# Reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_dataloaders(train_set, val_set, batch_size, num_workers=4):
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def train_one_epoch(model, device, loader, loss_fn, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.shape[0]
        preds = logits.argmax(dim=1)
        batch_correct = (preds == yb).sum().item()
        correct += batch_correct
        total += xb.shape[0]

        # Logging
        mlflow.log_metric(
            "train_batch_loss", loss.item(), step=(epoch - 1) * len(loader) + batch
        )
        mlflow.log_metric(
            "train_batch_acc",
            batch_correct / xb.shape[0],
            step=(epoch - 1) * len(loader) + batch,
        )

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def evaluate(model, device, loader, loss_fn):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)

            running_loss += loss.item() * xb.shape[0]
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.shape[0]

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def plot_metrics(history, out_path):
    epochs = list(range(1, len(history["train_loss"]) + 1))
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def optimizer_parameter_log(optimizer):
    param_groups = []
    for param_group in optimizer.param_groups:
        param_groups.append(
            {key: value for key, value in param_group.items() if key != "params"}
        )

    return param_groups


def train(
    model_name: str,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epochs: int,
    batch_size: int,
    train_set: torch.utils.data.Dataset,
    val_set: torch.utils.data.Dataset,
    num_workers: int = 4,
):
    set_seed()

    device = get_device()

    train_loader, val_loader = get_dataloaders(
        train_set, val_set, batch_size, num_workers
    )

    model.to(device)

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("optimizer", optimizer.__class__.__name__)
        mlflow.log_param("loss_function", loss_fn.__class__.__name__)
        mlflow.log_param("optimizer_parameters", optimizer_parameter_log(optimizer))
        mlflow.set_tag("model_type", model.__class__.__name__)

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        model_dir = os.path.join(
            "models", model_name, f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
        )

        os.makedirs(model_dir)

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, device, train_loader, loss_fn, optimizer, epoch
            )
            val_loss, val_acc = evaluate(model, device, val_loader, loss_fn)
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            # Log metrics for this epoch
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)
            print(
                f"Epoch {epoch}/{epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            # Save Checkpoint
            checkpoint_path = os.path.join(model_dir, f"checkpoint_{epoch}.pth")
            torch.save({"state_dict": model.state_dict()}, checkpoint_path)
            mlflow.log_artifact(checkpoint_path, artifact_path=f"checkpoint_{epoch}")
            mlflow.pytorch.log_state_dict(
                model.state_dict(), artifact_path=f"checkpoint_{epoch}_state_dict"
            )

        # Save model and log as artifact
        model_path = os.path.join(model_dir, "model.pth")
        torch.save({"state_dict": model.state_dict()}, model_path)
        mlflow.log_artifact(model_path, artifact_path="model")
        mlflow.pytorch.log_state_dict(
            model.state_dict(), artifact_path="model_state_dict"
        )

        # Plot and log metrics figure
        metrics_png = os.path.join(model_dir, "metrics.png")
        plot_metrics(history, metrics_png)
        mlflow.log_artifact(metrics_png, artifact_path="plots")

        print("Training complete. Artifacts and metrics logged to MLflow.")
