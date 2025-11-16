import os
import random
from typing import Any
import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import datetime
from tqdm import tqdm

import matplotlib.pyplot as plt


# Reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.accelerator.current_accelerator()


def save_checkpoint(model, model_dir, step):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(model_dir, f"checkpoint_{step}_{timestamp}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    mlflow.log_artifact(checkpoint_path, artifact_path=f"checkpoint_{step}")
    mlflow.pytorch.log_state_dict(
        model.state_dict(), artifact_path=f"checkpoint_{step}_state_dict"
    )


def get_dataloaders(train_set, val_set, batch_size, num_workers=4):
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        # num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        # num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def train_one_epoch(
    model,
    device,
    loader,
    loss_fn,
    optimizer,
    epoch,
    step,
    model_dir,
    save_every_n_steps,
):
    model.train()
    running_loss = 0.0
    running_policy_loss = 0.0
    running_value_loss = 0.0
    total = 0

    tqdm_loader = tqdm(loader, desc=f"Epoch {epoch} Training", unit="batch")
    for batch, (xb, mask, yb_policy, yb_value) in enumerate(tqdm_loader):
        xb, mask, yb_policy, yb_value = (
            xb.to(device),
            mask.to(device),
            yb_policy.to(device),
            yb_value.to(device),
        )
        optimizer.zero_grad()
        policy, value = model(xb)
        loss, policy_loss, value_loss = loss_fn(
            mask, *(policy, value), *(yb_policy, yb_value)
        )
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.shape[0]
        running_policy_loss += policy_loss.item() * xb.shape[0]
        running_value_loss += value_loss.item() * xb.shape[0]
        total += xb.shape[0]

        # Logging
        mlflow.log_metric("train_batch_loss", loss.item(), step=step[0])
        mlflow.log_metric("train_batch_policy_loss", policy_loss.item(), step=step[0])
        mlflow.log_metric("train_batch_value_loss", value_loss.item(), step=step[0])
        step[0] += 1
        if step[0] % save_every_n_steps == 0:
            save_checkpoint(model, model_dir, step[0])

    avg_loss = running_loss / total
    avg_policy_loss = running_policy_loss / total
    avg_value_loss = running_value_loss / total
    return avg_loss, avg_policy_loss, avg_value_loss


def evaluate(model, device, loader, loss_fn):
    model.eval()
    running_loss = 0.0
    running_policy_loss = 0.0
    running_value_loss = 0.0
    total = 0
    with torch.no_grad():
        for xb, mask, yb_policy, yb_value in loader:
            xb, mask, yb_policy, yb_value = (
                xb.to(device),
                mask.to(device),
                yb_policy.to(device),
                yb_value.to(device),
            )
            policy, value = model(xb)
            loss, policy_loss, value_loss = loss_fn(
                mask, *(policy, value), *(yb_policy, yb_value)
            )

            running_loss += loss.item() * xb.shape[0]
            running_policy_loss += policy_loss.item() * xb.shape[0]
            running_value_loss += value_loss.item() * xb.shape[0]
            total += xb.shape[0]

    avg_loss = running_loss / total
    avg_policy_loss = running_policy_loss / total
    avg_value_loss = running_value_loss / total
    return avg_loss, avg_policy_loss, avg_value_loss


# def plot_metrics(history, out_path):
#     epochs = list(range(1, len(history["train_loss"]) + 1))
#     plt.figure(figsize=(10, 4))

#     plt.subplot(1, 3, 1)
#     plt.plot(epochs, history["train_loss"], label="train_loss")
#     plt.plot(epochs, history["val_loss"], label="val_loss")
#     plt.xlabel("epoch")
#     plt.ylabel("loss")
#     plt.legend()

#     plt.subplot(1, 3, 2)
#     plt.plot(epochs, history["train_policy_loss"], label="train_policy_loss")
#     plt.plot(epochs, history["val_policy_loss"], label="val_policy_loss")
#     plt.xlabel("epoch")
#     plt.ylabel("policy loss")
#     plt.legend()

#     plt.subplot(1, 3, 3)
#     plt.plot(epochs, history["train_value_loss"], label="train_value_loss")
#     plt.plot(epochs, history["val_value_loss"], label="val_value_loss")
#     plt.xlabel("epoch")
#     plt.ylabel("value loss")
#     plt.legend()

#     plt.tight_layout()
#     plt.savefig(out_path)
#     plt.close()


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
    scheduler: Any,
    epochs: int,
    batch_size: int,
    train_set: torch.utils.data.Dataset,
    val_set: torch.utils.data.Dataset,
    num_workers: int = 4,
    seed: int = 42,
    save_every_n_steps: int = 5000,
):
    set_seed(seed)

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

        mlflow.set_tag("model_dir", str(model_dir))

        os.makedirs(model_dir)
        step = [1]

        for epoch in range(1, epochs + 1):
            train_loss, train_policy_loss, train_value_loss = train_one_epoch(
                model,
                device,
                train_loader,
                loss_fn,
                optimizer,
                epoch,
                step,
                model_dir,
                save_every_n_steps,
            )
            val_loss, val_policy_loss, val_value_loss = evaluate(
                model, device, val_loader, loss_fn
            )
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_policy_loss"].append(train_policy_loss)
            history["val_policy_loss"].append(val_policy_loss)
            history["train_value_loss"].append(train_value_loss)
            history["val_value_loss"].append(val_value_loss)

            # Log metrics for this epoch
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("train_policy_loss", train_policy_loss, step=epoch)
            mlflow.log_metric("val_policy_loss", val_policy_loss, step=epoch)
            mlflow.log_metric("train_value_loss", train_value_loss, step=epoch)
            mlflow.log_metric("val_value_loss", val_value_loss, step=epoch)
            print(
                f"Epoch {epoch}/{epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_policy_loss={val_policy_loss:.4f} val_value_loss={val_value_loss:.4f} train_policy_loss={train_policy_loss:.4f} train_value_loss={train_value_loss:.4f}"
            )

            save_checkpoint(model, model_dir, step[0])

        # Save model and log as artifact
        model_path = os.path.join(model_dir, "model.pth")
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path, artifact_path="model")
        mlflow.pytorch.log_state_dict(
            model.state_dict(), artifact_path="model_state_dict"
        )

        # Plot and log metrics figure
        # metrics_png = os.path.join(model_dir, "metrics.png")
        # plot_metrics(history, metrics_png)
        # mlflow.log_artifact(metrics_png, artifact_path="plots")

        print("Training complete. Artifacts and metrics logged to MLflow.")
