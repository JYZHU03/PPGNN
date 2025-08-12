from __future__ import annotations

"""Training routines for different task types.

Each task is categorised by its prediction level (``node`` or ``graph``)
and objective (``classification`` or ``regression``).  This file exposes a
set of trainer functions and a :data:`TRAINERS` dispatch table to allow a
clean mapping from task type to the correct training loop.
"""

from typing import Dict, Iterable, Tuple, Callable

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from utils import accuracy, mae

try:  # Optional MLflow dependency
    import mlflow  # type: ignore
except Exception:  # pragma: no cover - allow training without MLflow
    mlflow = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _setup_optim(
    model, model_name: str, train_cfg: Dict[str, float], default_epochs: int | None
) -> Tuple[int, torch.optim.Optimizer, float | None]:
    epochs = (
        default_epochs
        if default_epochs is not None
        else train_cfg.get("epochs", 600 if model_name == "ppgnn" else 200)
    )
    optim = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.get("lr", 2e-3 if model_name == "ppgnn" else 1e-2),
        weight_decay=train_cfg.get("weight_decay", 5e-4),
    )
    clip_value = train_cfg.get("clip_value", 5.0 if model_name == "ppgnn" else None)
    return epochs, optim, clip_value


# ---------------------------------------------------------------------------
# Node-level tasks
# ---------------------------------------------------------------------------

def train_node_classification(
    data, model, model_name: str, train_cfg: Dict, default_epochs: int | None
) -> None:
    model = model.to(DEVICE)
    data = data.to(DEVICE)
    epochs, optim, clip_value = _setup_optim(model, model_name, train_cfg, default_epochs)

    best_val = best_test = best_epoch = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        optim.zero_grad()
        out = model(data)
        train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
        if train_mask.ndim > 1:
            train_mask = train_mask[:, 0]
            val_mask = val_mask[:, 0]
            test_mask = test_mask[:, 0]
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        if clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optim.step()

        model.eval()
        with torch.no_grad():
            logits = model(data)
            train_loss = F.cross_entropy(logits[train_mask], data.y[train_mask])
            val_loss = F.cross_entropy(logits[val_mask], data.y[val_mask])
            test_loss = F.cross_entropy(logits[test_mask], data.y[test_mask])
            train_acc = accuracy(logits[train_mask], data.y[train_mask])
            val_acc = accuracy(logits[val_mask], data.y[val_mask])
            test_acc = accuracy(logits[test_mask], data.y[test_mask])

        if mlflow is not None:
            mlflow.log_metrics(
                {
                    "train_loss": float(train_loss.item()),
                    "val_loss": float(val_loss.item()),
                    "test_loss": float(test_loss.item()),
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "test_acc": test_acc,
                },
                step=epoch,
            )

        if val_acc > best_val:
            best_val, best_test, best_epoch = val_acc, test_acc, epoch

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(
                f"Epoch: {epoch:03d}  train_loss:{train_loss:.3f}  val_loss:{val_loss:.3f}  test_loss:{test_loss:.3f}  "
                f"train_acc:{train_acc:.3f}  val_acc:{val_acc:.3f}  test_acc:{test_acc:.3f}"
            )

    print(
        f"★  Best epoch in evaluation split: Epoch={best_epoch}  val={best_val:.3f}  corresponding test={best_test:.3f}"
    )
    if mlflow is not None:
        mlflow.log_metrics(
            {"best_val": best_val, "best_test": best_test}, step=best_epoch
        )


def train_node_regression(
    data, model, model_name: str, train_cfg: Dict, default_epochs: int | None
) -> None:
    model = model.to(DEVICE)
    data = data.to(DEVICE)
    epochs, optim, clip_value = _setup_optim(model, model_name, train_cfg, default_epochs)

    best_val = float("inf")
    best_test = float("inf")
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optim.zero_grad()
        out = model(data)
        train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
        if train_mask.ndim > 1:
            train_mask = train_mask[:, 0]
            val_mask = val_mask[:, 0]
            test_mask = test_mask[:, 0]
        loss = F.l1_loss(out[train_mask], data.y[train_mask].view_as(out[train_mask]))
        loss.backward()
        if clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optim.step()

        model.eval()
        with torch.no_grad():
            pred = model(data)
            train_mae = mae(pred[train_mask], data.y[train_mask].view_as(pred[train_mask]))
            val_mae = mae(pred[val_mask], data.y[val_mask].view_as(pred[val_mask]))
            test_mae = mae(pred[test_mask], data.y[test_mask].view_as(pred[test_mask]))

        if mlflow is not None:
            mlflow.log_metrics(
                {
                    "train_mae": train_mae,
                    "val_mae": val_mae,
                    "test_mae": test_mae,
                },
                step=epoch,
            )

        if val_mae < best_val:
            best_val, best_test, best_epoch = val_mae, test_mae, epoch

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(
                f"Epoch: {epoch:03d}  train_mae:{train_mae:.3f}  val_mae:{val_mae:.3f}  test_mae:{test_mae:.3f}"
            )

    print(
        f"★  Best epoch in evaluation split: Epoch={best_epoch}  val_mae={best_val:.3f}  corresponding test_mae={best_test:.3f}"
    )
    if mlflow is not None:
        mlflow.log_metrics(
            {"best_val_mae": best_val, "best_test_mae": best_test}, step=best_epoch
        )


# ---------------------------------------------------------------------------
# Graph-level tasks
# ---------------------------------------------------------------------------

def train_graph_classification(
    loaders: Dict[str, DataLoader],
    model,
    model_name: str,
    train_cfg: Dict,
    default_epochs: int | None,
) -> None:
    model = model.to(DEVICE)
    epochs, optim, clip_value = _setup_optim(model, model_name, train_cfg, default_epochs)

    def evaluate(loader: DataLoader) -> Tuple[float, float]:
        model.eval()
        total_loss = correct = count = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(DEVICE)
                out = model(batch)
                out = global_mean_pool(out, batch.batch)
                loss = F.cross_entropy(out, batch.y)
                total_loss += loss.item() * batch.num_graphs
                correct += (out.argmax(dim=-1) == batch.y).sum().item()
                count += batch.num_graphs
        return total_loss / count, correct / count

    best_val = best_test = best_epoch = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in loaders["train"]:
            batch = batch.to(DEVICE)
            optim.zero_grad()
            out = model(batch)
            out = global_mean_pool(out, batch.batch)
            loss = F.cross_entropy(out, batch.y)
            loss.backward()
            if clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optim.step()

        train_loss, train_acc = evaluate(loaders["train"])
        val_loss, val_acc = evaluate(loaders["val"])
        test_loss, test_acc = evaluate(loaders["test"])

        if mlflow is not None:
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "test_loss": test_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "test_acc": test_acc,
                },
                step=epoch,
            )

        if val_acc > best_val:
            best_val, best_test, best_epoch = val_acc, test_acc, epoch

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(
                f"Epoch: {epoch:03d}  train_loss:{train_loss:.3f}  val_loss:{val_loss:.3f}  test_loss:{test_loss:.3f}  "
                f"train_acc:{train_acc:.3f}  val_acc:{val_acc:.3f}  test_acc:{test_acc:.3f}"
            )

    print(
        f"★  Best epoch in evaluation split: Epoch={best_epoch}  val_acc={best_val:.3f}  corresponding test_acc={best_test:.3f}"
    )
    if mlflow is not None:
        mlflow.log_metrics(
            {"best_val": best_val, "best_test": best_test}, step=best_epoch
        )


def train_graph_regression(
    loaders: Dict[str, DataLoader],
    model,
    model_name: str,
    train_cfg: Dict,
    default_epochs: int | None,
) -> None:
    model = model.to(DEVICE)
    epochs, optim, clip_value = _setup_optim(model, model_name, train_cfg, default_epochs)

    def evaluate(loader: DataLoader) -> float:
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(DEVICE)
                out = model(batch)
                out = global_mean_pool(out, batch.batch)
                preds.append(out.detach())
                trues.append(batch.y.view_as(out).detach())
        pred = torch.cat(preds, dim=0)
        true = torch.cat(trues, dim=0)
        return mae(pred, true)

    best_val = float("inf")
    best_test = float("inf")
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in loaders["train"]:
            batch = batch.to(DEVICE)
            optim.zero_grad()
            out = model(batch)
            out = global_mean_pool(out, batch.batch)
            loss = F.l1_loss(out, batch.y.view_as(out))
            loss.backward()
            if clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optim.step()

        train_mae = evaluate(loaders["train"])
        val_mae = evaluate(loaders["val"])
        test_mae = evaluate(loaders["test"])
        if mlflow is not None:
            mlflow.log_metrics(
                {
                    "train_mae": train_mae,
                    "val_mae": val_mae,
                    "test_mae": test_mae,
                },
                step=epoch,
            )
        if val_mae < best_val:
            best_val, best_test, best_epoch = val_mae, test_mae, epoch

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(
                f"Epoch: {epoch:03d}  train_mae:{train_mae:.4f}  val_mae:{val_mae:.4f}  test_mae:{test_mae:.4f}"
            )

    print(
        f"★  Best epoch in evaluation split: Epoch={best_epoch}  val_mae={best_val:.4f}  corresponding test_mae={best_test:.4f}"
    )
    if mlflow is not None:
        mlflow.log_metrics(
            {"best_val_mae": best_val, "best_test_mae": best_test}, step=best_epoch
        )


# ---------------------------------------------------------------------------
# Trainer dispatch table
# ---------------------------------------------------------------------------

TRAINERS: Dict[Tuple[str, str], Callable] = {
    ("node", "classification"): train_node_classification,
    ("node", "regression"): train_node_regression,
    ("graph", "classification"): train_graph_classification,
    ("graph", "regression"): train_graph_regression,
}