from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from utils import accuracy, mae

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _setup_optim(model, model_name: str, train_cfg: Dict[str, float], default_epochs: int | None):
    epochs = default_epochs if default_epochs is not None else train_cfg.get("epochs", 600 if model_name == "ppgnn" else 200)
    optim = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.get("lr", 2e-3 if model_name == "ppgnn" else 1e-2),
        weight_decay=train_cfg.get("weight_decay", 5e-4 if model_name == "ppgnn" else 5e-4),
    )
    clip_value = train_cfg.get("clip_value", 5.0 if model_name == "ppgnn" else None)
    return epochs, optim, clip_value


def train_node_classification(data, model, model_name: str, train_cfg: Dict, default_epochs: int | None):
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

        if val_acc > best_val:
            best_val, best_test, best_epoch = val_acc, test_acc, epoch

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(
                f"Epoch: {epoch:03d}  train_loss:{train_loss:.3f}  val_loss:{val_loss:.3f}  test_loss:{test_loss:.3f}  "
                f"train_acc:{train_acc:.3f}  val_acc:{val_acc:.3f}  test_acc:{test_acc:.3f}"
            )

    print(f"★  Best epoch in evaluation split: Epoch={best_epoch}  val={best_val:.3f}  corresponding test={best_test:.3f}")

