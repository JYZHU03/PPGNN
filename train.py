from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Any

try:  # Optional YAML dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback if PyYAML is absent
    yaml = None
import json
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeFeatures, ToUndirected, Compose
from torch_geometric.nn import global_mean_pool

from utils import set_seed, accuracy, mae
from models import PPGNN, GCN, GraphSAGE, GAT


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(name: str, in_channels: int, num_classes: int, cfg: Dict[str, Any] | None = None):
    """Create a model by name with hyper-parameters from *cfg*."""
    cfg = cfg or {}
    params = dict(in_channels=in_channels, hidden=cfg.get("hidden", 64), num_classes=num_classes)
    if name == "ppgnn":
        return PPGNN(
            **params,
            layers=cfg.get("layers", 15),
            dt=cfg.get("dt", 0.1),
            dropout=cfg.get("dropout", 0.4),
            norm_type=cfg.get("norm_type", "sym"),
            jacobi_steps=cfg.get("jacobi_steps", 2),
            use_x_only=cfg.get("use_x_only", True),
            y0_mode=cfg.get("y0_mode", "ones"),
        )
    if name == "gcn":
        return GCN(**params, layers=cfg.get("layers", 2))
    if name == "sage":
        return GraphSAGE(**params, layers=cfg.get("layers", 2))
    if name == "gat":
        return GAT(**params, heads=cfg.get("heads", 8), layers=cfg.get("layers", 2))
    raise ValueError(name)


def load_yaml_config(dataset: str, cfg_dir: str = "configs") -> Dict[str, Any]:
    """Load configuration for *dataset* from *cfg_dir*.

    If PyYAML is unavailable, the file is parsed as JSON.
    """
    path = Path(cfg_dir) / f"{dataset}.yaml"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        if yaml is not None:
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    return data or {}


def load_dataset(name: str) -> Dict:
    """Load dataset *name* and return a description dictionary."""
    if name in {"Cora", "CiteSeer", "PubMed"}:
        dataset = Planetoid(
            root="data/Planetoid",
            name=name,
            transform=Compose([NormalizeFeatures(), ToUndirected()]),
        )
        data = dataset[0].to(DEVICE)
        return dict(
            task="node",
            in_channels=dataset.num_features,
            num_classes=dataset.num_classes,
            data=data,
        )

    # 如需 graph-level 任务，按需开启
    # from torch_geometric.datasets import PeptidesStructuralDataset
    # if name == "peptides-struct":
    #     train_ds = PeptidesStructuralDataset(root="data/Peptides", split="train")
    #     val_ds = PeptidesStructuralDataset(root="data/Peptides", split="val")
    #     test_ds = PeptidesStructuralDataset(root="data/Peptides", split="test")
    #     loaders = {
    #         "train": DataLoader(train_ds, batch_size=64, shuffle=True),
    #         "val": DataLoader(val_ds, batch_size=128),
    #         "test": DataLoader(test_ds, batch_size=128),
    #     }
    #     target_dim = train_ds[0].y.size(-1)
    #     return dict(
    #         task="graph",
    #         in_channels=train_ds.num_features,
    #         num_classes=target_dim,
    #         loaders=loaders,
    #     )

    raise ValueError(f"Unknown dataset: {name}")


def train_node(data, model_name: str, num_classes: int, cfg: Dict[str, Any], default_epochs: int | None):
    model = get_model(model_name, data.num_features, num_classes, cfg.get("model")).to(DEVICE)

    train_cfg = cfg.get("train", {})
    epochs = default_epochs if default_epochs is not None else train_cfg.get("epochs", 600)
    optim = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.get("lr", 2e-3 if model_name == "ppgnn" else 1e-2),
        weight_decay=train_cfg.get("weight_decay", 5e-4 if model_name == "ppgnn" else 5e-4),
    )
    clip_value = train_cfg.get("clip_value", 5.0 if model_name == "ppgnn" else None)

    best_val = best_test = 0.0
    for epoch in range(1, epochs + 1):
        # train
        model.train()
        optim.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        if clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optim.step()

        # eval
        model.eval()
        with torch.no_grad():
            logits = model(data)
            val_acc = accuracy(logits[data.val_mask], data.y[data.val_mask])
            test_acc = accuracy(logits[data.test_mask], data.y[data.test_mask])

        if val_acc > best_val:
            best_val, best_test = val_acc, test_acc

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(f"E{epoch:03d}  loss:{loss:.3f}  val:{val_acc:.3f}  test:{test_acc:.3f}")

    print(f"★ Best val={best_val:.3f}  corresponding test={best_test:.3f}")


def train_graph(loaders: Dict[str, DataLoader], model_name: str, in_channels: int, num_targets: int, cfg: Dict[str, Any], default_epochs: int | None):
    model = get_model(model_name, in_channels, num_targets, cfg.get("model")).to(DEVICE)

    train_cfg = cfg.get("train", {})
    epochs = default_epochs if default_epochs is not None else train_cfg.get("epochs", 600)
    optim = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.get("lr", 2e-3 if model_name == "ppgnn" else 1e-2),
        weight_decay=train_cfg.get("weight_decay", 5e-4 if model_name == "ppgnn" else 5e-4),
    )
    clip_value = train_cfg.get("clip_value", 5.0 if model_name == "ppgnn" else None)

    def evaluate(loader: DataLoader) -> float:
        model.eval()
        err = n = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(DEVICE)
                out = model(batch)
                out = global_mean_pool(out, batch.batch)
                err += mae(out, batch.y) * batch.y.size(0)
                n += batch.y.size(0)
        return err / n

    best_val = float("inf")
    best_test = float("inf")
    for epoch in range(1, epochs + 1):
        # train
        model.train()
        for batch in loaders["train"]:
            batch = batch.to(DEVICE)
            optim.zero_grad()
            out = model(batch)
            out = global_mean_pool(out, batch.batch)
            loss = F.l1_loss(out, batch.y)
            loss.backward()

        val_mae = evaluate(loaders["val"])
        test_mae = evaluate(loaders["test"])
        if val_mae < best_val:
            best_val, best_test = val_mae, test_mae

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"E{epoch:03d}  val:{val_mae:.4f}  test:{test_mae:.4f}")

    print(f"★ Best val={best_val:.4f}  corresponding test={best_test:.4f}")


def main(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["PubMed"],  # "Cora", "CiteSeer", "PubMed"
        help="Dataset(s) to evaluate",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ppgnn", "gcn"],
        help="Models to train",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--config-dir", type=str, default="configs", help="Configuration directory")
    args = parser.parse_args(argv)

    set_seed(42)

    for dataset_name in args.dataset:
        print(f"\n=== Dataset: {dataset_name} ===")
        info = load_dataset(dataset_name)
        cfg = load_yaml_config(dataset_name, args.config_dir)

        for model_name in args.models:
            print(f"\n--- {model_name.upper()} ---")
            model_cfg = cfg.get(model_name, {})
            if info["task"] == "node":
                train_node(info["data"], model_name, info["num_classes"], model_cfg, args.epochs)
            else:
                train_graph(
                    info["loaders"],
                    model_name,
                    info["in_channels"],
                    info["num_classes"],
                    model_cfg,
                    args.epochs,
                )


if __name__ == "__main__":
    main()
