from __future__ import annotations

"""Command line training utility."""

import argparse
from pathlib import Path
from typing import Dict, Any, Iterable

try:  # Optional YAML dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback if PyYAML is absent
    yaml = None
import json

import torch

from utils import set_seed
from models import PPGNN, GCN, GraphSAGE, GAT
from data import load_dataset
from tasks import TRAINERS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def get_model(
    name: str, in_channels: int, out_channels: int, cfg: Dict[str, Any] | None = None
):
    """Create a model by name with hyper-parameters from *cfg*."""
    cfg = cfg or {}
    params = dict(in_channels=in_channels, hidden=cfg.get("hidden", 64), num_classes=out_channels)
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
            alpha0=cfg.get("alpha0", 0.2),
            beta0=cfg.get("beta0", 0.1),
            dx0=cfg.get("dx0", 0.7),
            dy0=cfg.get("dy0", 0.8),
        )
    if name == "gcn":
        return GCN(**params, layers=cfg.get("layers", 2), dropout=cfg.get("dropout", 0.5))
    if name == "sage":
        return GraphSAGE(**params, layers=cfg.get("layers", 2), dropout=cfg.get("dropout", 0.5))
    if name == "gat":
        return GAT(
            **params,
            heads=cfg.get("heads", 8),
            layers=cfg.get("layers", 2),
            dropout=cfg.get("dropout", 0.5),
        )
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

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



def main(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["Cora"], # e.g. "Cora", "CiteSeer", "PubMed", "Cornell", "Texas", "Wisconsin", "ZINC", "MNIST", "ogbn-arxiv", "tr20_teTexas"
        help="Dataset(s) to evaluate",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ppgnn", "gcn", "sage", "gat"],  # e.g. "ppgnn", "gcn", "sage", "gat"
        help="Models to train",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--config-dir", type=str, default="configs", help="Configuration directory")
    args = parser.parse_args(argv)

    set_seed(0)

    for dataset_name in args.dataset:
        print(f"\n=== Dataset: {dataset_name} ===")
        info = load_dataset(dataset_name)
        cfg = load_yaml_config(dataset_name, args.config_dir)

        for model_name in args.models:
            print(f"\n--- {model_name.upper()} ---")
            model_cfg = cfg.get(model_name, {})
            model = get_model(
                model_name,
                info["in_channels"],
                info["out_channels"],
                model_cfg.get("model"),
            )
            trainer = TRAINERS[(info["level"], info["task"])]
            train_cfg = model_cfg.get("train", {})
            data_or_loader = info.get("data") or info.get("loaders")
            trainer(data_or_loader, model, model_name, train_cfg, args.epochs)


if __name__ == "__main__":
    main()
