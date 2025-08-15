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

try:  # Optional MLflow dependency
    import mlflow  # type: ignore
except Exception:  # pragma: no cover - allow running without MLflow
    mlflow = None



from utils import set_seed
from models import PPGNN, GCN, GraphSAGE, GAT
from data import load_dataset
from tasks import TRAINERS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default hyper-parameters used when neither CLI arguments nor YAML specify a
# value. These mirror the previous ``argparse`` defaults to maintain backwards
# compatibility.
MODEL_DEFAULTS = {"hidden": 128, "layers": 5, "dropout": 0.2}
TRAIN_DEFAULTS = {"lr": 0.001, "weight_decay": 0.0005, "epochs": 300}


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def get_model(
    name: str, in_channels: int, out_channels: int, cfg: Dict[str, Any], level: str
):
    """Create a model by name with hyper-parameters from *cfg*."""
    cfg = cfg or {}
    params = dict(
        in_channels=in_channels,
        hidden=cfg.get("hidden", 64),
        num_classes=out_channels,
        level=level,
    )
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
            norm=cfg.get("norm", "BatchNorm1d"),
            lift_type=cfg.get("lift_type", "linear"),
            lift_layers=cfg.get("lift_layers", 2),
        )
    if name == "gcn":
        return GCN(
            **params,
            layers=cfg.get("layers", 2),
            dropout=cfg.get("dropout", 0.5),
            norm=cfg.get("norm", "BatchNorm1d"),
        )
    if name == "sage":
        return GraphSAGE(
            **params,
            layers=cfg.get("layers", 2),
            dropout=cfg.get("dropout", 0.5),
            norm=cfg.get("norm", "None"),
        )
    if name == "gat":
        return GAT(
            **params,
            heads=cfg.get("heads", 8),
            layers=cfg.get("layers", 2),
            dropout=cfg.get("dropout", 0.5),
            norm=cfg.get("norm", "None"),
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
        default=["Peptides-struct"], # e.g. "Cora", "CiteSeer", "PubMed", "CS", "Computers", "Photo", "ogbn-arxiv", "Cornell", "Texas", "Wisconsin", "ZINC", "MNIST", "tr20_teTexas", "tr20_te100", "PascalVOC-SP", "COCO-SP", "Peptides-func", "Peptides-struct", "PCQM-Contact",
        help="Dataset(s) to evaluate",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ppgnn", "gcn", "sage", "gat"],  # e.g. "ppgnn", "gcn", "sage", "gat"
        help="Models to train",
    )
    # Hyper-parameters are optional on the command line. If omitted, values from
    # the dataset configuration file are used and, if still unspecified, fall
    # back to ``MODEL_DEFAULTS``/``TRAIN_DEFAULTS`` defined above.
    parser.add_argument("--hidden", type=int, default=None, help="Hidden dimension")
    parser.add_argument("--layers", type=int, default=None, help="Number of layers")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
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
            model_params = model_cfg.get("model", {})
            for key, default in MODEL_DEFAULTS.items():
                arg_val = getattr(args, key)
                if arg_val is not None:
                    model_params[key] = arg_val
                else:
                    model_params.setdefault(key, default)
            model_cfg["model"] = model_params

            train_cfg = model_cfg.get("train", {})
            for key, default in TRAIN_DEFAULTS.items():
                arg_val = getattr(args, key)
                if arg_val is not None:
                    train_cfg[key] = arg_val
                else:
                    train_cfg.setdefault(key, default)
            model_cfg["train"] = train_cfg

            # Print resolved configuration for transparency
            print("Model parameters:")
            for k, v in model_params.items():
                print(f"  {k}: {v}")
            print("Training parameters:")
            for k, v in train_cfg.items():
                print(f"  {k}: {v}")

            model = get_model(
                model_name,
                info["in_channels"],
                info["out_channels"],
                model_params,
                info["level"],
            )
            trainer = TRAINERS[(info["level"], info["task"])]
            data_or_loader = info.get("data") or info.get("loaders")

            if mlflow is not None:
                mlflow.set_experiment(dataset_name)
                with mlflow.start_run(run_name=model_name):
                    mlflow.log_param("dataset", dataset_name)
                    mlflow.log_param("model", model_name)
                    for k, v in model_params.items():
                        mlflow.log_param(f"model_{k}", v)
                    for k, v in train_cfg.items():
                        mlflow.log_param(f"train_{k}", v)
                    trainer(data_or_loader, model, model_name, train_cfg, None)
            else:
                trainer(data_or_loader, model, model_name, train_cfg, None)

if __name__ == "__main__":
    main()
