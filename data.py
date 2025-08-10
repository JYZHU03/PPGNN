from __future__ import annotations

"""Dataset loading utilities.

This module centralises all dataset handling so that training scripts can
simply query :func:`load_dataset` with a dataset name.  The returned
information clearly states whether the task is node/graph level and
classification/regression, together with the relevant data objects or
loaders.
"""

from typing import Dict

import torch
from torch_geometric.datasets import Planetoid, WebKB, ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, NormalizeFeatures, ToUndirected

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


PLANETOID = {"Cora", "CiteSeer", "PubMed"}
WEBKB = {"Cornell", "Texas", "Wisconsin"}


def load_dataset(name: str) -> Dict:
    """Load dataset *name* and return a description dictionary.

    The dictionary contains the following keys:
    ``level`` ("node" or "graph"), ``task`` ("classification" or
    "regression"), ``in_channels`` and ``out_channels``.  Depending on
    ``level`` either ``data`` (for node-level tasks) or ``loaders`` (for
    graph-level tasks) will be present.
    """

    if name in PLANETOID:
        dataset = Planetoid(
            root="data/Planetoid",
            name=name,
            transform=Compose([NormalizeFeatures(), ToUndirected()]),
        )
        data = dataset[0].to(DEVICE)
        return dict(
            level="node",
            task="classification",
            in_channels=dataset.num_features,
            out_channels=dataset.num_classes,
            data=data,
        )

    if name in WEBKB:
        dataset = WebKB(
            root="data/WebKB",
            name=name,
            transform=Compose([NormalizeFeatures(), ToUndirected()]),
        )
        data = dataset[0].to(DEVICE)
        return dict(
            level="node",
            task="classification",
            in_channels=dataset.num_features,
            out_channels=dataset.num_classes,
            data=data,
        )

    if name == "ZINC":
        train_ds = ZINC(root="data/ZINC", split="train")
        val_ds = ZINC(root="data/ZINC", split="val")
        test_ds = ZINC(root="data/ZINC", split="test")
        loaders = {
            "train": DataLoader(train_ds, batch_size=64, shuffle=True),
            "val": DataLoader(val_ds, batch_size=128),
            "test": DataLoader(test_ds, batch_size=128),
        }
        out_dim = train_ds[0].y.size(-1)
        return dict(
            level="graph",
            task="regression",
            in_channels=train_ds.num_features,
            out_channels=out_dim,
            loaders=loaders,
        )

    raise ValueError(f"Unknown dataset: {name}")