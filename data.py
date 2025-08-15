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
from torch_geometric.datasets import Planetoid, WebKB, ZINC, MNISTSuperpixels
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, NormalizeFeatures, ToUndirected
from torch_geometric.data import Batch
from power_grid_data import preformat_Powergrid

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

    if name in {"MNIST", "MNISTSuperpixels"}:
        train_full = MNISTSuperpixels(root="data/MNIST", train=True).shuffle()
        test_ds = MNISTSuperpixels(root="data/MNIST", train=False)
        train_ds = train_full[:55000]
        val_ds = train_full[55000:]
        loaders = {
            "train": DataLoader(train_ds, batch_size=64, shuffle=True),
            "val": DataLoader(val_ds, batch_size=128),
            "test": DataLoader(test_ds, batch_size=128),
        }
        return dict(
            level="graph",
            task="classification",
            in_channels=train_ds.num_features,
            out_channels=10,
            loaders=loaders,
        )

    if name == "ogbn-arxiv":
            from ogb.nodeproppred import PygNodePropPredDataset
            dataset = PygNodePropPredDataset(root="data/ogbn-arxiv", name="ogbn-arxiv")
            data = dataset[0].to(DEVICE)
            split_idx = dataset.get_idx_split()
            data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.train_mask[split_idx["train"]] = True
            data.val_mask[split_idx["valid"]] = True
            data.test_mask[split_idx["test"]] = True
            data.y = data.y.squeeze()
            return dict(
                level="node",
                task="classification",
                in_channels=dataset.num_features,
                out_channels=dataset.num_classes,
                data=data,
            )

    if name == "tr20_teTexas":
        dataset = preformat_Powergrid(
            "data/PowerGrid",
            train_dataset="dataset20",
            test_dataset="Texas",
        )
        data_list = [dataset[i] for i in range(len(dataset))]
        batch = Batch.from_data_list(data_list).to(DEVICE)
        train_idx, val_idx, test_idx = dataset.split_idxs
        train_mask = torch.zeros(batch.num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(batch.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(batch.num_nodes, dtype=torch.bool)
        ptr = 0
        for i, d in enumerate(data_list):
            n = d.num_nodes
            if i in train_idx:
                train_mask[ptr:ptr + n] = True
            elif i in val_idx:
                val_mask[ptr:ptr + n] = True
            else:
                test_mask[ptr:ptr + n] = True
            ptr += n
        batch.train_mask = train_mask
        batch.val_mask = val_mask
        batch.test_mask = test_mask
        out_dim = batch.y.size(-1) if batch.y.ndim > 1 else 1
        return dict(
            level="node",
            task="regression",
            in_channels=batch.num_features,
            out_channels=out_dim,
            data=batch,
        )

    if name == "tr20_te100":
        dataset = preformat_Powergrid(
            "data/PowerGrid",
            train_dataset="dataset20",
            test_dataset="dataset100",
        )
        data_list = [dataset[i] for i in range(len(dataset))]
        batch = Batch.from_data_list(data_list).to(DEVICE)
        train_idx, val_idx, test_idx = dataset.split_idxs
        train_mask = torch.zeros(batch.num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(batch.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(batch.num_nodes, dtype=torch.bool)
        ptr = 0
        for i, d in enumerate(data_list):
            n = d.num_nodes
            if i in train_idx:
                train_mask[ptr:ptr + n] = True
            elif i in val_idx:
                val_mask[ptr:ptr + n] = True
            else:
                test_mask[ptr:ptr + n] = True
            ptr += n
        batch.train_mask = train_mask
        batch.val_mask = val_mask
        batch.test_mask = test_mask
        out_dim = batch.y.size(-1) if batch.y.ndim > 1 else 1
        return dict(
            level="node",
            task="regression",
            in_channels=batch.num_features,
            out_channels=out_dim,
            data=batch,
        )

    raise ValueError(f"Unknown dataset: {name}")