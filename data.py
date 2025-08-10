from __future__ import annotations

from typing import Dict

import torch
from torch_geometric.datasets import Planetoid, WebKB, TUDataset, ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, NormalizeFeatures, ToUndirected

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(name: str) -> Dict:
