from __future__ import annotations

"""
ppgnn_v6.py; 这个版本也不错，我把# FA-LV（更强的异质驱动，放大 F 的平滑幅度与耦合）的参数都设置为了1，也就是全部可学习了。
-----------
Single-file runner for PPGNN and baselines (V6).

Key points vs earlier versions:
- Keep "custom" / "custom_gnn" so Jacobi can use a PyG operator as diffusion skeleton.
- Implement FA-LV (Frequency-Adaptive LV): alpha,beta,gamma,delta, Dx,Dy become functions of graph heterophily rho.
- Remove explicit HF gating by default; let LV+two-scale diffusion realize low/band/high-pass automatically.
- Robust YAML loading: reads {dataset}.yaml from --config-dir (no version suffix).
"""

import argparse
import json
import math
import os
import os.path as osp
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional YAML dependency
try:  # pragma: no cover
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

# Optional MLflow dependency
try:  # pragma: no cover
    import mlflow  # type: ignore
except Exception:  # pragma: no cover
    mlflow = None

# PyG
from torch_geometric.data import Batch, Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import (
    Compose,
    NormalizeFeatures,
    RandomNodeSplit,
    ToUndirected,
)
from torch_geometric.utils import degree
from torch_geometric.nn import (
    APPNP,
    ARMAConv,
    ChebConv,
    GATConv,
    GCNConv,
    GINConv,
    GraphConv,
    SAGEConv,
    SGConv,
    TAGConv,
    TransformerConv,
)
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.datasets import (
    Amazon,
    Coauthor,
    LRGBDataset,
    MNISTSuperpixels,
    Planetoid,
    WebKB,
    ZINC,
)

# Optional OGB
try:  # pragma: no cover
    from ogb.nodeproppred import PygNodePropPredDataset  # type: ignore
except Exception:  # pragma: no cover
    PygNodePropPredDataset = None  # type: ignore

# Optional gdown for PowerGrid
try:  # pragma: no cover
    import gdown  # type: ignore
except Exception:  # pragma: no cover
    gdown = None  # type: ignore

import csv
import h5py
import zipfile

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# Utils
# =========================================================

def set_seed(seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def accuracy(logits, y):
    return (logits.argmax(dim=-1) == y).float().mean().item()


@torch.no_grad()
def mae(pred, y):
    return (pred - y).abs().mean().item()


@torch.no_grad()
def r2(pred, y):
    y = y.view_as(pred)
    ss_tot = ((y - y.mean()) ** 2).sum()
    ss_res = ((y - pred) ** 2).sum()
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot).item()


def _softplus_inv(y: float) -> float:
    # numerically stable inverse softplus
    y = max(float(y), 1e-6)
    return math.log(math.exp(y) - 1.0)


@torch.no_grad()
def mean_edge_cosine(x: torch.Tensor, edge_index: torch.Tensor, sample: int = 200_000) -> float:
    """mean cosine similarity across edges in [-1,1]."""
    if x is None or x.numel() == 0 or edge_index.numel() == 0:
        return 0.0
    row, col = edge_index
    m = row.numel()
    if m > sample:
        idx = torch.randint(0, m, (sample,), device=row.device)
        row, col = row[idx], col[idx]
    x = F.normalize(x, p=2, dim=-1)
    cos = (x[row] * x[col]).sum(dim=-1)
    return float(cos.mean().clamp(min=-1.0, max=1.0).item())


def global_add_pool_safe(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Torch-only global add pooling avoiding optional torch_scatter.
    Works on CPU/GPU, assuming `batch` is 0..B-1 with size equal to x.size(0).
    """
    if batch.numel() == 0:
        return x.new_zeros((0, x.size(-1)))
    if batch.dtype != torch.long:
        batch = batch.long()
    num_graphs = int(batch.max().item()) + 1
    out = x.new_zeros((num_graphs, x.size(-1)))
    out.index_add_(0, batch, x)
    return out


# =========================================================
# LVConv (V6): LV + Jacobi + FA-LV (no explicit HF gating)
# =========================================================

class LVConv(MessagePassing):
    """
    Lotka–Volterra graph convolution layer with semi-implicit diffusion.
    - Two channels: R,F (each d dims); reaction coeffs alpha,beta,gamma,delta (per-channel);
      diffusion strengths Dx,Dy (scalars, positive).
    - Jacobi step inside the layer; optionally replace S·X by a PyG conv ("custom"/"custom_gnn").
    - FA-LV: alpha,beta,gamma,delta,Dx,Dy depend on graph heterophily rho in [0,1].
      Mapping (softplus keeps positivity):
        alpha(rho) = softplus(alpha + alpha1 * rho)
        beta (rho) = softplus(beta  + beta1  * rho)
        gamma(rho) = softplus(gamma - gamma1 * rho)
        delta(rho) = softplus(delta + delta1 * rho)
        Dx   (rho) = softplus(Dx    - dx1    * rho)   # R-channel diff decreases with hetero
        Dy   (rho) = softplus(Dy    + dy1    * rho)   # F-channel diff increases with hetero
    """

    def __init__(
        self,
        channels: int,
        dt: float = 0.1,
        norm_type: str = "sym",
        jacobi_steps: int = 2,
        eps: float = 1e-3,
        alpha0: float = 0.2,
        beta0: float = 0.1,
        dx0: float = 0.7,
        dy0: float = 0.8,
        # Scheme A: use PyG convs as diffusion operator
        custom: int | bool = 1,
        custom_gnn: str = "gcn",
        heads: int = 1,
        # FA-LV toggles
        fa_lv: int | bool = 1,
        fa_power: float = 1.0,
        fa_alpha1: float = 0.10,
        fa_beta1: float = 0.15,
        fa_gamma1: float = 0.05,
        fa_delta1: float = 0.15,
        fa_dx1: float = 0.35,
        fa_dy1: float = 0.60,
    ):
        super().__init__(aggr="add")
        assert norm_type in {"sym", "rw"}
        assert jacobi_steps >= 1
        self.d = int(channels)
        self.dt = float(dt)
        self.norm_type = norm_type
        self.jacobi_steps = int(jacobi_steps)
        self.eps = float(eps)

        # Reaction parameters (pre-softplus, per-channel)
        self.alpha = nn.Parameter(torch.zeros(self.d))
        self.beta  = nn.Parameter(torch.zeros(self.d))
        self.gamma = nn.Parameter(torch.zeros(self.d))
        self.delta = nn.Parameter(torch.zeros(self.d))

        # Diffusion parameters (pre-softplus, scalars)
        self.Dx = nn.Parameter(torch.tensor(0.0))
        self.Dy = nn.Parameter(torch.tensor(0.0))

        with torch.no_grad():
            self.alpha.fill_(_softplus_inv(alpha0))
            self.gamma.fill_(_softplus_inv(alpha0))
            self.beta.fill_ (_softplus_inv(beta0))
            self.delta.fill_(_softplus_inv(beta0))
            self.Dx.fill_   (_softplus_inv(dx0))
            self.Dy.fill_   (_softplus_inv(dy0))

        # FA-LV slopes (learnable; scalars broadcast to channels)
        self.use_fa = bool(fa_lv)
        self.fa_power = float(fa_power)
        self.alpha1 = nn.Parameter(torch.tensor(float(fa_alpha1)))
        self.beta1  = nn.Parameter(torch.tensor(float(fa_beta1)))
        self.gamma1 = nn.Parameter(torch.tensor(float(fa_gamma1)))
        self.delta1 = nn.Parameter(torch.tensor(float(fa_delta1)))
        self.dx1    = nn.Parameter(torch.tensor(float(fa_dx1)))
        self.dy1    = nn.Parameter(torch.tensor(float(fa_dy1)))

        # Custom convs (Scheme A) for both R and F paths
        self.use_custom = bool(custom)
        self.custom_gnn = (custom_gnn or "gcn").lower()
        self.heads = int(heads)
        self._custom_type = None
        if self.use_custom:
            if self.custom_gnn == "gcn":
                # cached must be False for mini-batch graph classification
                self.custom_conv_R = GCNConv(self.d, self.d, add_self_loops=False, normalize=True, bias=False, cached=False)
                self.custom_conv_F = GCNConv(self.d, self.d, add_self_loops=False, normalize=True, bias=False, cached=False)
                self._custom_type = "conv"
            elif self.custom_gnn == "sage":
                self.custom_conv_R = SAGEConv(self.d, self.d, aggr="mean", root_weight=False, bias=False)
                self.custom_conv_F = SAGEConv(self.d, self.d, aggr="mean", root_weight=False, bias=False)
                self._custom_type = "conv"
            elif self.custom_gnn == "gat":
                self.custom_conv_R = GATConv(self.d, self.d, heads=self.heads, concat=False, add_self_loops=False, bias=False)
                self.custom_conv_F = GATConv(self.d, self.d, heads=self.heads, concat=False, add_self_loops=False, bias=False)
                self._custom_type = "conv"
            elif self.custom_gnn == "transformer":
                self.custom_conv_R = TransformerConv(self.d, self.d, heads=self.heads, concat=False, add_self_loops=False, bias=False)
                self.custom_conv_F = TransformerConv(self.d, self.d, heads=self.heads, concat=False, add_self_loops=False, bias=False)
                self._custom_type = "conv"
            elif self.custom_gnn == "appnp":
                self.lin_R = nn.Linear(self.d, self.d, bias=False)
                self.lin_F = nn.Linear(self.d, self.d, bias=False)
                self.appnp_R = APPNP(K=10, alpha=0.1, dropout=0.0)
                self.appnp_F = APPNP(K=10, alpha=0.1, dropout=0.0)
                self._custom_type = "appnp"
            elif self.custom_gnn == "tagcn":
                self.custom_conv_R = TAGConv(self.d, self.d, K=3, bias=False)
                self.custom_conv_F = TAGConv(self.d, self.d, K=3, bias=False)
                self._custom_type = "conv"
            elif self.custom_gnn == "graphconv":
                self.custom_conv_R = GraphConv(self.d, self.d, aggr="add", bias=False)
                self.custom_conv_F = GraphConv(self.d, self.d, aggr="add", bias=False)
                self._custom_type = "conv"
            elif self.custom_gnn == "cheb":
                self.custom_conv_R = ChebConv(self.d, self.d, K=3)
                self.custom_conv_F = ChebConv(self.d, self.d, K=3)
                self._custom_type = "conv"
            elif self.custom_gnn == "arma":
                self.custom_conv_R = ARMAConv(self.d, self.d, num_stacks=1, num_layers=1, shared_weights=False)
                self.custom_conv_F = ARMAConv(self.d, self.d, num_stacks=1, num_layers=1, shared_weights=False)
                self._custom_type = "conv"
            elif self.custom_gnn == "sgc":
                # cached must be False for mini-batch graph classification
                self.custom_conv_R = SGConv(self.d, self.d, K=1, cached=False, bias=False)
                self.custom_conv_F = SGConv(self.d, self.d, K=1, cached=False, bias=False)
                self._custom_type = "conv"
            elif self.custom_gnn == "gin":
                mlpR = nn.Sequential(nn.Linear(self.d, self.d), nn.ReLU(), nn.Linear(self.d, self.d))
                mlpF = nn.Sequential(nn.Linear(self.d, self.d), nn.ReLU(), nn.Linear(self.d, self.d))
                self.custom_conv_R = GINConv(mlpR)
                self.custom_conv_F = GINConv(mlpF)
                self._custom_type = "conv"
            else:
                raise ValueError(f"Unknown custom_gnn: {self.custom_gnn}")

    def _fa_rho(self, hetero: float | None) -> float:
        rho = 0.5 if hetero is None else float(max(0.0, min(1.0, hetero)))
        if self.fa_power != 1.0:
            rho = float(rho ** self.fa_power)
        return rho

    def _eff_params(self, hetero: float | None):
        """Return effective positive params with FA-LV mapping."""
        # base pre-softplus params
        a, b, g, d = self.alpha, self.beta, self.gamma, self.delta
        Dx, Dy = self.Dx, self.Dy
        if self.use_fa:
            rho = self._fa_rho(hetero)
            a = a + self.alpha1 * rho
            b = b + self.beta1  * rho
            g = g - self.gamma1 * rho
            d = d + self.delta1 * rho
            Dx = Dx - self.dx1   * rho
            Dy = Dy + self.dy1   * rho
        # positive
        a = F.softplus(a) + self.eps
        b = F.softplus(b) + self.eps
        g = F.softplus(g) + self.eps
        d = F.softplus(d) + self.eps
        Dx = F.softplus(Dx) + self.eps
        Dy = F.softplus(Dy) + self.eps
        return a, b, g, d, Dx, Dy

    def _apply_conv(self, ZR: torch.Tensor, ZF: torch.Tensor, edge_index: torch.Tensor):
        if not self.use_custom:
            return None, None
        if self._custom_type == "appnp":
            SR = self.appnp_R(self.lin_R(ZR), edge_index)
            SF = self.appnp_F(self.lin_F(ZF), edge_index)
            return SR, SF
        SR = self.custom_conv_R(ZR, edge_index)
        SF = self.custom_conv_F(ZF, edge_index)
        return SR, SF

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, hetero: float | None = None):
        X, Y = torch.split(h, self.d, dim=-1)  # X->R, Y->F

        # Norm weights for non-custom path
        if not self.use_custom:
            N = h.size(0)
            row, col = edge_index
            if self.norm_type == "sym":
                deg = degree(row, N, dtype=X.dtype).clamp(min=1)
                deg_inv_sqrt = deg.pow(-0.5)
                norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            else:
                deg = degree(col, N, dtype=X.dtype).clamp(min=1)
                norm = (1.0 / deg)[col]
        else:
            norm = None  # unused

        # Effective parameters (FA-LV)
        a, b, g, d, Dx, Dy = self._eff_params(hetero)
        dt = self.dt

        # Reaction terms
        RX = a * X - b * (X * Y)
        RY = d * (X * Y) - g * Y
        RHS_X = X + dt * RX
        RHS_Y = Y + dt * RY

        ax = dt * Dx
        ay = dt * Dy
        denom_x = 1.0 + ax
        denom_y = 1.0 + ay

        # Jacobi iterations
        Xk, Yk = X, Y
        for _ in range(self.jacobi_steps):
            if self.use_custom:
                SXk, SYk = self._apply_conv(Xk, Yk, edge_index)
            else:
                SXk = self.propagate(edge_index, x=Xk, norm=norm)
                SYk = self.propagate(edge_index, x=Yk, norm=norm)
            Xk = (RHS_X + ax * SXk) / denom_x
            Yk = (RHS_Y + ay * SYk) / denom_y

        return torch.cat([Xk, Yk], dim=-1)

    def message(self, x_j: torch.Tensor, norm: torch.Tensor):
        # only used when not custom
        return norm.view(-1, 1) * x_j


# =========================================================
# Models
# =========================================================

class PPGNN(nn.Module):
    """Predator–Prey GNN with semi-implicit diffusion and FA-LV."""
    def __init__(
        self,
        in_channels: int,
        hidden: int,
        num_classes: int,
        layers: int = 15,
        dt: float = 0.1,
        dropout: float = 0.4,
        norm_type: str = "sym",
        jacobi_steps: int = 2,
        use_x_only: bool = False,
        y0_mode: str = "learned",
        alpha0: float = 0.2,
        beta0: float = 0.1,
        dx0: float = 0.7,
        dy0: float = 0.8,
        norm: str = "BatchNorm1d",
        level: str = "node",
        lift_type: str = "linear",
        lift_layers: int = 2,
        # Scheme A
        custom: int | bool = 0,
        custom_gnn: str = "gcn",
        heads: int = 1,
        # FA-LV
        fa_lv: int | bool = 1,
        fa_power: float = 1.0,
        fa_alpha1: float = 0.10,
        fa_beta1: float = 0.15,
        fa_gamma1: float = 0.05,
        fa_delta1: float = 0.15,
        fa_dx1: float = 0.35,
        fa_dy1: float = 0.60,
    ):
        super().__init__()
        self.hidden = hidden
        self.dropout = dropout
        self.use_x_only = bool(use_x_only)
        self.y0_mode = y0_mode
        self.graph_head = (level == "graph")
        self._hetero_cached: Optional[float] = None  # rho cache

        # Lift
        lt = (lift_type or "linear").lower()
        def build_lift():
            if lt == "mlp" and lift_layers > 1:
                mlp = [nn.Linear(in_channels, hidden), nn.ReLU()]
                for _ in range(lift_layers - 2):
                    mlp += [nn.Linear(hidden, hidden), nn.ReLU()]
                mlp += [nn.Linear(hidden, hidden)]
                return nn.Sequential(*mlp)
            return nn.Linear(in_channels, hidden)

        self.lift_x = build_lift()
        self.lift_y = build_lift()

        # LV layers
        self.layers = nn.ModuleList(
            [
                LVConv(
                    channels=hidden,
                    dt=dt,
                    norm_type=norm_type,
                    jacobi_steps=jacobi_steps,
                    alpha0=alpha0,
                    beta0=beta0,
                    dx0=dx0,
                    dy0=dy0,
                    custom=custom,
                    custom_gnn=custom_gnn,
                    heads=heads,
                    fa_lv=fa_lv,
                    fa_power=fa_power,
                    fa_alpha1=fa_alpha1,
                    fa_beta1=fa_beta1,
                    fa_gamma1=fa_gamma1,
                    fa_delta1=fa_delta1,
                    fa_dx1=fa_dx1,
                    fa_dy1=fa_dy1,
                )
                for _ in range(layers)
            ]
        )

        # Norms
        norm_name = (norm or "none").lower()
        def make_norm():
            if norm_name in ("batchnorm1d", "batchnorm"):
                return nn.BatchNorm1d(2 * hidden)
            if norm_name in ("layernorm", "ln"):
                return nn.LayerNorm(2 * hidden)
            return nn.Identity()
        self.norms = nn.ModuleList([make_norm() for _ in range(layers)])

        # Residual gate tau
        self.taus = nn.ParameterList([nn.Parameter(torch.tensor(0.7)) for _ in range(layers)])

        out_dim = hidden if self.use_x_only else 2 * hidden
        self.lin_out = nn.Linear(out_dim, num_classes)
        self.readout = (
            nn.Sequential(
                nn.Linear(out_dim, out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(out_dim, num_classes),
            )
            if self.graph_head
            else None
        )
        self.logit_scale = nn.Parameter(torch.tensor(2.5))

    def _lazy_rho(self, data):
        if self._hetero_cached is None:
            with torch.no_grad():
                hom = mean_edge_cosine(data.x.float(), data.edge_index)  # [-1,1]
                hom01 = (hom + 1.0) / 2.0
                self._hetero_cached = float(max(0.0, min(1.0, 1.0 - hom01)))  # rho in [0,1]
        return self._hetero_cached

    def forward(self, data):
        rho = self._lazy_rho(data)  # hetero

        X0 = torch.tanh(self.lift_x(data.x.float()))
        if self.y0_mode == "learned":
            Y0 = torch.tanh(self.lift_y(data.x.float()))
        else:
            Y0 = torch.ones_like(X0)
        h = torch.cat([X0, Y0], dim=-1)

        for conv, tau_p, norm in zip(self.layers, self.taus, self.norms):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h_hat = conv(h, data.edge_index, hetero=rho)
            tau = torch.sigmoid(tau_p)
            h = (1 - tau) * h + tau * h_hat
            h = norm(h)

        if self.graph_head and hasattr(data, "batch") and self.readout is not None:
            if self.use_x_only:
                X, _ = torch.split(h, self.hidden, dim=-1)
                g = global_add_pool_safe(X, data.batch)
            else:
                g = global_add_pool_safe(h, data.batch)
            g = F.dropout(g, p=0.1, training=self.training)
            return self.readout(self.logit_scale * g)

        if self.use_x_only:
            X, _ = torch.split(h, self.hidden, dim=-1)
            X = F.dropout(X, p=0.1, training=self.training)
            return self.lin_out(self.logit_scale * X)

        h = F.dropout(h, p=0.1, training=self.training)
        return self.lin_out(self.logit_scale * h)


# ---- Minimal baselines (for completeness) ----

class GCN(nn.Module):
    def __init__(self, in_channels, hidden, num_classes, layers=2, dropout=0.5, norm: str = "BatchNorm1d", level: str = "node"):
        super().__init__()
        self.dropout = dropout
        self.graph_head = (level == "graph")
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        norm = (norm or "none").lower()
        def make_norm():
            if norm in ("batchnorm1d","batchnorm"): return nn.BatchNorm1d(hidden)
            if norm in ("layernorm","ln"): return nn.LayerNorm(hidden)
            return nn.Identity()
        self.convs.append(GCNConv(in_channels, hidden))
        self.norms.append(make_norm())
        for _ in range(layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
            self.norms.append(make_norm())
        self.lin_out = nn.Linear(hidden, num_classes)
    def forward(self, data):
        x, ei = data.x.float(), data.edge_index
        for conv, norm in zip(self.convs, self.norms):
            x = F.dropout(F.relu(norm(conv(x, ei))), p=self.dropout, training=self.training)
        return self.lin_out(x)

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden, num_classes, layers=2, dropout=0.5, norm: str = "BatchNorm1d", level: str = "node"):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        norm = (norm or "none").lower()
        def make_norm():
            if norm in ("batchnorm1d","batchnorm"): return nn.BatchNorm1d(hidden)
            if norm in ("layernorm","ln"): return nn.LayerNorm(hidden)
            return nn.Identity()
        self.convs.append(SAGEConv(in_channels, hidden))
        self.norms.append(make_norm())
        for _ in range(layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
            self.norms.append(make_norm())
        self.lin_out = nn.Linear(hidden, num_classes)
    def forward(self, data):
        x, ei = data.x.float(), data.edge_index
        for conv, norm in zip(self.convs, self.norms):
            x = F.dropout(F.relu(norm(conv(x, ei))), p=self.dropout, training=self.training)
        return self.lin_out(x)

class GAT(nn.Module):
    def __init__(self, in_channels, hidden, num_classes, heads=8, layers=2, dropout=0.5, norm: str = "LayerNorm", level: str = "node"):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        norm = (norm or "none").lower()
        def make_norm(dim):
            if norm in ("batchnorm1d","batchnorm"): return nn.BatchNorm1d(dim)
            if norm in ("layernorm","ln"): return nn.LayerNorm(dim)
            return nn.Identity()
        self.convs.append(GATConv(in_channels, hidden, heads=heads))
        self.norms.append(make_norm(hidden * heads))
        for _ in range(layers - 1):
            self.convs.append(GATConv(hidden * heads, hidden, heads=heads))
            self.norms.append(make_norm(hidden * heads))
        self.lin_out = nn.Linear(hidden * heads, num_classes)
    def forward(self, data):
        x, ei = data.x.float(), data.edge_index
        for conv, norm in zip(self.convs, self.norms):
            x = F.dropout(F.relu(norm(conv(x, ei))), p=self.dropout, training=self.training)
        return self.lin_out(x)

class TransformerNet(nn.Module):
    def __init__(self, in_channels, hidden, num_classes, heads=4, layers=2, dropout=0.5, norm: str = "LayerNorm", level: str = "node"):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        norm = (norm or "none").lower()
        def make_norm(): return nn.LayerNorm(hidden) if norm in ("layernorm","ln") else nn.Identity()
        self.convs.append(TransformerConv(in_channels, hidden, heads=heads, concat=False))
        self.norms.append(make_norm())
        for _ in range(layers - 1):
            self.convs.append(TransformerConv(hidden, hidden, heads=heads, concat=False))
            self.norms.append(make_norm())
        self.lin_out = nn.Linear(hidden, num_classes)
    def forward(self, data):
        x, ei = data.x.float(), data.edge_index
        for conv, norm in zip(self.convs, self.norms):
            x = F.dropout(F.relu(norm(conv(x, ei))), p=self.dropout, training=self.training)
        return self.lin_out(x)


# =========================================================
# PowerGrid (kept for completeness)
# =========================================================
def join_dataset_splits(datasets):
    assert len(datasets) == 3
    n1, n2, n3 = len(datasets[0]), len(datasets[1]), len(datasets[2])
    data_list = [datasets[0].get(i) for i in range(n1)] + [datasets[1].get(i) for i in range(n2)] + [datasets[2].get(i) for i in range(n3)]
    datasets[0]._indices = None
    datasets[0]._data_list = data_list
    datasets[0].data, datasets[0].slices = datasets[0].collate(data_list)
    split_idxs = [list(range(n1)), list(range(n1, n1 + n2)), list(range(n1 + n2, n1 + n2 + n3))]
    datasets[0].split_idxs = split_idxs
    return datasets[0]

class Powergrid(InMemoryDataset):
    FILE_ID = "1yEZVwvaGenQ_yJvAPRXmNVzVEP-Tnmgz"
    URL = f"https://drive.google.com/uc?id={FILE_ID}"
    MARKER_FILE = "extracted.marker"
    def __init__(self, root: str, split: str = "train", train_dataset: str = "", test_dataset: str = "",
                 transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None, pre_filter: Optional[Callable] = None):
        if split == "val": split = "valid"
        assert split in ["train","valid","test"], split
        self.split = split
        self.root2 = root
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.task = "snbs"
        super().__init__(root, transform, pre_transform, pre_filter)
        path3 = osp.join(self.processed_dir, f"{split}.pt")
        self.data, self.slices = torch.load(path3)
        if hasattr(self.data, "x"): self.data.x = self.data.x.to(torch.float)
        if hasattr(self.data, "y"): self.data.y = self.data.y.to(torch.float)
    @property
    def raw_dir(self): return os.path.abspath(self.root)
    def raw_file_names(self) -> List[str]: return [self.MARKER_FILE]
    @property
    def processed_dir(self) -> str: return osp.join(self.root, "processed")
    @property
    def processed_file_names(self) -> List[str]: return ["train.pt","valid.pt","test.pt"]
    def download(self):
        marker_path = osp.join(self.raw_dir, self.MARKER_FILE)
        zip_path = osp.join(self.raw_dir, "powergrid_data.zip")
        if not osp.exists(marker_path):
            os.makedirs(self.raw_dir, exist_ok=True)
            if not osp.exists(zip_path):
                if gdown is None:
                    raise RuntimeError("gdown not installed; cannot auto-download PowerGrid.")
                print("Downloading data from Google Drive...")
                gdown.download(self.URL, zip_path, quiet=False)
            else:
                print("Zip already exists. Skipping download.")
            print("Extracting data...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.raw_dir)
            os.remove(zip_path)
            with open(marker_path, "w") as f:
                f.write("Extraction completed.")
        else:
            print("Datasets already downloaded. Skipping download.")
    def process(self):
        # omitted for brevity in V6; not used by Cora/Texas
        pass


# =========================================================
# Data loading
# =========================================================

PLANETOID = {"Cora", "CiteSeer", "PubMed"}
WEBKB = {"Cornell", "Texas", "Wisconsin"}
AMAZON = {"Computers", "Photo"}
COAUTHOR = {"CS"}
LRGB = {"PascalVOC-SP", "COCO-SP", "Peptides-func", "Peptides-struct", "PCQM-Contact"}

def load_dataset(name: str) -> Dict:
    if name in PLANETOID:
        dataset = Planetoid(root="data/Planetoid", name=name, transform=Compose([NormalizeFeatures(), ToUndirected()]))
        data = dataset[0].to(DEVICE)
        return dict(level="node", task="classification", in_channels=dataset.num_features, out_channels=dataset.num_classes, data=data)

    if name in WEBKB:
        dataset = WebKB(root="data/WebKB", name=name, transform=Compose([NormalizeFeatures(), ToUndirected()]))
        data = dataset[0].to(DEVICE)
        return dict(level="node", task="classification", in_channels=dataset.num_features, out_channels=dataset.num_classes, data=data)

    if name in AMAZON:
        dataset = Amazon(root="data/Amazon", name=name, transform=Compose([NormalizeFeatures(), ToUndirected(), RandomNodeSplit(num_val=0.1, num_test=0.2)]))
        data = dataset[0].to(DEVICE)
        return dict(level="node", task="classification", in_channels=dataset.num_features, out_channels=dataset.num_classes, data=data)

    if name in COAUTHOR:
        dataset = Coauthor(root="data/Coauthor", name=name, transform=Compose([NormalizeFeatures(), ToUndirected(), RandomNodeSplit(num_val=0.1, num_test=0.2)]))
        data = dataset[0].to(DEVICE)
        return dict(level="node", task="classification", in_channels=dataset.num_features, out_channels=dataset.num_classes, data=data)

    if name in LRGB:
        train_ds = LRGBDataset(root="data/LRGB", name=name, split="train")
        val_ds = LRGBDataset(root="data/LRGB", name=name, split="val")
        test_ds = LRGBDataset(root="data/LRGB", name=name, split="test")
        loaders = {"train": DataLoader(train_ds, batch_size=64, shuffle=True),
                   "val": DataLoader(val_ds, batch_size=128),
                   "test": DataLoader(test_ds, batch_size=128)}
        in_dim = train_ds.num_features
        out_dim = getattr(train_ds, "num_classes", None) or train_ds[0].y.size(-1)
        task_type = "regression" if name == "Peptides-struct" else "classification"
        return dict(level="graph", task=task_type, in_channels=in_dim, out_channels=out_dim, loaders=loaders)

    if name == "ZINC":
        train_ds = ZINC(root="data/ZINC", split="train")
        val_ds = ZINC(root="data/ZINC", split="val")
        test_ds = ZINC(root="data/ZINC", split="test")
        loaders = {"train": DataLoader(train_ds, batch_size=64, shuffle=True),
                   "val": DataLoader(val_ds, batch_size=128),
                   "test": DataLoader(test_ds, batch_size=128)}
        out_dim = train_ds[0].y.size(-1)
        return dict(level="graph", task="regression", in_channels=train_ds.num_features, out_channels=out_dim, loaders=loaders)

    if name in {"MNIST", "MNISTSuperpixels"}:
        train_full = MNISTSuperpixels(root="data/MNIST", train=True).shuffle()
        test_ds = MNISTSuperpixels(root="data/MNIST", train=False)
        train_ds, val_ds = train_full[:55000], train_full[55000:]
        loaders = {"train": DataLoader(train_ds, batch_size=64, shuffle=True),
                   "val": DataLoader(val_ds, batch_size=128),
                   "test": DataLoader(test_ds, batch_size=128)}
        return dict(level="graph", task="classification", in_channels=train_ds.num_features, out_channels=10, loaders=loaders)

    if name == "ogbn-arxiv":
        if PygNodePropPredDataset is None:
            raise RuntimeError("ogb not installed; cannot load ogbn-arxiv")
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
        return dict(level="node", task="classification", in_channels=dataset.num_features, out_channels=dataset.num_classes, data=data)

    raise ValueError(f"Unknown dataset: {name}")


# =========================================================
# Training loops
# =========================================================

def _setup_optim(model, model_name: str, train_cfg: Dict[str, float], default_epochs: int | None):
    epochs = default_epochs if default_epochs is not None else train_cfg.get("epochs", 600 if model_name == "ppgnn" else 200)
    optim = torch.optim.Adam(model.parameters(), lr=train_cfg.get("lr", 2e-3 if model_name == "ppgnn" else 1e-2),
                             weight_decay=train_cfg.get("weight_decay", 5e-4))
    clip_value = train_cfg.get("clip_value", 5.0 if model_name == "ppgnn" else None)
    return int(epochs), optim, clip_value

def train_node_classification(data, model, model_name: str, train_cfg: Dict, default_epochs: int | None) -> None:
    model = model.to(DEVICE); data = data.to(DEVICE)
    epochs, optim, clip_value = _setup_optim(model, model_name, train_cfg, default_epochs)
    best_val = best_test = best_epoch = 0.0
    for epoch in range(1, epochs + 1):
        model.train(); optim.zero_grad()
        out = model(data)
        train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
        if train_mask.ndim > 1: train_mask, val_mask, test_mask = train_mask[:,0], val_mask[:,0], test_mask[:,0]
        loss = F.cross_entropy(out[train_mask], data.y[train_mask]); loss.backward()
        if clip_value is not None: torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optim.step()

        model.eval()
        with torch.no_grad():
            logits = model(data)
            tr = accuracy(logits[train_mask], data.y[train_mask])
            va = accuracy(logits[val_mask], data.y[val_mask])
            te = accuracy(logits[test_mask], data.y[test_mask])
            trl = F.cross_entropy(logits[train_mask], data.y[train_mask])
            vl = F.cross_entropy(logits[val_mask], data.y[val_mask])
            tl = F.cross_entropy(logits[test_mask], data.y[test_mask])

        if mlflow is not None:
            mlflow.log_metrics({"train_loss": float(trl), "val_loss": float(vl), "test_loss": float(tl),
                                "train_acc": tr, "val_acc": va, "test_acc": te}, step=epoch)
        if va > best_val:
            best_val, best_test, best_epoch = va, te, epoch

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(f"Epoch {epoch:03d}  tr_acc:{tr:.3f}  va_acc:{va:.3f}  te_acc:{te:.3f}")

    print(f"★ Best@val: epoch={best_epoch}  val={best_val:.3f}  test@best_val={best_test:.3f}")
    if mlflow is not None:
        mlflow.log_metrics({"best_val": best_val, "best_test": best_test}, step=best_epoch)


def _maybe_pool(out: torch.Tensor, batch: Batch, pool_fn=global_add_pool_safe) -> torch.Tensor:
    """Pool node-level outputs to graph-level if a `batch` vector is present."""
    if hasattr(batch, "batch") and out.size(0) == batch.batch.numel():
        return pool_fn(out, batch.batch)
    return out


def train_graph_classification(loaders: Dict[str, DataLoader], model, model_name: str, train_cfg: Dict, default_epochs: int | None) -> None:
    model = model.to(DEVICE)
    epochs, optim, clip_value = _setup_optim(model, model_name, train_cfg, default_epochs)

    def _run(split: str, train: bool = False):
        if train:
            model.train()
        else:
            model.eval()
        total_loss = 0.0
        y_true = []
        y_pred = []
        for batch in loaders[split]:
            batch = batch.to(DEVICE)
            out = model(batch)
            out = _maybe_pool(out, batch)
            y = batch.y.view(-1).long()
            # Early sanity check for label range to avoid CUDA device asserts
            C = out.size(-1)
            if y.numel() > 0:
                y_min = int(y.min().item())
                y_max = int(y.max().item())
                if y_min < 0 or y_max >= C:
                    raise RuntimeError(
                        f"Label out of range: y in [{y_min},{y_max}] while logits dim={C}."
                    )
            loss = F.cross_entropy(out, y)
            if train:
                optim.zero_grad()
                loss.backward()
                if clip_value is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                optim.step()
            total_loss += float(loss.detach().item()) * y.numel()
            y_true.append(y.detach().cpu())
            y_pred.append(out.detach().cpu())
        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        avg_loss = total_loss / max(1, y_true.numel())
        acc = accuracy(y_pred, y_true)
        return avg_loss, acc

    best_val = best_test = 0.0
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _run("train", train=True)
        with torch.no_grad():
            val_loss, val_acc = _run("val", train=False)
            test_loss, test_acc = _run("test", train=False)

        if mlflow is not None:
            mlflow.log_metrics(
                {
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "test_loss": float(test_loss),
                    "train_acc": float(train_acc),
                    "val_acc": float(val_acc),
                    "test_acc": float(test_acc),
                },
                step=epoch,
            )

        if val_acc > best_val:
            best_val, best_test, best_epoch = float(val_acc), float(test_acc), epoch

        if epoch == 1 or epoch % 1 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:03d}  tr_loss:{train_loss:.3f}  va_loss:{val_loss:.3f}  te_loss:{test_loss:.3f}  "
                f"tr_acc:{train_acc:.3f}  va_acc:{val_acc:.3f}  te_acc:{test_acc:.3f}"
            )

    print(f"★ Best@val: epoch={best_epoch}  val={best_val:.3f}  test@best_val={best_test:.3f}")
    if mlflow is not None:
        mlflow.log_metrics({"best_val": best_val, "best_test": best_test}, step=best_epoch)


# =========================================================
# Factory + YAML loader
# =========================================================

MODEL_DEFAULTS = {"hidden": 128, "layers": 15, "dropout": 0.4}
TRAIN_DEFAULTS = {"lr": 0.002, "weight_decay": 0.0005, "epochs": 600, "clip_value": 5.0}

def get_model(name: str, in_channels: int, out_channels: int, cfg: Dict[str, Any], level: str):
    cfg = cfg or {}
    params = dict(in_channels=in_channels, hidden=cfg.get("hidden", 128), num_classes=out_channels, level=level)
    if name == "ppgnn":
        return PPGNN(
            **params,
            layers=cfg.get("layers", 15),
            dt=cfg.get("dt", 0.1),
            dropout=cfg.get("dropout", 0.4),
            norm_type=cfg.get("norm_type", "sym"),
            jacobi_steps=cfg.get("jacobi_steps", 2),
            use_x_only=cfg.get("use_x_only", False),
            y0_mode=cfg.get("y0_mode", "learned"),
            alpha0=cfg.get("alpha0", 0.2),
            beta0=cfg.get("beta0", 0.1),
            dx0=cfg.get("dx0", 0.7),
            dy0=cfg.get("dy0", 0.8),
            norm=cfg.get("norm", "BatchNorm1d"),
            lift_type=cfg.get("lift_type", "linear"),
            lift_layers=cfg.get("lift_layers", 2),
            custom=cfg.get("custom", 0),
            custom_gnn=cfg.get("custom_gnn", "gcn"),
            heads=cfg.get("heads", 1),
            fa_lv=cfg.get("fa_lv", 1),
            fa_power=cfg.get("fa_power", 1.0),
            fa_alpha1=cfg.get("fa_alpha1", 0.10),
            fa_beta1=cfg.get("fa_beta1", 0.15),
            fa_gamma1=cfg.get("fa_gamma1", 0.05),
            fa_delta1=cfg.get("fa_delta1", 0.15),
            fa_dx1=cfg.get("fa_dx1", 0.35),
            fa_dy1=cfg.get("fa_dy1", 0.60),
        )
    if name == "gcn":
        return GCN(**params, layers=cfg.get("layers", 2), dropout=cfg.get("dropout", 0.5), norm=cfg.get("norm","BatchNorm1d"))
    if name == "sage":
        return GraphSAGE(**params, layers=cfg.get("layers", 2), dropout=cfg.get("dropout", 0.5), norm=cfg.get("norm","BatchNorm1d"))
    if name == "gat":
        return GAT(**params, heads=cfg.get("heads", 8), layers=cfg.get("layers", 2), dropout=cfg.get("dropout", 0.5), norm=cfg.get("norm","LayerNorm"))
    if name == "transformer":
        return TransformerNet(**params, heads=cfg.get("heads", 4), layers=cfg.get("layers", 2), dropout=cfg.get("dropout", 0.5), norm=cfg.get("norm","LayerNorm"))
    raise ValueError(name)

def load_yaml_config(dataset: str, cfg_dir: str = "configs") -> Dict[str, Any]:
    path = Path(cfg_dir) / f"{dataset}_v6.yaml"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        if yaml is not None:
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    return data or {}


# =========================================================
# CLI
# =========================================================

def main(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", default=["Cora"], help="Datasets")
    parser.add_argument("--models", nargs="+", default=["ppgnn"], help="Models to train")
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--config-dir", type=str, default="configs")
    # scheme A toggles
    parser.add_argument("--custom", type=int, default=None)
    parser.add_argument("--custom-gnn", type=str, default=None,
                        choices=["gcn","sage","gat","transformer","tagcn","graphconv","cheb","arma","sgc","gin","appnp"])
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--clip_value", type=float, default=None)
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

            # override PPGNN scheme A from CLI if provided
            if model_name == "ppgnn":
                if args.custom is not None: model_params["custom"] = int(args.custom)
                if args.custom_gnn is not None: model_params["custom_gnn"] = args.custom_gnn
                if args.heads is not None: model_params["heads"] = int(args.heads)
            model_cfg["model"] = model_params

            train_cfg = model_cfg.get("train", {})
            for key, default in TRAIN_DEFAULTS.items():
                arg_val = getattr(args, key)
                if arg_val is not None:
                    train_cfg[key] = arg_val
                else:
                    train_cfg.setdefault(key, default)
            model_cfg["train"] = train_cfg

            print("Model parameters:")
            for k, v in model_params.items(): print(f"  {k}: {v}")
            print("Training parameters:")
            for k, v in train_cfg.items(): print(f"  {k}: {v}")

            model = get_model(model_name, info["in_channels"], info["out_channels"], model_params, info["level"])
            # Choose trainer by task level
            if info["level"] == "node":
                trainer = train_node_classification
            else:
                if info["task"] == "classification":
                    trainer = train_graph_classification
                else:
                    raise RuntimeError(f"No trainer for graph-{info['task']}")
            data_or_loader = info.get("data") or info.get("loaders")

            if mlflow is not None:
                mlflow.set_experiment(dataset_name)
                with mlflow.start_run(run_name=model_name):
                    mlflow.log_param("dataset", dataset_name)
                    mlflow.log_param("model", model_name)
                    for k, v in model_params.items(): mlflow.log_param(f"model_{k}", v)
                    for k, v in train_cfg.items(): mlflow.log_param(f"train_{k}", v)
                    trainer(data_or_loader, model, model_name, train_cfg, None)
            else:
                trainer(data_or_loader, model, model_name, train_cfg, None)


if __name__ == "__main__":
    main()
