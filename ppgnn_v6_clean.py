from __future__ import annotations

"""
Minimal PPGNN (LVConv-only) runner for Cora.
- Removes FA-LV and other baseline GNNs; keeps only the LVConv backbone.
- Configuration is read from Cora_v6_clean.yaml (model/train sections).
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Amazon, Planetoid, WebKB, WikipediaNetwork, Actor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.transforms import (
    Compose,
    NormalizeFeatures,
    RandomNodeSplit,
    ToUndirected,
)
from torch_geometric.datasets import Amazon
from torch_geometric.utils import degree

# Optional YAML dependency
try:  # pragma: no cover
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

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
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == y).float().mean().item()


def _softplus_inv(y: float) -> float:
    # numerically stable inverse softplus
    y = max(float(y), 1e-6)
    return math.log(math.exp(y) - 1.0)


# =========================================================
# LVConv (no FA-LV)
# =========================================================

class LVConv(MessagePassing):
    """
    Lotka–Volterra graph convolution layer with fixed diffusion kernel.
    - Two channels: R,F (each d dims); reaction coeffs alpha,beta,gamma,delta (per-channel);
      diffusion strengths Dx,Dy (scalars, positive).
    - Diffusion kernel S is fixed, feature-independent: sym/rw normalized adjacency, or a TAG-like
      polynomial over S (norm_type in {"sym","rw","tag","tag_sym","tag_rw"}).
    - Discretization selectable: semi-implicit Euler + Jacobi (imex) or explicit Euler.
    """

    def __init__(
        self,
        channels: int,
        dt: float = 0.1,
        norm_type: str = "sym",
        jacobi_steps: int = 2,
        solver: str = "explicit",  # "imex" or "explicit"
        learn_dt: int | bool = 0,
        eps: float = 1e-3,
        alpha0: float = 0.2,
        beta0: float = 0.1,
        dx0: float = 0.7,
        dy0: float = 0.8,
        tag_k: int = 3,
    ):
        super().__init__(aggr="add")
        allowed_norm = {"sym", "rw", "tag", "tag_sym", "tag_rw"}
        if norm_type not in allowed_norm:
            raise ValueError(f"norm_type must be one of {allowed_norm}, got {norm_type}")
        solver = (solver or "imex").lower()
        if solver not in {"imex", "explicit"}:
            raise ValueError(f"solver must be 'imex' or 'explicit', got {solver}")
        assert jacobi_steps >= 1
        self.d = int(channels)
        self.learn_dt = bool(learn_dt)
        if self.learn_dt:
            # initialize so that softplus(dt_param) ≈ dt (preserve user-specified step)
            self.dt_param = nn.Parameter(torch.tensor(_softplus_inv(float(dt))))
            self.register_buffer("dt_buffer", None)
        else:
            self.dt_param = None
            self.register_buffer("dt_buffer", torch.tensor(float(dt)))
        self.norm_type = norm_type
        self.jacobi_steps = int(jacobi_steps)
        self.solver = solver
        self.eps = float(eps)
        self.tag_k = int(tag_k)
        if self.norm_type.startswith("tag"):
            if self.tag_k < 0:
                raise ValueError("tag_k must be non-negative")
            # Shared scalar coefficients for polynomial S(X) = sum c_k S^k X
            self.tag_coeff = nn.Parameter(torch.ones(self.tag_k + 1))
        else:
            self.tag_coeff = None

        # Reaction parameters (pre-softplus, per-channel)
        self.alpha = nn.Parameter(torch.zeros(self.d))
        self.beta = nn.Parameter(torch.zeros(self.d))
        self.gamma = nn.Parameter(torch.zeros(self.d))
        self.delta = nn.Parameter(torch.zeros(self.d))

        # Diffusion parameters (pre-softplus, scalars)
        self.Dx = nn.Parameter(torch.tensor(0.0))
        self.Dy = nn.Parameter(torch.tensor(0.0))

        with torch.no_grad():
            self.alpha.fill_(_softplus_inv(alpha0))
            self.gamma.fill_(_softplus_inv(alpha0))
            self.beta.fill_(_softplus_inv(beta0))
            self.delta.fill_(_softplus_inv(beta0))
            self.Dx.fill_(_softplus_inv(dx0))
            self.Dy.fill_(_softplus_inv(dy0))

    def _base_norm_type(self) -> str:
        if self.norm_type in {"sym", "tag", "tag_sym"}:
            return "sym"
        if self.norm_type in {"rw", "tag_rw"}:
            return "rw"
        raise ValueError(f"Unexpected norm_type: {self.norm_type}")

    def _apply_kernel(self, Z: torch.Tensor, edge_index: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        # Base normalized adjacency (one-hop)
        if not self.norm_type.startswith("tag"):
            return self.propagate(edge_index, x=Z, norm=norm)
        # TAG-like polynomial S(X) = sum_{k=0}^K c_k S^k X with shared scalar coeffs
        outs = [Z]
        cur = Z
        for _ in range(self.tag_k):
            cur = self.propagate(edge_index, x=cur, norm=norm)
            outs.append(cur)
        assert self.tag_coeff is not None
        return sum(c * o for c, o in zip(self.tag_coeff, outs))

    def _eff_params(self):
        """Return positive parameters."""
        a = F.softplus(self.alpha) + self.eps
        b = F.softplus(self.beta) + self.eps
        g = F.softplus(self.gamma) + self.eps
        d = F.softplus(self.delta) + self.eps
        Dx = F.softplus(self.Dx) + self.eps
        Dy = F.softplus(self.Dy) + self.eps
        return a, b, g, d, Dx, Dy

    def _dt(self) -> torch.Tensor:
        if self.learn_dt:
            return F.softplus(self.dt_param) + self.eps
        return self.dt_buffer

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor):
        X, Y = torch.split(h, self.d, dim=-1)  # X->R, Y->F

        # Norm weights from normalized adjacency (sym or rw)
        N = h.size(0)
        row, col = edge_index
        base_norm = self._base_norm_type()
        if base_norm == "sym":
            deg = degree(row, N, dtype=X.dtype).clamp(min=1)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        else:
            deg = degree(col, N, dtype=X.dtype).clamp(min=1)
            norm = (1.0 / deg)[col]

        # Effective parameters
        a, b, g, d, Dx, Dy = self._eff_params()
        dt = self._dt()

        # Reaction terms
        RX = a * X - b * (X * Y)
        RY = d * (X * Y) - g * Y
        ax = dt * Dx
        ay = dt * Dy

        if self.solver == "explicit":
            SX = self._apply_kernel(X, edge_index, norm)
            SY = self._apply_kernel(Y, edge_index, norm)
            X_out = X + dt * (RX + Dx * (SX - X))
            Y_out = Y + dt * (RY + Dy * (SY - Y))
            return torch.cat([X_out, Y_out], dim=-1)

        # semi-implicit (IMEX) with Jacobi inner steps
        RHS_X = X + dt * RX
        RHS_Y = Y + dt * RY
        denom_x = 1.0 + ax
        denom_y = 1.0 + ay
        Xk, Yk = X, Y
        for _ in range(self.jacobi_steps):
            SXk = self._apply_kernel(Xk, edge_index, norm)
            SYk = self._apply_kernel(Yk, edge_index, norm)
            Xk = (RHS_X + ax * SXk) / denom_x
            Yk = (RHS_Y + ay * SYk) / denom_y

        return torch.cat([Xk, Yk], dim=-1)

    def message(self, x_j: torch.Tensor, norm: torch.Tensor):
        # Scale neighbor features by normalized edge weights
        return norm.view(-1, 1) * x_j


# =========================================================
# Model
# =========================================================

class PPGNN(nn.Module):
    """Predator–Prey GNN with semi-implicit diffusion (no FA-LV)."""

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
        solver: str = "imex",
        learn_dt: int | bool = 0,
        use_x_only: bool = False,
        y0_mode: str = "learned",
        alpha0: float = 0.2,
        beta0: float = 0.1,
        dx0: float = 0.7,
        dy0: float = 0.8,
        tag_k: int = 0,
        norm: str = "BatchNorm1d",
        act: str = "identity",
        act_x0: str = "relu",
        act_y0: str = "tanh",
        lift_type: str = "linear",
        lift_layers: int = 2,
    ):
        super().__init__()
        self.hidden = hidden
        self.dropout = dropout
        self.use_x_only = bool(use_x_only)
        self.y0_mode = y0_mode
        self.act = self._make_activation(act)
        self.act_x0 = self._make_activation(act_x0)
        self.act_y0 = self._make_activation(act_y0)

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
                    solver=solver,
                    learn_dt=learn_dt,
                    alpha0=alpha0,
                    beta0=beta0,
                    dx0=dx0,
                    dy0=dy0,
                    tag_k=tag_k,
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
        self.logit_scale = nn.Parameter(torch.tensor(2.5))

    @staticmethod
    def _make_activation(name: str) -> nn.Module:
        n = (name or "identity").lower()
        if n in {"tanh"}:
            return nn.Tanh()
        if n in {"relu"}:
            return nn.ReLU()
        if n in {"gelu"}:
            return nn.GELU()
        if n in {"sigmoid"}:
            return nn.Sigmoid()
        if n in {"softplus"}:
            return nn.Softplus()
        if n in {"leaky_relu", "lrelu"}:
            return nn.LeakyReLU(0.01)
        if n in {"identity", "none"}:
            return nn.Identity()
        raise ValueError(f"Unknown activation: {name}")

    def forward(self, data):
        X0 = self.act_x0(self.lift_x(data.x.float()))
        if self.y0_mode == "learned":
            Y0 = self.act_y0(self.lift_y(data.x.float()))
        else:
            Y0 = torch.ones_like(X0)
        h = torch.cat([X0, Y0], dim=-1)

        for conv, tau_p, norm in zip(self.layers, self.taus, self.norms):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h_hat = conv(h, data.edge_index)
            h_hat = self.act(h_hat)
            tau = torch.sigmoid(tau_p)
            h = (1 - tau) * h + tau * h_hat
            h = norm(h)

        if self.use_x_only:
            X, _ = torch.split(h, self.hidden, dim=-1)
            X = F.dropout(X, p=0.1, training=self.training)
            return self.lin_out(self.logit_scale * X)

        h = F.dropout(h, p=0.1, training=self.training)
        return self.lin_out(self.logit_scale * h)


# =========================================================
# Data + config
# =========================================================

def load_dataset(name: str) -> Dict[str, Any]:
    name = name.strip()
    lname = name.lower()
    if lname in {"cora", "citeseer", "pubmed"}:
        proper = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}[lname]
        dataset = Planetoid(
            root="data/Planetoid",
            name=proper,
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

    if lname in {"computers", "photo"}:
        proper = {"computers": "Computers", "photo": "Photo"}[lname]
        dataset = Amazon(
            root="data/Amazon",
            name=proper,
            transform=Compose(
                [
                    NormalizeFeatures(),
                    ToUndirected(),
                    RandomNodeSplit(num_val=0.1, num_test=0.2),
                ]
            ),
        )
        data = dataset[0].to(DEVICE)
        return dict(
            level="node",
            task="classification",
            in_channels=dataset.num_features,
            out_channels=dataset.num_classes,
            data=data,
        )

    if lname in {"texas", "wisconsin", "cornell"}:
        proper = {"texas": "Texas", "wisconsin": "Wisconsin", "cornell": "Cornell"}[lname]
        dataset = WebKB(
            root="data/WebKB",
            name=proper,
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

    if lname in {"chameleon", "squirrel"}:
        # WikipediaNetwork uses lowercase names
        dataset = WikipediaNetwork(
            root="data/WikipediaNetwork",
            name=lname,
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

    if lname in {"actor"}:
        dataset = Actor(
            root="data/Actor",
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

    raise ValueError(f"Unknown dataset: {name}")


def load_yaml_config(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    with p.open("r", encoding="utf-8") as f:
        if yaml is not None:
            return yaml.safe_load(f) or {}
        return json.load(f) or {}


# =========================================================
# Training
# =========================================================

def _setup_optim(model, train_cfg: Dict[str, float]):
    epochs = int(train_cfg.get("epochs", 600))
    optim = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.get("lr", 2e-3),
        weight_decay=train_cfg.get("weight_decay", 5e-4),
    )
    clip_value = train_cfg.get("clip_value", 5.0)
    return epochs, optim, clip_value


def train_node_classification(data, model, train_cfg: Dict[str, Any]) -> None:
    model = model.to(DEVICE)
    data = data.to(DEVICE)
    epochs, optim, clip_value = _setup_optim(model, train_cfg)
    best_val = best_test = best_epoch = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        optim.zero_grad()
        out = model(data)
        train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
        if train_mask.ndim > 1:
            train_mask, val_mask, test_mask = train_mask[:, 0], val_mask[:, 0], test_mask[:, 0]
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        if clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optim.step()

        model.eval()
        with torch.no_grad():
            logits = model(data)
            tr = accuracy(logits[train_mask], data.y[train_mask])
            va = accuracy(logits[val_mask], data.y[val_mask])
            te = accuracy(logits[test_mask], data.y[test_mask])

        if va > best_val:
            best_val, best_test, best_epoch = va, te, epoch

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            dt_values = []
            if hasattr(model, "layers"):
                for conv in model.layers:
                    if hasattr(conv, "_dt"):
                        dt_curr = conv._dt().detach().mean().item()
                        dt_values.append(dt_curr)
            dt_str = f"  dt[0]:{dt_values[0]:.4f}" if dt_values else ""
            print(f"Epoch {epoch:03d}  tr_acc:{tr:.3f}  va_acc:{va:.3f}  te_acc:{te:.3f}{dt_str}")

    print(f"★ Best@val: epoch={best_epoch}  val={best_val:.3f}  test@best_val={best_test:.3f}")


# =========================================================
# CLI
# =========================================================

def build_model_from_cfg(cfg: Dict[str, Any], in_channels: int, out_channels: int) -> PPGNN:
    cfg = cfg or {}
    return PPGNN(
        in_channels=in_channels,
        hidden=cfg.get("hidden", 128),
        num_classes=out_channels,
        layers=cfg.get("layers", 15),
        dt=cfg.get("dt", 0.1),
        dropout=cfg.get("dropout", 0.4),
        norm_type=cfg.get("norm_type", "sym"),
        jacobi_steps=cfg.get("jacobi_steps", 2),
        solver=cfg.get("solver", "imex"),
        learn_dt=cfg.get("learn_dt", 1),
        use_x_only=cfg.get("use_x_only", False),
        y0_mode=cfg.get("y0_mode", "learned"),
        alpha0=cfg.get("alpha0", 0.2),
        beta0=cfg.get("beta0", 0.1),
        dx0=cfg.get("dx0", 0.7),
        dy0=cfg.get("dy0", 0.8),
        tag_k=cfg.get("tag_k", 0),
        norm=cfg.get("norm", "BatchNorm1d"),
        act=cfg.get("act", "tanh"),
        act_x0=cfg.get("act_x0", "tanh"),
        act_y0=cfg.get("act_y0", "tanh"),
        lift_type=cfg.get("lift_type", "linear"),
        lift_layers=cfg.get("lift_layers", 2),
    )


def main(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="Cora",
        help="Dataset name (Cora/CiteSeer/PubMed/Computers/Photo/Texas/Wisconsin/Cornell/Actor/Chameleon/Squirrel)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/Cora_v6_clean.yaml",
        help="YAML config path (if not set, try configs/{dataset}_v6_clean.yaml)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="override epochs")
    parser.add_argument("--lr", type=float, default=None, help="override learning rate")
    parser.add_argument("--weight_decay", type=float, default=None, help="override weight decay")
    parser.add_argument("--clip_value", type=float, default=None, help="override grad clipping")
    args = parser.parse_args(argv)

    set_seed(0)
    data_info = load_dataset(args.dataset)
    default_cfg_path = Path(f"configs/{args.dataset}_v6_clean.yaml")
    cfg_path = args.config if args.config is not None else (default_cfg_path if default_cfg_path.is_file() else None)
    cfg_all = load_yaml_config(cfg_path).get("ppgnn", {})
    model_cfg = cfg_all.get("model", {})
    train_cfg = cfg_all.get("train", {})

    # CLI overrides
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.lr is not None:
        train_cfg["lr"] = args.lr
    if args.weight_decay is not None:
        train_cfg["weight_decay"] = args.weight_decay
    if args.clip_value is not None:
        train_cfg["clip_value"] = args.clip_value

    print("Model parameters:")
    for k, v in model_cfg.items():
        print(f"  {k}: {v}")
    print("Training parameters:")
    for k, v in train_cfg.items():
        print(f"  {k}: {v}")

    model = build_model_from_cfg(model_cfg, data_info["in_channels"], data_info["out_channels"])
    train_node_classification(data_info["data"], model, train_cfg)


if __name__ == "__main__":
    main()
