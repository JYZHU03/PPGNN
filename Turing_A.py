from __future__ import annotations

"""
Design A runner: Pattern-forming (Turing) Activator–Inhibitor RD-GNN (AI-GNN) for node classification.

Key idea (Design A):
- Each hidden feature coordinate is a 2D activator–inhibitor state (u,v).
- Local reaction is (near equilibrium) linear with Jacobian J = [[a,b],[c,d]] at h*=0.
- Diffusion is anisotropic: C = diag(Du, Dv), typically Dv >> Du.
- A lightweight "verifiable band" regularizer enforces a diffusion-driven instability band:
    stable at mu=0 (local stability), but unstable for mu in (mu_a, mu_b).

This script is standalone and mirrors your Design B code structure:
- same dataset loader (Planetoid etc.)
- same normalization options (sym/rw/tag/tag_sym/tag_rw)
- same solvers (explicit / IMEX with Jacobi)
- YAML config compatible with your existing ppgnn.model fields;
  Design A adds optional fields under ppgnn.model:
    a0, b0, c0, s0, du0, dv0, mu_a, mu_b, turing_margin, turing_mu_out
  and under ppgnn.train:
    turing_reg_weight
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, WebKB, WikipediaNetwork, Actor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.transforms import Compose, NormalizeFeatures, RandomNodeSplit, ToUndirected
from torch_geometric.utils import degree

# Optional YAML dependency
try:  # pragma: no cover
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

# Optional ogb dependency
try:  # pragma: no cover
    from ogb.nodeproppred import PygNodePropPredDataset  # type: ignore
except Exception:  # pragma: no cover
    PygNodePropPredDataset = None

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
    y = max(float(y), 1e-8)
    return math.log(math.exp(y) - 1.0)


# =========================================================
# Design A: Activator–Inhibitor RD layer (pattern-forming)
# =========================================================

class AIConv(MessagePassing):
    """
    Activator–Inhibitor reaction–diffusion layer.

    State at each node: h = [U, V] where U,V in R^d (d=channels).
    Reaction (per coordinate j, shared scalars by default):
        d/dt [u; v] = J [u; v]  - kappa * [u^3; v^3]
      with J = [[a,b],[c,d]] and equilibrium at (0,0).

    Diffusion (graph):
        + Du * sum_j G_ij u_j   and + Dv * sum_j G_ij v_j
      implemented as Du*(S u - u), Dv*(S v - v) where G = S - I.

    Turing-band regularizer (cheap, no eigendecomposition):
      enforce local stability (mu=0) + instability at mu_a, mu_b + return to stability at mu_out.
      With tr(J) < 0, instability iff det(J - mu*diag(Du,Dv)) < 0.
    """

    def __init__(
        self,
        channels: int,
        dt: float = 0.1,
        norm_type: str = "sym",
        jacobi_steps: int = 2,
        solver: str = "imex",  # "imex" or "explicit"
        learn_dt: int | bool = 0,
        eps: float = 1e-3,
        tag_k: int = 3,

        # ---- J init (shared across all feature coordinates) ----
        a0: float = 0.7,
        b0: float = -1.8,   # will be forced negative
        c0: float = 2.2,    # will be forced positive
        s0: float = 2.0,    # trace = -s < 0 is enforced via d = -s - a

        # ---- diffusion init ----
        du0: float = 0.37,
        dv0: float = 17.4,

        # ---- saturation strength (fixed, no extra params) ----
        kappa: float = 1.0,

        # ---- band targets for the cheap regularizer ----
        mu_a: float = 0.3,
        mu_b: float = 1.5,
        turing_mu_out: float = 2.0,
        turing_margin: float = 1e-3,
    ):
        super().__init__(aggr="add")
        allowed_norm = {"sym", "rw", "tag", "tag_sym", "tag_rw"}
        if norm_type not in allowed_norm:
            raise ValueError(f"norm_type must be one of {allowed_norm}, got {norm_type}")
        solver = (solver or "imex").lower()
        if solver not in {"imex", "explicit"}:
            raise ValueError(f"solver must be 'imex' or 'explicit', got {solver}")
        assert jacobi_steps >= 1

        self.d_feat = int(channels)
        self.learn_dt = bool(learn_dt)
        if self.learn_dt:
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
            self.tag_coeff = nn.Parameter(torch.ones(self.tag_k + 1))
        else:
            self.tag_coeff = None

        # ----- Reaction params (shared scalars) -----
        # a is free (can be +/-), b is forced negative, c forced positive, s forced positive
        self.a_param = nn.Parameter(torch.tensor(float(a0)))
        self.b_param = nn.Parameter(torch.tensor(_softplus_inv(abs(float(b0)))))  # b = -softplus(...)
        self.c_param = nn.Parameter(torch.tensor(_softplus_inv(abs(float(c0)))))  # c = +softplus(...)
        self.s_param = nn.Parameter(torch.tensor(_softplus_inv(max(float(s0), 1e-3))))  # s = +softplus(...)

        # ----- Diffusion params (positive scalars) -----
        self.Du_param = nn.Parameter(torch.tensor(_softplus_inv(float(du0))))
        self.Dv_param = nn.Parameter(torch.tensor(_softplus_inv(float(dv0))))

        self.kappa = float(kappa)

        # ----- Band targets (not learned) -----
        self.mu_a = float(mu_a)
        self.mu_b = float(mu_b)
        self.mu_out = float(turing_mu_out)
        self.turing_margin = float(turing_margin)

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

        # TAG-like polynomial S(X) = sum_{k=0}^K c_k S^k X
        outs = [Z]
        cur = Z
        for _ in range(self.tag_k):
            cur = self.propagate(edge_index, x=cur, norm=norm)
            outs.append(cur)
        assert self.tag_coeff is not None
        return sum(c * o for c, o in zip(self.tag_coeff, outs))

    def _dt(self) -> torch.Tensor:
        if self.learn_dt:
            return F.softplus(self.dt_param) + self.eps
        return self.dt_buffer

    def _eff_params(self):
        """
        Return effective scalars:
          a (free), b<0, c>0, s>0, d = -s - a => trace(J) = -s < 0 always
          Du,Dv > 0
        """
        a = self.a_param
        b = -(F.softplus(self.b_param) + self.eps)
        c = (F.softplus(self.c_param) + self.eps)
        s = (F.softplus(self.s_param) + self.eps)
        d = -s - a
        Du = (F.softplus(self.Du_param) + self.eps)
        Dv = (F.softplus(self.Dv_param) + self.eps)
        return a, b, c, d, s, Du, Dv

    def turing_penalty(self) -> torch.Tensor:
        """
        Cheap differentiable penalties to *encourage* a Turing band.

        Conditions encouraged:
          det(J) > margin          (local stability with tr<0)
          B = a*Dv + d*Du > margin
          Delta = B^2 - 4 det(J) Du Dv > margin  (ensures two positive roots)
          det(M(mu_a)) < -margin   (inside band)
          det(M(mu_b)) < -margin   (inside band)
          det(M(mu_out)) > margin  (outside band, come back stable)

        NOTE: This is a "soft" constraint. You keep parameters learnable.
        """
        a, b, c, d, _, Du, Dv = self._eff_params()
        margin = torch.tensor(self.turing_margin, device=a.device, dtype=a.dtype)

        detJ = a * d - b * c
        B = a * Dv + d * Du
        Delta = B * B - 4.0 * detJ * Du * Dv

        def detM(mu: float) -> torch.Tensor:
            mu_t = torch.tensor(mu, device=a.device, dtype=a.dtype)
            return detJ - mu_t * B + (mu_t * mu_t) * Du * Dv

        # Want: detJ > margin, B > margin, Delta > margin
        p1 = F.softplus(margin - detJ)
        p2 = F.softplus(margin - B)
        p3 = F.softplus(margin - Delta)

        # Want: detM(mu_a) < -margin and detM(mu_b) < -margin
        p4 = F.softplus(detM(self.mu_a) + margin)
        p5 = F.softplus(detM(self.mu_b) + margin)

        # Want: detM(mu_out) > margin
        p6 = F.softplus(margin - detM(self.mu_out))

        return p1 + p2 + p3 + p4 + p5 + p6

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor):
        U, V = torch.split(h, self.d_feat, dim=-1)

        # Norm weights from normalized adjacency (sym or rw)
        N = h.size(0)
        row, col = edge_index
        base_norm = self._base_norm_type()
        if base_norm == "sym":
            deg = degree(row, N, dtype=U.dtype).clamp(min=1)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        else:
            deg = degree(col, N, dtype=U.dtype).clamp(min=1)
            norm = (1.0 / deg)[col]

        # Effective params
        a, b, c, d, _, Du, Dv = self._eff_params()
        dt = self._dt()

        # Reaction (vectorized across feature dims, scalars broadcast)
        # Linear part gives Jacobian J at (0,0); cubic saturates amplitude.
        RU = a * U + b * V - self.kappa * (U * U * U)
        RV = c * U + d * V - self.kappa * (V * V * V)

        ax = dt * Du
        ay = dt * Dv

        if self.solver == "explicit":
            SU = self._apply_kernel(U, edge_index, norm)
            SV = self._apply_kernel(V, edge_index, norm)
            U_out = U + dt * (RU + Du * (SU - U))
            V_out = V + dt * (RV + Dv * (SV - V))
            return torch.cat([U_out, V_out], dim=-1)

        # semi-implicit (IMEX) with Jacobi inner steps
        RHS_U = U + dt * RU
        RHS_V = V + dt * RV
        denom_u = 1.0 + ax
        denom_v = 1.0 + ay

        Uk, Vk = U, V
        for _ in range(self.jacobi_steps):
            SUk = self._apply_kernel(Uk, edge_index, norm)
            SVk = self._apply_kernel(Vk, edge_index, norm)
            Uk = (RHS_U + ax * SUk) / denom_u
            Vk = (RHS_V + ay * SVk) / denom_v

        return torch.cat([Uk, Vk], dim=-1)

    def message(self, x_j: torch.Tensor, norm: torch.Tensor):
        return norm.view(-1, 1) * x_j


# =========================================================
# Model (Design A)
# =========================================================

class AIGNN(nn.Module):
    """
    Design A model:
      - lift input x -> U0, V0
      - stack AIConv layers
      - linear classifier on final state
    """

    def __init__(
        self,
        in_channels: int,
        hidden: int,
        num_classes: int,
        layers: int = 8,
        dt: float = 0.1,
        dropout: float = 0.4,
        norm_type: str = "sym",
        jacobi_steps: int = 2,
        solver: str = "imex",
        learn_dt: int | bool = 1,
        use_x_only: bool = False,
        y0_mode: str = "learned",
        tag_k: int = 3,
        norm: str = "BatchNorm1d",
        act: str = "identity",
        act_x0: str = "tanh",
        act_y0: str = "tanh",
        lift_type: str = "linear",
        lift_layers: int = 2,

        # ---- Design A specific inits ----
        a0: float = 0.7,
        b0: float = -1.8,
        c0: float = 2.2,
        s0: float = 2.0,
        du0: float = 0.37,
        dv0: float = 17.4,
        kappa: float = 1.0,
        mu_a: float = 0.3,
        mu_b: float = 1.5,
        turing_mu_out: float = 2.0,
        turing_margin: float = 1e-3,
    ):
        super().__init__()
        self.hidden = hidden
        self.dropout = float(dropout)
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

        self.lift_u = build_lift()
        self.lift_v = build_lift()

        # AI layers
        self.layers = nn.ModuleList(
            [
                AIConv(
                    channels=hidden,
                    dt=dt,
                    norm_type=norm_type,
                    jacobi_steps=jacobi_steps,
                    solver=solver,
                    learn_dt=learn_dt,
                    tag_k=tag_k,
                    a0=a0,
                    b0=b0,
                    c0=c0,
                    s0=s0,
                    du0=du0,
                    dv0=dv0,
                    kappa=kappa,
                    mu_a=mu_a,
                    mu_b=mu_b,
                    turing_mu_out=turing_mu_out,
                    turing_margin=turing_margin,
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

        # Residual gate tau (same style as your B code)
        self.taus = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(layers)])

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

    def turing_regularizer(self) -> torch.Tensor:
        # Average per-layer penalties (keeps scale stable w.r.t. depth)
        ps = [conv.turing_penalty() for conv in self.layers]
        return torch.stack(ps).mean()

    def forward(self, data):
        U0 = self.act_x0(self.lift_u(data.x.float()))
        if self.y0_mode == "learned":
            V0 = self.act_y0(self.lift_v(data.x.float()))
        else:
            V0 = torch.zeros_like(U0)  # equilibrium at 0 fits the theory
        h = torch.cat([U0, V0], dim=-1)

        for conv, tau_p, norm in zip(self.layers, self.taus, self.norms):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h_hat = conv(h, data.edge_index)
            h_hat = self.act(h_hat)
            tau = torch.sigmoid(tau_p)
            h = (1 - tau) * h + tau * h_hat
            h = norm(h)

        if self.use_x_only:
            U, _ = torch.split(h, self.hidden, dim=-1)
            U = F.dropout(U, p=0.1, training=self.training)
            return self.lin_out(self.logit_scale * U)

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
            transform=Compose([NormalizeFeatures(), ToUndirected(), RandomNodeSplit(num_val=0.1, num_test=0.2)]),
        )
        data = dataset[0].to(DEVICE)
        return dict(
            level="node",
            task="classification",
            in_channels=dataset.num_features,
            out_channels=dataset.num_classes,
            data=data,
        )

    if lname in {"cs", "coauthorcs", "coauthor_cs", "coauthor-cs"}:
        dataset = Coauthor(
            root="data/Coauthor",
            name="CS",
            transform=Compose([NormalizeFeatures(), ToUndirected(), RandomNodeSplit(num_val=0.1, num_test=0.2)]),
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

    if lname in {"ogbn-arxiv", "ogbn_arxiv"}:
        if PygNodePropPredDataset is None:
            raise RuntimeError("ogb not installed; cannot load ogbn-arxiv")
        dataset = PygNodePropPredDataset(root="data/ogbn-arxiv", name="ogbn-arxiv")
        data = dataset[0]
        split_idx = dataset.get_idx_split()
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[split_idx["train"]] = True
        data.val_mask[split_idx["valid"]] = True
        data.test_mask[split_idx["test"]] = True
        data.y = data.y.squeeze()
        data = data.to(DEVICE)
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


def train_node_classification(data, model: AIGNN, train_cfg: Dict[str, Any]) -> None:
    model = model.to(DEVICE)
    data = data.to(DEVICE)

    epochs, optim, clip_value = _setup_optim(model, train_cfg)
    turing_reg_weight = float(train_cfg.get("turing_reg_weight", 1e-3))

    best_val = best_test = best_epoch = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        optim.zero_grad()
        out = model(data)
        train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
        if train_mask.ndim > 1:
            train_mask, val_mask, test_mask = train_mask[:, 0], val_mask[:, 0], test_mask[:, 0]

        ce = F.cross_entropy(out[train_mask], data.y[train_mask])
        reg = model.turing_regularizer() if turing_reg_weight > 0 else ce.new_zeros(())
        loss = ce + turing_reg_weight * reg

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
            # print dt of first layer for quick sanity
            dt0 = None
            if hasattr(model, "layers") and len(model.layers) > 0:
                dt0 = model.layers[0]._dt().detach().mean().item()
            dt_str = f"  dt0:{dt0:.4f}" if dt0 is not None else ""
            reg_str = f"  reg:{reg.detach().item():.4e}" if turing_reg_weight > 0 else ""
            print(f"Epoch {epoch:03d}  loss:{loss.item():.4f}  tr:{tr:.3f}  va:{va:.3f}  te:{te:.3f}{dt_str}{reg_str}")

    print(f"★ Best@val: epoch={best_epoch}  val={best_val:.3f}  test@best_val={best_test:.3f}")


# =========================================================
# CLI / Build from config
# =========================================================

def build_model_from_cfg(cfg: Dict[str, Any], in_channels: int, out_channels: int) -> AIGNN:
    cfg = cfg or {}

    # keep compatibility with your old cfg names; add new ones if present
    hidden = int(cfg.get("hidden", 128))
    layers = int(cfg.get("layers", 8))
    dt = float(cfg.get("dt", 0.1))
    dropout = float(cfg.get("dropout", 0.4))
    norm_type = str(cfg.get("norm_type", "sym"))
    jacobi_steps = int(cfg.get("jacobi_steps", 2))
    solver = str(cfg.get("solver", "imex"))
    learn_dt = bool(cfg.get("learn_dt", 1))
    use_x_only = bool(cfg.get("use_x_only", False))
    y0_mode = str(cfg.get("y0_mode", "learned"))
    tag_k = int(cfg.get("tag_k", 0))
    norm = str(cfg.get("norm", "BatchNorm1d"))
    act = str(cfg.get("act", "identity"))
    act_x0 = str(cfg.get("act_x0", "tanh"))
    act_y0 = str(cfg.get("act_y0", "tanh"))
    lift_type = str(cfg.get("lift_type", "linear"))
    lift_layers = int(cfg.get("lift_layers", 2))

    # Design A specific defaults (safe even if not in YAML)
    a0 = float(cfg.get("a0", 0.7))
    b0 = float(cfg.get("b0", -1.8))
    c0 = float(cfg.get("c0", 2.2))
    s0 = float(cfg.get("s0", 2.0))

    # If user only has dx0/dy0 in YAML, reuse them as du0/dv0 defaults.
    du0 = float(cfg.get("du0", cfg.get("dx0", 0.37)))
    dv0 = float(cfg.get("dv0", cfg.get("dy0", 17.4)))

    kappa = float(cfg.get("kappa", 1.0))
    mu_a = float(cfg.get("mu_a", 0.3))
    mu_b = float(cfg.get("mu_b", 1.5))
    turing_mu_out = float(cfg.get("turing_mu_out", 2.0))
    turing_margin = float(cfg.get("turing_margin", 1e-3))

    return AIGNN(
        in_channels=in_channels,
        hidden=hidden,
        num_classes=out_channels,
        layers=layers,
        dt=dt,
        dropout=dropout,
        norm_type=norm_type,
        jacobi_steps=jacobi_steps,
        solver=solver,
        learn_dt=learn_dt,
        use_x_only=use_x_only,
        y0_mode=y0_mode,
        tag_k=tag_k,
        norm=norm,
        act=act,
        act_x0=act_x0,
        act_y0=act_y0,
        lift_type=lift_type,
        lift_layers=lift_layers,
        a0=a0,
        b0=b0,
        c0=c0,
        s0=s0,
        du0=du0,
        dv0=dv0,
        kappa=kappa,
        mu_a=mu_a,
        mu_b=mu_b,
        turing_mu_out=turing_mu_out,
        turing_margin=turing_margin,
    )


def main(argv: Optional[Iterable[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="Computers",
        help="Dataset name (Cora/CiteSeer/PubMed/Computers/Photo/CS/ogbn-arxiv/Texas/Wisconsin/Cornell/Actor/Chameleon/Squirrel)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/Turing/Computer_v6_clean_turing.yaml",
        help="YAML config path (reads ppgnn.model + ppgnn.train)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="override epochs")
    parser.add_argument("--lr", type=float, default=None, help="override learning rate")
    parser.add_argument("--weight_decay", type=float, default=None, help="override weight decay")
    parser.add_argument("--clip_value", type=float, default=None, help="override grad clipping")
    parser.add_argument("--turing_reg_weight", type=float, default=None, help="override turing regularizer weight")
    args = parser.parse_args(argv)

    set_seed(0)
    data_info = load_dataset(args.dataset)

    cfg_all = load_yaml_config(args.config).get("ppgnn", {})
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
    if args.turing_reg_weight is not None:
        train_cfg["turing_reg_weight"] = args.turing_reg_weight

    print("Model parameters (ppgnn.model):")
    for k, v in model_cfg.items():
        print(f"  {k}: {v}")
    print("Training parameters (ppgnn.train):")
    for k, v in train_cfg.items():
        print(f"  {k}: {v}")

    model = build_model_from_cfg(model_cfg, data_info["in_channels"], data_info["out_channels"])
    train_node_classification(data_info["data"], model, train_cfg)


if __name__ == "__main__":
    main()
