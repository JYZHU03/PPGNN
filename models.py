# ---------------- models.py ----------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from lvconv import LVConv
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_add_pool


class PPGNN(nn.Module):
    """Predator–Prey GNN with semi-implicit diffusion (Jacobi)."""

    def __init__(
        self,
        in_channels: int,
        hidden: int,
        num_classes: int,
        layers: int = 15,
        dt: float = 0.1,  # 固定步长
        dropout: float = 0.3,  # 轻微 dropout，避免弱信号被抑制
        norm_type: str = "sym",  # 'sym' or 'rw'
        jacobi_steps: int = 2,  # Jacobi 近似步数
        use_x_only: bool = True,  # 只用 X 通道读出
        y0_mode: str = "ones",  # Y 初值
        alpha0: float = 0.2,
        beta0: float = 0.1,
        dx0: float = 0.15,
        dy0: float = 0.8,
        norm: str = "BatchNorm1d",
        level: str = "node",
        lift_type: str = "linear",  # 'linear' or 'mlp'
        lift_layers: int = 2,  # 仅在使用 MLP 时生效
    ):
        super().__init__()
        self.hidden = hidden
        self.dropout = dropout
        self.use_x_only = use_x_only
        self.y0_mode = y0_mode
        self.graph_head = level == "graph"

        lift_type = (lift_type or "linear").lower()

        def build_lift():
            if lift_type == "mlp" and lift_layers > 1:
                mlp = [nn.Linear(in_channels, hidden), nn.ReLU()]
                for _ in range(lift_layers - 2):
                    mlp.extend([nn.Linear(hidden, hidden), nn.ReLU()])
                mlp.append(nn.Linear(hidden, hidden))
                return nn.Sequential(*mlp)
            return nn.Linear(in_channels, hidden)

        # 两套 lift
        self.lift_x = build_lift()
        self.lift_y = build_lift()

        # PP 层堆叠
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
                )
                for _ in range(layers)
            ]
        )

        norm = (norm or "none").lower()

        def make_norm():
            if norm == "batchnorm1d" or norm == "batchnorm":
                return nn.BatchNorm1d(2 * hidden)
            if norm == "layernorm" or norm == "ln":
                return nn.LayerNorm(2 * hidden)
            return nn.Identity()

        self.norms = nn.ModuleList([make_norm() for _ in range(layers)])

        # 残差门控 τ
        self.taus = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.7)) for _ in range(layers)]
        )

        # 节点级输出头
        out_dim = hidden if use_x_only else 2 * hidden
        self.lin_out = nn.Linear(out_dim, num_classes)

        # 图级读出头
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

    def forward(self, data):
        X0 = torch.tanh(self.lift_x(data.x.float()))
        if self.y0_mode == "learned":
            Y0 = torch.tanh(self.lift_y(data.x.float()))
        else:
            Y0 = torch.ones_like(X0)

        h = torch.cat([X0, Y0], dim=-1)

        for conv, tau_param, norm in zip(self.layers, self.taus, self.norms):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h_hat = conv(h, data.edge_index)
            tau = torch.sigmoid(tau_param)
            h = (1 - tau) * h + tau * h_hat
            h = norm(h)
            # h = norm(h_hat)

        if self.graph_head and hasattr(data, "batch") and self.readout is not None:
            if self.use_x_only:
                X, _ = torch.split(h, self.hidden, dim=-1)
                g = global_add_pool(X, data.batch)
            else:
                g = global_add_pool(h, data.batch)
            g = F.dropout(g, p=0.1, training=self.training)
            return self.readout(self.logit_scale * g)

        if self.use_x_only:
            X, _ = torch.split(h, self.hidden, dim=-1)
            X = F.dropout(X, p=0.1, training=self.training)
            return self.lin_out(self.logit_scale * X)
        h = F.dropout(h, p=0.1, training=self.training)
        return self.lin_out(self.logit_scale * h)


# ---------- 改进后的 Baseline GCN ----------
class GCN(nn.Module):
    """
    安全策略：
    - 默认只输出节点级表示，避免误触发图级池化。
    - 当 level="graph" 且 batch 存在时，执行 global_add_pool -> MLP。
    - 始终保留 lin_out，便于在节点任务或未触发读出时直接返回节点级输出。
    """

    def __init__(
        self,
        in_channels,
        hidden,
        num_classes,
        layers=5,
        dropout=0.2,
        norm: str = "BatchNorm1d",
        level: str = "node",
    ):
        super().__init__()
        self.dropout = dropout
        self.graph_head = level == "graph"

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        norm = (norm or "none").lower()

        def make_norm():
            if norm == "batchnorm1d" or norm == "batchnorm":
                return nn.BatchNorm1d(hidden)
            if norm == "layernorm" or norm == "ln":
                return nn.LayerNorm(hidden)
            return nn.Identity()

        # 第 1 层
        self.convs.append(GCNConv(in_channels, hidden))
        self.norms.append(make_norm())
        # 后续层
        for _ in range(layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
            self.norms.append(make_norm())

        # 节点级输出头：无论如何都保留
        self.lin_out = nn.Linear(hidden, num_classes)

        # 图级读出头：仅在需要时使用（图任务）
        self.readout = (
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_classes),
            )
            if self.graph_head
            else None
        )

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 仅当明确开启图读出且 batch 存在时，做图级池化
        if self.graph_head and hasattr(data, "batch") and self.readout is not None:
            g = global_add_pool(x, data.batch)
            return self.readout(g)

        # 否则返回节点级输出（节点分类/回归；或把图级读出交给 tasks.py 的 _maybe_pool）
        return self.lin_out(x)



class GraphSAGE(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden,
        num_classes,
        layers=2,
        dropout=0.5,
        norm: str = "None",
        level: str = "node",
    ):
        super().__init__()
        self.dropout = dropout
        self.graph_head = level == "graph"

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        norm = (norm or "none").lower()

        def make_norm():
            if norm == "batchnorm1d" or norm == "batchnorm":
                return nn.BatchNorm1d(hidden)
            if norm == "layernorm" or norm == "ln":
                return nn.LayerNorm(hidden)
            return nn.Identity()

        self.convs.append(SAGEConv(in_channels, hidden))
        self.norms.append(make_norm())
        for _ in range(layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
            self.norms.append(make_norm())

        # 节点级输出头
        self.lin_out = nn.Linear(hidden, num_classes)

        # 图级读出头
        self.readout = (
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_classes),
            )
            if self.graph_head
            else None
        )

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.graph_head and hasattr(data, "batch") and self.readout is not None:
            g = global_add_pool(x, data.batch)
            return self.readout(g)
        return self.lin_out(x)


class GAT(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden,
        num_classes,
        heads=8,
        layers=2,
        dropout=0.5,
        norm: str = "None",
        level: str = "node",
    ):
        super().__init__()
        self.dropout = dropout
        self.graph_head = level == "graph"

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        norm = (norm or "none").lower()

        def make_norm():
            if norm == "batchnorm1d" or norm == "batchnorm":
                return nn.BatchNorm1d(hidden * heads)
            if norm == "layernorm" or norm == "ln":
                return nn.LayerNorm(hidden * heads)
            return nn.Identity()

        self.convs.append(GATConv(in_channels, hidden, heads=heads))
        self.norms.append(make_norm())
        for _ in range(layers - 1):
            self.convs.append(GATConv(hidden * heads, hidden, heads=heads))
            self.norms.append(make_norm())

        # 节点级输出头
        self.lin_out = nn.Linear(hidden * heads, num_classes)

        # 图级读出头
        self.readout = (
            nn.Sequential(
                nn.Linear(hidden * heads, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_classes),
            )
            if self.graph_head
            else None
        )

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.graph_head and hasattr(data, "batch") and self.readout is not None:
            g = global_add_pool(x, data.batch)
            return self.readout(g)
        return self.lin_out(x)
