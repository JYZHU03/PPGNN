# ---------------- models.py ----------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from lvconv import LVConv
from torch_geometric.nn import GCNConv, GATConv, SAGEConv


class PPGNN(nn.Module):
    """Predator–Prey GNN with semi-implicit diffusion (Jacobi)."""
    def __init__(
        self,
        in_channels: int,
        hidden: int,
        num_classes: int,
        layers: int = 15,
        dt: float = 0.1,               # 固定步长（你这组更稳）
        dropout: float = 0.3,          # 降一点，避免把弱信号掉没
        norm_type: str = "sym",        # 'sym' or 'rw'
        jacobi_steps: int = 2,         # 回到 2 步（和你 0.79 那组一致）
        use_x_only: bool = True,       # 只用 X 通道读出
        y0_mode: str = "ones",         # Y 初值先用常数 1，训练更稳
    ):
        super().__init__()
        self.hidden = hidden
        self.dropout = dropout
        self.use_x_only = use_x_only
        self.y0_mode = y0_mode

        # 两套 lift（用 tanh，训练更顺滑）
        self.lift_x = nn.Linear(in_channels, hidden)
        self.lift_y = nn.Linear(in_channels, hidden)

        # PP 层堆叠（固定 dt）
        self.layers = nn.ModuleList([
            LVConv(
                channels=hidden,
                dt=dt,
                norm_type=norm_type,
                jacobi_steps=jacobi_steps,
            )
            for _ in range(layers)
        ])

        # 每层一个残差门控 τ（初值稍大，鼓励更新）
        self.taus = nn.ParameterList([nn.Parameter(torch.tensor(0.7)) for _ in range(layers)])

        # 读出（线性 + 温度缩放 + 轻微 dropout）
        out_dim = hidden if use_x_only else 2 * hidden
        self.readout = nn.Linear(out_dim, num_classes)
        self.logit_scale = nn.Parameter(torch.tensor(2.5))  # 温度参数：拉低“温”软最大

    def forward(self, data):
        # lift：改回 tanh（避免 ReLU 大面积卡零，把 LV 乘法项打没）
        X0 = torch.tanh(self.lift_x(data.x))
        if self.y0_mode == "learned":
            Y0 = torch.tanh(self.lift_y(data.x))
        else:
            Y0 = torch.ones_like(X0)

        h = torch.cat([X0, Y0], dim=-1)

        for conv, tau_param in zip(self.layers, self.taus):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h_hat = conv(h, data.edge_index)      # 半隐式一步（Jacobi 近似）
            tau = torch.sigmoid(tau_param)        # 保证在 (0,1)
            h = (1 - tau) * h + tau * h_hat       # 残差门控

        # 读出：只用 X + 轻微 dropout + 温度缩放
        if self.use_x_only:
            X, _ = torch.split(h, self.hidden, dim=-1)
            X = F.dropout(X, p=0.1, training=self.training)
            return self.readout(self.logit_scale * X)
        else:
            h = F.dropout(h, p=0.1, training=self.training)
            return self.readout(self.logit_scale * h)


# ---------- Baseline GNNs（保持不变） ----------
class GCN(nn.Module):
    def __init__(self, in_channels, hidden, num_classes, layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden))
        for _ in range(layers - 2):
            self.convs.append(GCNConv(hidden, hidden))
        self.convs.append(GCNConv(hidden, num_classes))

    def forward(self, data):
        x = data.x
        for conv in self.convs[:-1]:
            x = conv(x, data.edge_index).relu()
        return self.convs[-1](x, data.edge_index)


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden, num_classes, layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden))
        for _ in range(layers - 2):
            self.convs.append(SAGEConv(hidden, hidden))
        self.convs.append(SAGEConv(hidden, num_classes))

    def forward(self, data):
        x = data.x
        for conv in self.convs[:-1]:
            x = conv(x, data.edge_index).relu()
        return self.convs[-1](x, data.edge_index)


class GAT(nn.Module):
    def __init__(self, in_channels, hidden, num_classes, heads=8, layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden, heads=heads))
        for _ in range(layers - 2):
            self.convs.append(GATConv(hidden * heads, hidden, heads=heads))
        self.convs.append(GATConv(hidden * heads, num_classes,
                                  heads=1, concat=False))

    def forward(self, data):
        x = data.x
        for conv in self.convs[:-1]:
            x = conv(x, data.edge_index).relu()
        return self.convs[-1](x, data.edge_index)
