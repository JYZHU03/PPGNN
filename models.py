# ---------------- models.py ----------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from lvconv import LVConv
from torch_geometric.nn import GCNConv, GATConv, SAGEConv


class PPGNN(nn.Module):
    r"""Predator–Prey GNN."""
    def __init__(self, in_channels, hidden, num_classes,
                 layers=5, dt=0.05, dropout=0.2):
        super().__init__()
        self.hidden = hidden
        self.dropout = dropout

        # 两套 lift 权重
        self.lift_x = nn.Linear(in_channels, hidden)
        self.lift_y = nn.Linear(in_channels, hidden)

        self.layers = nn.ModuleList(
            LVConv(hidden, dt=dt) for _ in range(layers)
        )

        # 读出拼接 [X | Y]
        self.readout = nn.Linear(2 * hidden, num_classes)

    def forward(self, data):
        X0 = torch.tanh(self.lift_x(data.x))
        Y0 = torch.tanh(self.lift_y(data.x))
        h = torch.cat([X0, Y0], dim=-1)

        for conv in self.layers:
            h = F.dropout(h, p=self.dropout,
                          training=self.training)
            h = conv(h, data.edge_index)

        return self.readout(h)


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
    def __init__(self, in_channels, hidden, num_classes,
                 heads=8, layers=2):
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
