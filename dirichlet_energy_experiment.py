import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
import matplotlib.pyplot as plt

from models import PPGNN


def grid_edge_index(width: int = 10, height: int = 10) -> torch.Tensor:
    """Create edge_index for a 2D grid with 4-neighbor connectivity."""
    edges = []
    for i in range(height):
        for j in range(width):
            idx = i * width + j
            if i < height - 1:
                edges.append((idx, (i + 1) * width + j))
            if j < width - 1:
                edges.append((idx, i * width + (j + 1)))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return edge_index


# def dirichlet_energy(x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> float:
#     row, col = edge_index
#     diff = x[row] - x[col]
#     return diff.pow(2).sum(dim=1).sum().item() / num_nodes

def dirichlet_energy(x, edge_index, num_nodes, undirected=True):
    row, col = edge_index
    if undirected:
        # 只保留 i<j 的边，避免双计数
        mask = row < col
        row, col = row[mask], col[mask]
    diff = x[row] - x[col]
    # 论文里常用 1/|V| 归一化，这里保持一致
    return (diff.pow(2).sum(dim=1).sum() / num_nodes).item()


def run_gcn(data: Data, channels: int, layers: int) -> list:
    x = data.x
    energies = [dirichlet_energy(x, data.edge_index, data.num_nodes)]
    convs = nn.ModuleList([GCNConv(channels, channels) for _ in range(layers)])
    for conv in convs:
        x = conv(x, data.edge_index).relu()
        energies.append(dirichlet_energy(x, data.edge_index, data.num_nodes))
    return energies


def run_gat(data: Data, channels: int, layers: int) -> list:
    x = data.x
    energies = [dirichlet_energy(x, data.edge_index, data.num_nodes)]
    convs = nn.ModuleList([
        GATConv(channels, channels, heads=1, concat=False) for _ in range(layers)
    ])
    for conv in convs:
        x = conv(x, data.edge_index).relu()
        energies.append(dirichlet_energy(x, data.edge_index, data.num_nodes))
    return energies


def run_ppgnn(data: Data, channels: int, layers: int) -> list:
    model = PPGNN(
        in_channels=channels,
        hidden=channels,
        num_classes=channels,
        layers=layers,
        dropout=0.0,
    )
    model.eval()
    x = data.x
    X0 = torch.tanh(model.lift_x(x.float()))
    if model.y0_mode == "learned":
        Y0 = torch.tanh(model.lift_y(x.float()))
    else:
        Y0 = torch.ones_like(X0)
    h = torch.cat([X0, Y0], dim=-1)
    energies = [dirichlet_energy(X0, data.edge_index, data.num_nodes)]
    for conv, tau_param in zip(model.layers, model.taus):
        h_hat = conv(h, data.edge_index)
        tau = torch.sigmoid(tau_param)
        h = (1 - tau) * h + tau * h_hat
        X, _ = torch.split(h, model.hidden, dim=-1)
        energies.append(dirichlet_energy(X, data.edge_index, data.num_nodes))
    return energies


def main():
    torch.manual_seed(0)
    size = 10
    channels = 16
    layers = 100

    edge_index = grid_edge_index(size, size)
    x = torch.rand(size * size, channels)
    data = Data(x=x, edge_index=edge_index)

    energies = {
        "GCN": run_gcn(data.clone(), channels, layers),
        "GAT": run_gat(data.clone(), channels, layers),
        "PPGNN": run_ppgnn(data.clone(), channels, layers),
    }

    # for name, vals in energies.items():
    #     plt.plot(range(len(vals)), vals, label=name)
    # plt.xlabel("Layer")
    # plt.ylabel("Dirichlet Energy E(X^n)")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    # plt.savefig("dirichlet_energy.png")

    for name, vals in energies.items():
        plt.plot(range(len(vals)), vals, label=name)
    plt.yscale('log');
    plt.xscale('log')  # 或至少 y 轴 log
    plt.xlabel("Layer n");
    plt.ylabel("Dirichlet energy $E(\mathbf{X}^n)$")
    plt.legend();
    plt.tight_layout()
    plt.savefig("dirichlet_energy.png")  # 先保存，再 show
    plt.show()  # 需要交互时再开


if __name__ == "__main__":
    main()
