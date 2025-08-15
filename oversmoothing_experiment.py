import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
import matplotlib.pyplot as plt

from models import PPGNN


def grid_edge_index(width: int = 20, height: int = 20) -> torch.Tensor:
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
    # Make undirected
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return edge_index


def propagate_gcn(x: torch.Tensor, edge_index: torch.Tensor, layers: int) -> torch.Tensor:
    """Repeatedly apply a GCN layer with identity weights."""
    conv = GCNConv(1, 1, bias=False)
    with torch.no_grad():
        conv.lin.weight.data = torch.eye(1)
    for _ in range(layers):
        x = conv(x, edge_index)
    return x


def propagate_gat(x: torch.Tensor, edge_index: torch.Tensor, layers: int) -> torch.Tensor:
    """Repeatedly apply a GAT layer with identity weights and uniform attention."""
    conv = GATConv(1, 1, heads=1, concat=False, bias=False)
    with torch.no_grad():
        if hasattr(conv, "lin_src"):
            conv.lin_src.weight.data = torch.eye(1)
            conv.lin_dst.weight.data = torch.eye(1)
        else:  # fallback for older PyG versions
            conv.lin.weight.data = torch.eye(1)
        if hasattr(conv, "att_src"):
            conv.att_src.data.fill_(0.0)
            conv.att_dst.data.fill_(0.0)
    for _ in range(layers):
        x = conv(x, edge_index)
    return x


def propagate_ppgnn(x: torch.Tensor, edge_index: torch.Tensor, layers: int) -> torch.Tensor:
    """Run PPGNN for a given number of layers and return node features."""
    model = PPGNN(
        in_channels=1,
        hidden=1,
        num_classes=1,
        layers=layers,
        dropout=0.0,
    )
    model.eval()
    X0 = torch.tanh(model.lift_x(x.float()))
    if model.y0_mode == "learned":
        Y0 = torch.tanh(model.lift_y(x.float()))
    else:
        Y0 = torch.ones_like(X0)
    h = torch.cat([X0, Y0], dim=-1)
    for conv, tau_param in zip(model.layers, model.taus):
        h_hat = conv(h, edge_index)
        tau = torch.sigmoid(tau_param)
        h = (1 - tau) * h + tau * h_hat
    X, _ = torch.split(h, model.hidden, dim=-1)
    return X


def main():
    torch.manual_seed(0)
    size = 20
    layers = 100

    edge_index = grid_edge_index(size, size)
    x = torch.rand(size * size, 1)
    data = Data(x=x, edge_index=edge_index)

    x_gcn = propagate_gcn(data.x.clone(), data.edge_index, layers)
    x_gat = propagate_gat(data.x.clone(), data.edge_index, layers)
    x_ppgnn = propagate_ppgnn(data.x.clone(), data.edge_index, layers)

    mats = {
        "Initial": data.x.view(size, size),
        "GCN (100 layers)": x_gcn.view(size, size),
        "GAT (100 layers)": x_gat.view(size, size),
        "PPGNN (100 layers)": x_ppgnn.view(size, size),
    }

    vmin = min(mat.min().item() for mat in mats.values())
    vmax = max(mat.max().item() for mat in mats.values())

    plt.style.use("seaborn-v0_8")
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), dpi=300)
    for ax, (title, mat) in zip(axes, mats.items()):
        im = ax.imshow(mat, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    plt.tight_layout()
    plt.savefig("oversmoothing_heatmaps.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
