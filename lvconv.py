# ---------------- lvconv.py ----------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree


class LVConv(MessagePassing):
    r"""
    Lotka–Volterra graph convolution (Predator–Prey GNN layer).

    关键特征
    ----------
    * 两通道 X、Y；Hadamard 反应系数 (4d) + 两个扩散标量 Dx, Dy。
    * 半隐式扩散（一次 Jacobi 近似），大步长仍稳定。
    * Softplus 约束所有正参数；可选梯度/幅值裁剪。
    """
    def __init__(self, channels: int, dt: float = 0.05):
        super().__init__(aggr="add")
        self.d = channels
        self.dt = dt

        # --- Reaction rates（Hadamard 向量，正值）---
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta  = nn.Parameter(torch.zeros(channels))
        self.gamma = nn.Parameter(torch.zeros(channels))
        self.delta = nn.Parameter(torch.zeros(channels))

        # --- Diffusivities（标量，正值）---
        self.Dx = nn.Parameter(torch.tensor(-3.0))   # softplus≈0.05
        self.Dy = nn.Parameter(torch.tensor(+1.5))   # softplus≈4.7

        # LayerNorm 有助于深层稳定
        self.norm = nn.LayerNorm(2 * channels)

    # ------------------------------------------------------------------
    def forward(self, h, edge_index):
        """
        h : Tensor [N, 2d]    (concat of X | Y)
        edge_index : [2, E]   (undirected, 无需显式 self-loop)
        """
        X, Y = torch.split(h, self.d, dim=-1)

        # -------- 规格化邻接  (random–walk) ----------
        row, col = edge_index                  # 注意： row = src
        deg = degree(row, X.size(0), dtype=X.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]   # Ẑ_ij

        AX = self.propagate(edge_index, x=X, norm=norm)
        AY = self.propagate(edge_index, x=Y, norm=norm)

        Lx = X - AX                            # L̃ X
        Ly = Y - AY                            # L̃ Y

        # -------- Softplus 取正 --------
        α = F.softplus(self.alpha) + 1e-3
        β = F.softplus(self.beta)  + 1e-3
        γ = F.softplus(self.gamma) + 1e-3
        δ = F.softplus(self.delta) + 1e-3
        Dx = F.softplus(self.Dx)    + 1e-3
        Dy = F.softplus(self.Dy)    + 1e-3

        # -------- 半隐式扩散（一次 Jacobi） ----------
        X_mid = X - self.dt * Dx * Lx          # (I + dt Dx L̃)^-1 ≈ I - dt Dx L̃
        Y_mid = Y - self.dt * Dy * Ly

        # -------- 反应项 ----------
        RX = α * X_mid - β * (X_mid * Y_mid)
        RY = δ * (X_mid * Y_mid) - γ * Y_mid

        Xn = X_mid + self.dt * RX
        Yn = Y_mid + self.dt * RY

        # -------- 安全裁剪 + LayerNorm ----------
        h_next = torch.cat([Xn, Yn], dim=-1).clamp(-10.0, 10.0)
        return self.norm(h_next)

    # PyG 消息函数
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
