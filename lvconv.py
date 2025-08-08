# ---------------- lvconv.py ----------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree


class LVConv(MessagePassing):
    r"""
    Lotka–Volterra graph convolution (Predator–Prey GNN layer).

    * 两通道 X、Y；反应系数逐通道 Hadamard(α,β,γ,δ)，扩散系数 Dx, Dy 为标量。
    * 半隐式扩散：Jacobi 迭代近似 (I + dt * D · L)^{-1} · RHS（1~3 步即可）。
    * 归一化：'sym' = D^{-1/2} A D^{-1/2}（默认）；'rw' = D^{-1}A（可切换）。
    * 正参数用 softplus 约束。
    """
    def __init__(
        self,
        channels: int,
        dt: float = 0.1,
        norm_type: str = "sym",         # 'sym' or 'rw'
        jacobi_steps: int = 2,          # 1~3，2/3 更稳
        eps: float = 1e-3,
    ):
        super().__init__(aggr="add")
        assert norm_type in {"sym", "rw"}
        assert jacobi_steps >= 1

        self.d = channels
        self.dt = float(dt)
        self.norm_type = norm_type
        self.jacobi_steps = int(jacobi_steps)
        self.eps = float(eps)

        # 反应系数（逐通道）
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta  = nn.Parameter(torch.zeros(channels))
        self.gamma = nn.Parameter(torch.zeros(channels))
        self.delta = nn.Parameter(torch.zeros(channels))

        # 扩散（标量）
        self.Dx = nn.Parameter(torch.tensor(0.0))
        self.Dy = nn.Parameter(torch.tensor(0.0))

        # 温和初始化（softplus^-1 设定目标值）
        # 目标：alpha≈gamma≈0.2, beta≈delta≈0.1, Dx≈Dy≈0.8
        def softplus_inv(y: float) -> float:
            return math.log(math.exp(y) - 1.0)

        with torch.no_grad():
            a0 = softplus_inv(0.2)
            b0 = softplus_inv(0.1)
            d0 = softplus_inv(0.8)
            self.alpha.fill_(a0)
            self.gamma.fill_(a0)
            self.beta.fill_(b0)
            self.delta.fill_(b0)
            self.Dx.fill_(d0)
            self.Dy.fill_(d0)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor):
        """
        h : [N, 2d]  (concat of X | Y)
        edge_index : [2, E]（无向图请用双向边；无需显式 self-loop）
        """
        N = h.size(0)
        X, Y = torch.split(h, self.d, dim=-1)

        # 归一化邻接 S
        row, col = edge_index  # row=j(src), col=i(dst)
        if self.norm_type == "sym":
            # S_sym = D^{-1/2} A D^{-1/2}
            deg = degree(row, N, dtype=X.dtype).clamp(min=1)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        else:
            # S_rw = D^{-1} A  （按 target 度归一化，使每行和为1）
            deg = degree(col, N, dtype=X.dtype).clamp(min=1)
            deg_inv = 1.0 / deg
            norm = deg_inv[col]

        # 正参数
        a = F.softplus(self.alpha) + self.eps
        b = F.softplus(self.beta)  + self.eps
        g = F.softplus(self.gamma) + self.eps
        d = F.softplus(self.delta) + self.eps
        Dx = F.softplus(self.Dx)   + self.eps
        Dy = F.softplus(self.Dy)   + self.eps

        # 反应（显式）
        RX = a * X - b * (X * Y)
        RY = d * (X * Y) - g * Y
        RHS_X = X + self.dt * RX
        RHS_Y = Y + self.dt * RY

        # Jacobi 近似的隐式扩散：X^{k+1} = (RHS + a*S*X^k)/(1+a)
        ax = self.dt * Dx
        ay = self.dt * Dy
        denom_x = (1.0 + ax)
        denom_y = (1.0 + ay)

        Xk, Yk = X, Y
        for _ in range(self.jacobi_steps):
            SXk = self.propagate(edge_index, x=Xk, norm=norm)
            SYk = self.propagate(edge_index, x=Yk, norm=norm)
            Xk = (RHS_X + ax * SXk) / denom_x
            Yk = (RHS_Y + ay * SYk) / denom_y

        return torch.cat([Xk, Yk], dim=-1)

    def message(self, x_j: torch.Tensor, norm: torch.Tensor):
        return norm.view(-1, 1) * x_j
