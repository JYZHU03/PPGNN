# ---------------- utils.py ----------------
import torch
import numpy as np
import random

def set_seed(seed: int = 42):
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
    """Mean absolute error used for regression benchmarks."""
    return (pred - y).abs().mean().item()

@torch.no_grad()
def r2(pred, y):
    """Coefficient of determination (R^2)."""
    y = y.view_as(pred)
    # Total sum of squares and residual sum of squares
    ss_tot = ((y - y.mean()) ** 2).sum()
    ss_res = ((y - pred) ** 2).sum()
    # Avoid division by zero for constant targets
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot).item()
