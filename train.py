# ---------------- train.py ----------------
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from utils import set_seed, accuracy
from models import PPGNN, GCN, GraphSAGE, GAT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42)

# ---------- 数据集 ----------
data = Planetoid(root="data/Planetoid",
                 name="Cora",
                 transform=NormalizeFeatures())[0].to(DEVICE)


# ---------- 选择模型 ----------
def get_model(name):
    params = dict(in_channels=data.num_features,
                  hidden=64, num_classes=7)
    if name == "ppgnn":
        return PPGNN(**params, layers=15, dt=0.1, dropout=0.4)
    if name == "gcn":
        return GCN(**params, layers=2)
    if name == "sage":
        return GraphSAGE(**params, layers=2)
    if name == "gat":
        return GAT(**params, heads=8, layers=2)
    raise ValueError(name)


for name in ["ppgnn", "gcn"]:
    print(f"\n=== {name.upper()} ===")
    model = get_model(name).to(DEVICE)

    optim = torch.optim.Adam(model.parameters(),
                             lr=3e-3 if name == "ppgnn" else 1e-2,
                             weight_decay=5e-4)
    # 可选梯度裁剪阈值
    clip_value = 5.0 if name == "ppgnn" else None

    best_val = best_test = 0.0
    for epoch in range(1, 301):
        model.train()
        optim.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out[data.train_mask],
                               data.y[data.train_mask])
        loss.backward()

        if clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           clip_value)
        optim.step()

        model.eval()
        with torch.no_grad():
            logits = model(data)
            val_acc = accuracy(logits[data.val_mask],
                               data.y[data.val_mask])
            test_acc = accuracy(logits[data.test_mask],
                                data.y[data.test_mask])

        if val_acc > best_val:
            best_val, best_test = val_acc, test_acc

        if epoch == 1 or epoch % 20 == 0:
            print(f"E{epoch:03d}  loss:{loss:.3f}  "
                  f"val:{val_acc:.3f}  test:{test_acc:.3f}")

    print(f"★ Best val={best_val:.3f}  corresponding test={best_test:.3f}")
