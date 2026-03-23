"""
Hyperparameter search for learned weighted eval.
Searches over LR, weight decay, and batch size.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import os

PROBE_DIR = "/home/s2411221/probe_results"
LAYERS    = [8, 9, 10, 11]
EPOCHS    = 100   # fixed — more epochs for a fair chance

# ── Load & scale embeddings (per-layer, same as original) ─────────────────────
print("Loading embeddings...")
train_embs, test_embs = [], []
for l in LAYERS:
    d = os.path.join(PROBE_DIR, f"layer_{l:02d}")
    train_embs.append(np.load(os.path.join(d, "embeddings.npy")))
    test_embs.append(np.load(os.path.join(d, "test_embeddings.npy")))

train_labels = np.load(os.path.join(PROBE_DIR, "layer_11", "labels.npy"))
test_labels  = np.load(os.path.join(PROBE_DIR, "layer_11", "test_labels.npy"))

scaled_train, scaled_test = [], []
for t, e in zip(train_embs, test_embs):
    sc = StandardScaler()
    scaled_train.append(sc.fit_transform(t).astype(np.float32))
    scaled_test.append(sc.transform(e).astype(np.float32))

X_train = torch.tensor(np.stack(scaled_train, axis=1))   # (N, 4, 768)
X_test  = torch.tensor(np.stack(scaled_test,  axis=1))   # (M, 4, 768)
y_train = torch.tensor(train_labels, dtype=torch.long)
y_test  = torch.tensor(test_labels,  dtype=torch.long)
num_classes = int(y_train.max().item()) + 1

print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]} | Classes: {num_classes}")

# ── Model ─────────────────────────────────────────────────────────────────────
class LearnedWeightedModel(nn.Module):
    def __init__(self, num_layers, embed_dim, num_classes):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        weights = torch.softmax(self.layer_logits, dim=0)
        fused   = (x * weights.view(1, -1, 1)).sum(dim=1)
        return self.head(fused)

    def get_weights(self):
        return torch.softmax(self.layer_logits, dim=0).detach().cpu().numpy()

# ── Grid ──────────────────────────────────────────────────────────────────────
LR_VALUES           = [1e-4, 5e-4, 1e-3, 5e-3]
WEIGHT_DECAY_VALUES = [0.0, 1e-4, 1e-3]
BATCH_VALUES        = [64, 256]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

test_ds     = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_ds, batch_size=512)

best_acc    = 0
best_config = None
all_results = []

total_configs = len(LR_VALUES) * len(WEIGHT_DECAY_VALUES) * len(BATCH_VALUES)
run = 0

for lr in LR_VALUES:
    for wd in WEIGHT_DECAY_VALUES:
        for batch in BATCH_VALUES:
            run += 1
            print(f"[{run}/{total_configs}] lr={lr}  wd={wd}  batch={batch}", flush=True)

            model     = LearnedWeightedModel(len(LAYERS), 768, num_classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
            criterion = nn.CrossEntropyLoss()

            train_loader = DataLoader(
                TensorDataset(X_train, y_train),
                batch_size=batch, shuffle=True, num_workers=4, pin_memory=True
            )

            for epoch in range(1, EPOCHS + 1):
                model.train()
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()
                scheduler.step()

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    correct += (model(xb).argmax(1) == yb).sum().item()
                    total   += len(yb)
            acc = correct / total

            w = model.get_weights()
            w_str = "  ".join(f"L{l}:{w[i]:.3f}" for i, l in enumerate(LAYERS))
            marker = " <-- best" if acc > best_acc else ""
            print(f"  acc={acc*100:.2f}%  weights=[{w_str}]{marker}", flush=True)

            if acc > best_acc:
                best_acc    = acc
                best_config = {"lr": lr, "weight_decay": wd, "batch": batch}

            all_results.append({
                "lr": lr, "weight_decay": wd, "batch": batch,
                "test_acc": acc,
                "learned_weights": {f"layer_{l}": float(w[i]) for i, l in enumerate(LAYERS)}
            })

print(f"\nBEST: lr={best_config['lr']}, wd={best_config['weight_decay']}, "
      f"batch={best_config['batch']}  ->  {best_acc*100:.2f}%")

out_dir = os.path.join(PROBE_DIR, "summary", "hyperparam_search")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "learned_search_results.json")
with open(out_path, "w") as f:
    json.dump({"best": {**best_config, "test_acc": best_acc}, "all": all_results}, f, indent=2)
print(f"Saved to {out_path}")
