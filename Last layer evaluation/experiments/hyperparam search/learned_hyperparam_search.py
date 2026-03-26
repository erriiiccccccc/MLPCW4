"""Search optimizer settings for the learned weighted fusion model."""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import os

PROBE_DIR = "/home/s2411221/probe_results"
LAYERS    = [8, 9, 10, 11]
EPOCHS    = 100

print("Loading embeddings...")
train_bits, test_bits = [], []
for l in LAYERS:
    d = os.path.join(PROBE_DIR, f"layer_{l:02d}")
    train_bits.append(np.load(os.path.join(d, "embeddings.npy")))
    test_bits.append(np.load(os.path.join(d, "test_embeddings.npy")))

train_labels = np.load(os.path.join(PROBE_DIR, "layer_11", "labels.npy"))
test_labels  = np.load(os.path.join(PROBE_DIR, "layer_11", "test_labels.npy"))

train_scaled, test_scaled = [], []
for t, e in zip(train_bits, test_bits):
    sc = StandardScaler()
    train_scaled.append(sc.fit_transform(t).astype(np.float32))
    test_scaled.append(sc.transform(e).astype(np.float32))

X_train = torch.tensor(np.stack(train_scaled, axis=1))
X_test  = torch.tensor(np.stack(test_scaled,  axis=1))
y_train = torch.tensor(train_labels, dtype=torch.long)
y_test  = torch.tensor(test_labels,  dtype=torch.long)
num_classes = int(y_train.max().item()) + 1

print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]} | Classes: {num_classes}")

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

LR_VALUES           = [1e-4, 5e-4, 1e-3, 5e-3]
WEIGHT_DECAY_VALUES = [0.0, 1e-4, 1e-3]
BATCH_VALUES        = [64, 256]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

test_ds     = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_ds, batch_size=512)

best_acc = 0
best_cfg = None
runs = []

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
            ok, n = 0, 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    ok += (model(xb).argmax(1) == yb).sum().item()
                    n  += len(yb)
            acc = ok / n

            ws = model.get_weights()
            w_line = "  ".join(f"L{l}:{ws[i]:.3f}" for i, l in enumerate(LAYERS))
            marker = " <-- best" if acc > best_acc else ""
            print(f"  acc={acc*100:.2f}%  weights=[{w_line}]{marker}", flush=True)

            if acc > best_acc:
                best_acc = acc
                best_cfg = {"lr": lr, "weight_decay": wd, "batch": batch}

            runs.append({
                "lr": lr, "weight_decay": wd, "batch": batch,
                "test_acc": acc,
                "learned_weights": {f"layer_{l}": float(ws[i]) for i, l in enumerate(LAYERS)}
            })

print(f"\nBEST: lr={best_cfg['lr']}, wd={best_cfg['weight_decay']}, "
      f"batch={best_cfg['batch']}  ->  {best_acc*100:.2f}%")

out_dir = os.path.join(PROBE_DIR, "summary", "hyperparam_search")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "learned_search_results.json")
with open(out_path, "w") as f:
    json.dump({"best": {**best_cfg, "test_acc": best_acc}, "all": runs}, f, indent=2)
print(f"Saved to {out_path}")
