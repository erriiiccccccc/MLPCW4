"""Train a learned weighted fusion over layers 8-11."""
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

PROBE_DIR = "/home/s2411221/probe_results"
LAYERS    = [8, 9, 10, 11]
EPOCHS    = 50
BATCH     = 64
LR        = 1e-3

print("Loading embeddings...")
train_bits, test_bits = [], []
for l in LAYERS:
    d = f"{PROBE_DIR}/layer_{l:02d}"
    train_bits.append(np.load(f"{d}/embeddings.npy"))
    test_bits.append(np.load(f"{d}/test_embeddings.npy"))

train_labels = np.load(f"{PROBE_DIR}/layer_11/labels.npy")
test_labels  = np.load(f"{PROBE_DIR}/layer_11/test_labels.npy")
print(f"  Train: {train_bits[0].shape[0]} samples | Test: {test_bits[0].shape[0]} samples")

print("Scaling embeddings...")
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
print(f"  Classes: {num_classes}")

class LearnedWeightedModel(nn.Module):
    def __init__(self, num_layers, embed_dim, num_classes):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        weights = torch.softmax(self.layer_logits, dim=0)
        fused = (x * weights.view(1, -1, 1)).sum(dim=1)
        return self.head(fused)

    def get_weights(self):
        return torch.softmax(self.layer_logits, dim=0).detach().cpu().numpy()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = LearnedWeightedModel(len(LAYERS), 768, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True
)

print(f"\nTraining for {EPOCHS} epochs...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct += (logits.argmax(1) == yb).sum().item()
        total += len(yb)
    scheduler.step()

    if epoch % 5 == 0 or epoch == 1:
        ws = model.get_weights()
        w_line = "  ".join(f"L{l}:{ws[i]:.3f}" for i, l in enumerate(LAYERS))
        print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={total_loss/total:.4f}  "
              f"train_acc={correct/total:.4f}  weights=[{w_line}]")

model.eval()
with torch.no_grad():
    all_correct, all_total = 0, 0
    for xb, yb in DataLoader(TensorDataset(X_test, y_test), batch_size=256):
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb).argmax(1)
        all_correct += (preds == yb).sum().item()
        all_total += len(yb)
test_acc = all_correct / all_total

final_ws = model.get_weights()

with open(f"{PROBE_DIR}/layer_11/probe_accuracy.json") as f:
    base_acc = json.load(f)["test_acc"]

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print("\nLearned layer weights (softmaxed):")
for i, l in enumerate(LAYERS):
    print(f"  Layer {l}: {final_ws[i]:.4f}")
print(f"\nTest accuracy (learned weighting): {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Test accuracy (layer 11 baseline): {base_acc:.4f} ({base_acc*100:.2f}%)")
delta = test_acc - base_acc
print(f"Delta vs baseline: {delta:+.4f} ({delta*100:+.2f}%)")
print("="*60)

dump = {
    "test_acc": test_acc,
    "baseline_accuracy": base_acc,
    "delta": delta,
    "learned_weights": {f"layer_{l}": float(final_ws[i]) for i, l in enumerate(LAYERS)},
    "layers_used": LAYERS,
    "epochs": EPOCHS,
}
out_path = f"{PROBE_DIR}/summary/learned_weighted_results.json"
with open(out_path, "w") as f:
    json.dump(dump, f, indent=2)
print(f"\nSaved to {out_path}")
