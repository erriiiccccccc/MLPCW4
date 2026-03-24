"""
Shapley-weighted embedding evaluation.

Takes the last 4 layers' embeddings, weights them by their normalized Shapley
values, sums to a single 768-dim vector, trains a linear probe, and evaluates
on the test set.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os

# ── config ────────────────────────────────────────────────────────────────────
PROBE_DIR  = "/home/s2411221/probe_results"
LAYERS     = [8, 9, 10, 11]
SHAPLEY    = {8: 0.014362, 9: 0.041173, 10: 0.106244, 11: 0.275068}
# ──────────────────────────────────────────────────────────────────────────────

# Normalize Shapley values so they sum to 1
total = sum(SHAPLEY[l] for l in LAYERS)
weights = {l: SHAPLEY[l] / total for l in LAYERS}
print("Normalized Shapley weights:")
for l in LAYERS:
    print(f"  Layer {l:2d}: {weights[l]:.4f}")

# Load and combine embeddings
print("\nLoading embeddings...")
train_combined = None
test_combined  = None

for l in LAYERS:
    layer_dir = os.path.join(PROBE_DIR, f"layer_{l:02d}")
    train_emb = np.load(os.path.join(layer_dir, "embeddings.npy"))      # (N, 768)
    test_emb  = np.load(os.path.join(layer_dir, "test_embeddings.npy")) # (M, 768)
    print(f"  Layer {l}: train {train_emb.shape}, test {test_emb.shape}")

    train_combined = weights[l] * train_emb if train_combined is None else train_combined + weights[l] * train_emb
    test_combined  = weights[l] * test_emb  if test_combined  is None else test_combined  + weights[l] * test_emb

print(f"\nCombined embedding shape: train {train_combined.shape}, test {test_combined.shape}")

# Load labels (same across layers)
train_labels = np.load(os.path.join(PROBE_DIR, "layer_11", "labels.npy"))
test_labels  = np.load(os.path.join(PROBE_DIR, "layer_11", "test_labels.npy"))

# Scale
scaler = StandardScaler()
train_combined = scaler.fit_transform(train_combined)
test_combined  = scaler.transform(test_combined)

# Train linear probe
print("\nTraining linear probe on weighted embeddings...")
clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', n_jobs=-1)
clf.fit(train_combined, train_labels)

# Evaluate
preds = clf.predict(test_combined)
acc = accuracy_score(test_labels, preds)
print(f"\n{'='*50}")
print(f"  Shapley-weighted fusion accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(f"{'='*50}")

# Compare to layer 11 alone
print("\nBaseline (layer 11 only):")
import json
with open(os.path.join(PROBE_DIR, "layer_11", "probe_accuracy.json")) as f:
    baseline = json.load(f)
print(f"  Layer 11 probe accuracy: {baseline['test_acc']:.4f} ({baseline['test_acc']*100:.2f}%)")

print(f"\nDelta: {acc - baseline['test_acc']:+.4f}")
"""
Shapley-weighted embedding evaluation.

Takes the last 4 layers' embeddings, weights them by their normalized Shapley
values, sums to a single 768-dim vector, trains a linear probe, and evaluates
on the test set.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os

# ── config ────────────────────────────────────────────────────────────────────
PROBE_DIR  = "/home/s2411221/probe_results"
LAYERS     = [8, 9, 10, 11]
SHAPLEY    = {8: 0.014362, 9: 0.041173, 10: 0.106244, 11: 0.275068}
# ──────────────────────────────────────────────────────────────────────────────

# Normalize Shapley values so they sum to 1
total = sum(SHAPLEY[l] for l in LAYERS)
weights = {l: SHAPLEY[l] / total for l in LAYERS}
print("Normalized Shapley weights:")
for l in LAYERS:
    print(f"  Layer {l:2d}: {weights[l]:.4f}")

# Load and combine embeddings
print("\nLoading embeddings...")
train_combined = None
test_combined  = None

for l in LAYERS:
    layer_dir = os.path.join(PROBE_DIR, f"layer_{l:02d}")
    train_emb = np.load(os.path.join(layer_dir, "embeddings.npy"))      # (N, 768)
    test_emb  = np.load(os.path.join(layer_dir, "test_embeddings.npy")) # (M, 768)
    print(f"  Layer {l}: train {train_emb.shape}, test {test_emb.shape}")

    train_combined = weights[l] * train_emb if train_combined is None else train_combined + weights[l] * train_emb
    test_combined  = weights[l] * test_emb  if test_combined  is None else test_combined  + weights[l] * test_emb

print(f"\nCombined embedding shape: train {train_combined.shape}, test {test_combined.shape}")

# Load labels (same across layers)
train_labels = np.load(os.path.join(PROBE_DIR, "layer_11", "labels.npy"))
test_labels  = np.load(os.path.join(PROBE_DIR, "layer_11", "test_labels.npy"))

# Scale
scaler = StandardScaler()
train_combined = scaler.fit_transform(train_combined)
test_combined  = scaler.transform(test_combined)

# Train linear probe
print("\nTraining linear probe on weighted embeddings...")
clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', n_jobs=-1)
clf.fit(train_combined, train_labels)

# Evaluate
preds = clf.predict(test_combined)
acc = accuracy_score(test_labels, preds)
print(f"\n{'='*50}")
print(f"  Shapley-weighted fusion accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(f"{'='*50}")

# Compare to layer 11 alone
print("\nBaseline (layer 11 only):")
import json
with open(os.path.join(PROBE_DIR, "layer_11", "probe_accuracy.json")) as f:
    baseline = json.load(f)
print(f"  Layer 11 probe accuracy: {baseline['test_acc']:.4f} ({baseline['test_acc']*100:.2f}%)")

print(f"\nDelta: {acc - baseline['test_acc']:+.4f}")
