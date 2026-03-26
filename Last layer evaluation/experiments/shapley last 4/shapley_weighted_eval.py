"""Evaluate a Shapley-weighted sum of layers 8-11."""

import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

PROBE_DIR  = "/home/s2411221/probe_results"
LAYERS     = [8, 9, 10, 11]
SHAPLEY    = {8: 0.014362, 9: 0.041173, 10: 0.106244, 11: 0.275068}

total = sum(SHAPLEY[layer] for layer in LAYERS)
weights = {layer: SHAPLEY[layer] / total for layer in LAYERS}
print("Normalized Shapley weights:")
for layer in LAYERS:
    print(f"  Layer {layer:2d}: {weights[layer]:.4f}")

print("\nLoading embeddings...")
train_combined = None
test_combined = None

for layer in LAYERS:
    layer_dir = os.path.join(PROBE_DIR, f"layer_{layer:02d}")
    train_emb = np.load(os.path.join(layer_dir, "embeddings.npy"))
    test_emb = np.load(os.path.join(layer_dir, "test_embeddings.npy"))
    print(f"  Layer {layer}: train {train_emb.shape}, test {test_emb.shape}")

    if train_combined is None:
        train_combined = weights[layer] * train_emb
        test_combined = weights[layer] * test_emb
    else:
        train_combined = train_combined + weights[layer] * train_emb
        test_combined = test_combined + weights[layer] * test_emb

print(f"\nCombined embedding shape: train {train_combined.shape}, test {test_combined.shape}")

train_labels = np.load(os.path.join(PROBE_DIR, "layer_11", "labels.npy"))
test_labels = np.load(os.path.join(PROBE_DIR, "layer_11", "test_labels.npy"))

scaler = StandardScaler()
train_combined = scaler.fit_transform(train_combined)
test_combined = scaler.transform(test_combined)

print("\nTraining linear probe on weighted embeddings...")
clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', n_jobs=-1)
clf.fit(train_combined, train_labels)

preds = clf.predict(test_combined)
acc = accuracy_score(test_labels, preds)
print(f"\n{'=' * 50}")
print(f"  Shapley-weighted fusion accuracy: {acc:.4f} ({acc * 100:.2f}%)")
print(f"{'=' * 50}")

print("\nBaseline (layer 11 only):")
with open(os.path.join(PROBE_DIR, "layer_11", "probe_accuracy.json")) as f:
    baseline = json.load(f)
print(f"  Layer 11 probe accuracy: {baseline['test_acc']:.4f} ({baseline['test_acc'] * 100:.2f}%)")

print(f"\nDelta: {acc - baseline['test_acc']:+.4f}")
