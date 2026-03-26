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

tot = sum(SHAPLEY[layer] for layer in LAYERS)
wts = {layer: SHAPLEY[layer] / tot for layer in LAYERS}
print("Normalized Shapley weights:")
for layer in LAYERS:
    print(f"  Layer {layer:2d}: {wts[layer]:.4f}")

print("\nLoading embeddings...")
train_mix = None
test_mix = None

for layer in LAYERS:
    d = os.path.join(PROBE_DIR, f"layer_{layer:02d}")
    train_emb = np.load(os.path.join(d, "embeddings.npy"))
    test_emb = np.load(os.path.join(d, "test_embeddings.npy"))
    print(f"  Layer {layer}: train {train_emb.shape}, test {test_emb.shape}")

    if train_mix is None:
        train_mix = wts[layer] * train_emb
        test_mix = wts[layer] * test_emb
    else:
        train_mix = train_mix + wts[layer] * train_emb
        test_mix = test_mix + wts[layer] * test_emb

print(f"\nCombined embedding shape: train {train_mix.shape}, test {test_mix.shape}")

train_labels = np.load(os.path.join(PROBE_DIR, "layer_11", "labels.npy"))
test_labels = np.load(os.path.join(PROBE_DIR, "layer_11", "test_labels.npy"))

scaler = StandardScaler()
train_mix = scaler.fit_transform(train_mix)
test_mix = scaler.transform(test_mix)

print("\nTraining linear probe on weighted embeddings...")
clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', n_jobs=-1)
clf.fit(train_mix, train_labels)

preds = clf.predict(test_mix)
acc = accuracy_score(test_labels, preds)
print(f"\n{'=' * 50}")
print(f"  Shapley-weighted fusion accuracy: {acc:.4f} ({acc * 100:.2f}%)")
print(f"{'=' * 50}")

print("\nBaseline (layer 11 only):")
with open(os.path.join(PROBE_DIR, "layer_11", "probe_accuracy.json")) as f:
    base = json.load(f)
print(f"  Layer 11 probe accuracy: {base['test_acc']:.4f} ({base['test_acc'] * 100:.2f}%)")

print(f"\nDelta: {acc - base['test_acc']:+.4f}")
