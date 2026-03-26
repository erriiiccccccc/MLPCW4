import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

PROBE_DIR = "/home/s2411221/probe_results"
LAYERS = [8, 9, 10, 11]
SHAPLEY = {8: 0.014362, 9: 0.041173, 10: 0.106244, 11: 0.275068}

tot = sum(SHAPLEY[l] for l in LAYERS)
wts = {l: SHAPLEY[l] / tot for l in LAYERS}
print("Shapley weights:")
for l in LAYERS:
    print(f"  Layer {l:2d}: {wts[l]:.4f}")

train_mix = None
test_mix = None

for l in LAYERS:
    d = os.path.join(PROBE_DIR, f"layer_{l:02d}")
    train_emb = np.load(os.path.join(d, "embeddings.npy"))
    test_emb = np.load(os.path.join(d, "test_embeddings.npy"))
    print(f"  Layer {l}: train {train_emb.shape}, test {test_emb.shape}")

    if train_mix is None:
        train_mix = wts[l] * train_emb
        test_mix = wts[l] * test_emb
    else:
        train_mix = train_mix + wts[l] * train_emb
        test_mix = test_mix + wts[l] * test_emb

train_labels = np.load(os.path.join(PROBE_DIR, "layer_11", "labels.npy"))
test_labels = np.load(os.path.join(PROBE_DIR, "layer_11", "test_labels.npy"))

scaler = StandardScaler()
train_mix = scaler.fit_transform(train_mix)
test_mix = scaler.transform(test_mix)

clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', n_jobs=-1)
clf.fit(train_mix, train_labels)

acc = accuracy_score(test_labels, clf.predict(test_mix))
print(f"\nShapley fusion acc: {acc:.4f} ({acc * 100:.2f}%)")

with open(os.path.join(PROBE_DIR, "layer_11", "probe_accuracy.json")) as f:
    base = json.load(f)
print(f"Layer 11 baseline: {base['test_acc']:.4f} ({base['test_acc'] * 100:.2f}%)")
print(f"Delta: {acc - base['test_acc']:+.4f}")
