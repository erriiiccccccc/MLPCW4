"""Evaluate concatenating layers 8-11 before fitting a linear probe."""

import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

PROBE_DIR = "/home/s2411221/probe_results"
LAYERS    = [8, 9, 10, 11]
EMBED_DIM = 768
C_ADJUSTED = 1.0 / len(LAYERS)

print("Loading and scaling embeddings...")
train_bits, test_bits = [], []

for l in LAYERS:
    d = os.path.join(PROBE_DIR, f"layer_{l:02d}")
    train_emb = np.load(os.path.join(d, "embeddings.npy"))
    test_emb  = np.load(os.path.join(d, "test_embeddings.npy"))
    print(f"  Layer {l}: train {train_emb.shape}, test {test_emb.shape}")

    sc = StandardScaler()
    train_bits.append(sc.fit_transform(train_emb))
    test_bits.append(sc.transform(test_emb))

train_concat = np.concatenate(train_bits, axis=1)
test_concat  = np.concatenate(test_bits,  axis=1)
print(f"\nConcatenated shape: train {train_concat.shape}, test {test_concat.shape}")

train_labels = np.load(os.path.join(PROBE_DIR, "layer_11", "labels.npy"))
test_labels  = np.load(os.path.join(PROBE_DIR, "layer_11", "test_labels.npy"))

print(f"\nTraining linear probe on concatenated embeddings (C={C_ADJUSTED})...")
clf = LogisticRegression(max_iter=1000, C=C_ADJUSTED, solver='lbfgs', n_jobs=-1)
clf.fit(train_concat, train_labels)

preds    = clf.predict(test_concat)
test_acc = accuracy_score(test_labels, preds)

with open(os.path.join(PROBE_DIR, "layer_11", "probe_accuracy.json")) as f:
    base_acc = json.load(f)["test_acc"]

shapley_results_path = os.path.join(PROBE_DIR, "summary", "shapley_weighted_results.json")
shap_acc = None
if os.path.exists(shapley_results_path):
    with open(shapley_results_path) as f:
        shap_acc = json.load(f).get("test_acc")

learned_results_path = os.path.join(PROBE_DIR, "summary", "learned_weighted_results.json")
learn_acc = None
if os.path.exists(learned_results_path):
    with open(learned_results_path) as f:
        learn_acc = json.load(f).get("test_acc")

print("\n" + "="*60)
print("RESULTS — Last-4-layer fusion comparison")
print("="*60)
print(f"  {'Method':<30} {'Test Acc':>10}  {'Delta vs L11':>14}")
print(f"  {'-'*56}")
print(f"  {'Layer 11 only (baseline)':<30} {base_acc:>10.4f}  {'—':>14}")
if shap_acc is not None:
    print(f"  {'Shapley-weighted sum':<30} {shap_acc:>10.4f}  {shap_acc - base_acc:>+14.4f}")
if learn_acc is not None:
    print(f"  {'Learned weighted sum':<30} {learn_acc:>10.4f}  {learn_acc - base_acc:>+14.4f}")
print(f"  {'Concatenation (C={:.2f})'.format(C_ADJUSTED):<30} {test_acc:>10.4f}  {test_acc - base_acc:>+14.4f}")
print("="*60)
print(f"\nConcat dim: {train_concat.shape[1]}  |  C adjusted: 1.0/{len(LAYERS)} = {C_ADJUSTED}")

out_dir = os.path.join(PROBE_DIR, "summary")
os.makedirs(out_dir, exist_ok=True)
dump = {
    "test_acc":          test_acc,
    "baseline_acc":      base_acc,
    "delta":             test_acc - base_acc,
    "concat_dim":        int(train_concat.shape[1]),
    "embed_dim":         EMBED_DIM,
    "num_layers":        len(LAYERS),
    "layers_used":       LAYERS,
    "C_adjusted":        C_ADJUSTED,
    "C_others":          1.0,
}
out_path = os.path.join(out_dir, "concat_results.json")
with open(out_path, "w") as f:
    json.dump(dump, f, indent=2)
print(f"\nSaved to {out_path}")
