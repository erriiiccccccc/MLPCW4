"""
Concatenated last-4-layers evaluation.

Concatenates CLS embeddings from layers 8-11 into a single 3072-dim vector
(4 × 768), then trains a linear probe.

Fair-comparison notes vs shapley_weighted_eval.py and learned_weighted_eval.py:
- Same train/test split, same StandardScaler per layer, same LogisticRegression solver.
- Concat produces a 3072-dim vector while the other fusion methods produce 768-dim.
  LogisticRegression with C=1.0 on 3072-dim would have 4× more parameters, giving
  an unfair capacity advantage.  To equalize per-feature regularization strength we
  set C = 1.0 / 4 = 0.25 (the L2 penalty scales as 1/C, so 4× the features needs
  4× stronger regularization to keep the same effective shrinkage per feature after
  StandardScaling).
- Unlike weighted-sum methods (which learn one scalar weight per layer shared across
  all classes), concat + linear probe implicitly learns per-class layer weights,
  making it strictly more expressive.  That difference is architectural, not a
  capacity cheat, and is the whole point of including it as a comparison.
"""

import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os

# ── config ────────────────────────────────────────────────────────────────────
PROBE_DIR = "/home/s2411221/probe_results"
LAYERS    = [8, 9, 10, 11]
EMBED_DIM = 768
# C is scaled down by num_layers so per-feature L2 regularization matches
# the C=1.0 used on 768-dim embeddings in shapley_weighted_eval.py
C_ADJUSTED = 1.0 / len(LAYERS)   # = 0.25
# ──────────────────────────────────────────────────────────────────────────────

# ── 1. Load & scale each layer independently (same as learned_weighted_eval) ─
print("Loading and scaling embeddings...")
scaled_train_parts, scaled_test_parts = [], []

for l in LAYERS:
    layer_dir = os.path.join(PROBE_DIR, f"layer_{l:02d}")
    train_emb = np.load(os.path.join(layer_dir, "embeddings.npy"))       # (N, 768)
    test_emb  = np.load(os.path.join(layer_dir, "test_embeddings.npy"))  # (M, 768)
    print(f"  Layer {l}: train {train_emb.shape}, test {test_emb.shape}")

    sc = StandardScaler()
    scaled_train_parts.append(sc.fit_transform(train_emb))
    scaled_test_parts.append(sc.transform(test_emb))

# ── 2. Concatenate → (N, 4×768) = (N, 3072) ──────────────────────────────────
train_concat = np.concatenate(scaled_train_parts, axis=1)  # (N, 3072)
test_concat  = np.concatenate(scaled_test_parts,  axis=1)  # (M, 3072)
print(f"\nConcatenated shape: train {train_concat.shape}, test {test_concat.shape}")

# ── 3. Labels ─────────────────────────────────────────────────────────────────
train_labels = np.load(os.path.join(PROBE_DIR, "layer_11", "labels.npy"))
test_labels  = np.load(os.path.join(PROBE_DIR, "layer_11", "test_labels.npy"))

# ── 4. Train linear probe ─────────────────────────────────────────────────────
print(f"\nTraining linear probe on concatenated embeddings (C={C_ADJUSTED})...")
clf = LogisticRegression(max_iter=1000, C=C_ADJUSTED, solver='lbfgs', n_jobs=-1)
clf.fit(train_concat, train_labels)

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
preds    = clf.predict(test_concat)
test_acc = accuracy_score(test_labels, preds)

# ── 6. Baselines from saved results ──────────────────────────────────────────
with open(os.path.join(PROBE_DIR, "layer_11", "probe_accuracy.json")) as f:
    baseline_acc = json.load(f)["test_acc"]

shapley_results_path = os.path.join(PROBE_DIR, "summary", "shapley_weighted_results.json")
shapley_acc = None
if os.path.exists(shapley_results_path):
    with open(shapley_results_path) as f:
        shapley_acc = json.load(f).get("test_acc")

learned_results_path = os.path.join(PROBE_DIR, "summary", "learned_weighted_results.json")
learned_acc = None
if os.path.exists(learned_results_path):
    with open(learned_results_path) as f:
        learned_acc = json.load(f).get("test_acc")

# ── 7. Print comparison table ─────────────────────────────────────────────────
print("\n" + "="*60)
print("RESULTS — Last-4-layer fusion comparison")
print("="*60)
print(f"  {'Method':<30} {'Test Acc':>10}  {'Delta vs L11':>14}")
print(f"  {'-'*56}")
print(f"  {'Layer 11 only (baseline)':<30} {baseline_acc:>10.4f}  {'—':>14}")
if shapley_acc is not None:
    print(f"  {'Shapley-weighted sum':<30} {shapley_acc:>10.4f}  {shapley_acc - baseline_acc:>+14.4f}")
if learned_acc is not None:
    print(f"  {'Learned weighted sum':<30} {learned_acc:>10.4f}  {learned_acc - baseline_acc:>+14.4f}")
print(f"  {'Concatenation (C={:.2f})'.format(C_ADJUSTED):<30} {test_acc:>10.4f}  {test_acc - baseline_acc:>+14.4f}")
print("="*60)
print(f"\nConcat dim: {train_concat.shape[1]}  |  C adjusted: 1.0/{len(LAYERS)} = {C_ADJUSTED}")

# ── 8. Save results ───────────────────────────────────────────────────────────
out_dir = os.path.join(PROBE_DIR, "summary")
os.makedirs(out_dir, exist_ok=True)
results = {
    "test_acc":          test_acc,
    "baseline_acc":      baseline_acc,
    "delta":             test_acc - baseline_acc,
    "concat_dim":        int(train_concat.shape[1]),
    "embed_dim":         EMBED_DIM,
    "num_layers":        len(LAYERS),
    "layers_used":       LAYERS,
    "C_adjusted":        C_ADJUSTED,
    "C_others":          1.0,
}
out_path = os.path.join(out_dir, "concat_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")
