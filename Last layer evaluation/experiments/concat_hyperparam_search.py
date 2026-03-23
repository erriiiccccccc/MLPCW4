"""
Hyperparameter search for concat eval.
Tries different C values, scalers, and solvers to find the best combo.
"""

import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

PROBE_DIR = "/home/s2411221/probe_results"
LAYERS    = [8, 9, 10, 11]

# ── Load embeddings ────────────────────────────────────────────────────────────
print("Loading embeddings...")
train_parts, test_parts = [], []
for l in LAYERS:
    layer_dir = os.path.join(PROBE_DIR, f"layer_{l:02d}")
    train_parts.append(np.load(os.path.join(layer_dir, "embeddings.npy")))
    test_parts.append(np.load(os.path.join(layer_dir, "test_embeddings.npy")))

train_labels = np.load(os.path.join(PROBE_DIR, "layer_11", "labels.npy"))
test_labels  = np.load(os.path.join(PROBE_DIR, "layer_11", "test_labels.npy"))

# ── Two scaling strategies ─────────────────────────────────────────────────────
# A) per-layer scale then concat (your current approach)
scaled_per_layer_train, scaled_per_layer_test = [], []
for tr, te in zip(train_parts, test_parts):
    sc = StandardScaler()
    scaled_per_layer_train.append(sc.fit_transform(tr))
    scaled_per_layer_test.append(sc.transform(te))
X_train_perlayer = np.concatenate(scaled_per_layer_train, axis=1)
X_test_perlayer  = np.concatenate(scaled_per_layer_test,  axis=1)

# B) concat raw then single global scale
X_raw_train = np.concatenate(train_parts, axis=1)
X_raw_test  = np.concatenate(test_parts,  axis=1)
sc_global = StandardScaler()
X_train_global = sc_global.fit_transform(X_raw_train)
X_test_global  = sc_global.transform(X_raw_test)

# ── Grid search ────────────────────────────────────────────────────────────────
C_values = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]

configs = [
    ("per-layer scale", X_train_perlayer, X_test_perlayer),
    ("global scale",    X_train_global,   X_test_global),
]

print(f"\n{'Config':<20} {'C':<8} {'Test Acc':>10}")
print("-" * 42)

best_acc = 0
best_config = None

for config_name, X_tr, X_te in configs:
    for C in C_values:
        clf = LogisticRegression(
            C=C,
            solver='saga',       # actually parallel, unlike lbfgs
            max_iter=1000,
            n_jobs=-1,
            random_state=42
        )
        clf.fit(X_tr, train_labels)
        acc = accuracy_score(test_labels, clf.predict(X_te))
        marker = " <-- best" if acc > best_acc else ""
        if acc > best_acc:
            best_acc = acc
            best_config = (config_name, C)
        print(f"  {config_name:<20} C={C:<6} {acc*100:>8.2f}%{marker}")
    print()

print(f"\nBEST: {best_config[0]}, C={best_config[1]}  ->  {best_acc*100:.2f}%")
print(f"Groupmate got: 53.84%")
print(f"Gap: {(best_acc*100 - 53.84):+.2f}%")
