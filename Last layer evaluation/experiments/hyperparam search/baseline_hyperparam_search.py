"""Search C for the layer-11 logistic-regression baseline."""

import json
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

PROBE_DIR = "/home/s2411221/probe_results"
LAYER     = 11

print("Loading embeddings...", flush=True)
layer_dir   = os.path.join(PROBE_DIR, f"layer_{LAYER:02d}")
X_train_raw = np.load(os.path.join(layer_dir, "embeddings.npy"))
X_test_raw  = np.load(os.path.join(layer_dir, "test_embeddings.npy"))
y_train     = np.load(os.path.join(layer_dir, "labels.npy"))
y_test      = np.load(os.path.join(layer_dir, "test_labels.npy"))
print(f"  Train: {X_train_raw.shape[0]} | Test: {X_test_raw.shape[0]}", flush=True)

sc = StandardScaler()
X_train = sc.fit_transform(X_train_raw)
X_test  = sc.transform(X_test_raw)

C_values = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]

print(f"\n{'C':<10} {'Test Acc':>10}", flush=True)
print("-" * 24, flush=True)

best_acc    = 0
best_C      = None
all_results = []

for C in C_values:
    clf = LogisticRegression(C=C, solver='lbfgs', max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    acc    = accuracy_score(y_test, clf.predict(X_test))
    marker = " <-- best" if acc > best_acc else ""
    if acc > best_acc:
        best_acc = acc
        best_C   = C
    print(f"  C={C:<8} {acc*100:>8.2f}%{marker}", flush=True)
    all_results.append({"C": C, "test_acc": acc})

print(f"\nBEST: C={best_C}  ->  {best_acc*100:.2f}%")

out_dir  = os.path.join(PROBE_DIR, "summary", "hyperparam_search")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "baseline_search_results.json")
with open(out_path, "w") as f:
    json.dump({"best": {"C": best_C, "test_acc": best_acc}, "all": all_results}, f, indent=2)
print(f"Saved to {out_path}")
