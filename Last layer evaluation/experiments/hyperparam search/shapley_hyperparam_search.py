"""Search C and scaling order for Shapley-weighted fusion."""

import json
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

PROBE_DIR = "/home/s2411221/probe_results"
LAYERS    = [8, 9, 10, 11]
SHAPLEY   = {8: 0.014362, 9: 0.041173, 10: 0.106244, 11: 0.275068}

total   = sum(SHAPLEY[l] for l in LAYERS)
weights = {l: SHAPLEY[l] / total for l in LAYERS}
print("Normalized Shapley weights:")
for l in LAYERS:
    print(f"  Layer {l}: {weights[l]:.4f}")

print("\nLoading embeddings...")
train_parts, test_parts = {}, {}
for l in LAYERS:
    layer_dir = os.path.join(PROBE_DIR, f"layer_{l:02d}")
    train_parts[l] = np.load(os.path.join(layer_dir, "embeddings.npy"))
    test_parts[l]  = np.load(os.path.join(layer_dir, "test_embeddings.npy"))
    print(f"  Layer {l}: train {train_parts[l].shape}, test {test_parts[l].shape}")

train_labels = np.load(os.path.join(PROBE_DIR, "layer_11", "labels.npy"))
test_labels  = np.load(os.path.join(PROBE_DIR, "layer_11", "test_labels.npy"))

train_combined_a = sum(weights[l] * train_parts[l] for l in LAYERS)
test_combined_a  = sum(weights[l] * test_parts[l]  for l in LAYERS)
sc_a = StandardScaler()
X_train_a = sc_a.fit_transform(train_combined_a)
X_test_a  = sc_a.transform(test_combined_a)

scaled_train_parts = {}
scaled_test_parts  = {}
for l in LAYERS:
    sc = StandardScaler()
    scaled_train_parts[l] = sc.fit_transform(train_parts[l])
    scaled_test_parts[l]  = sc.transform(test_parts[l])
X_train_b = sum(weights[l] * scaled_train_parts[l] for l in LAYERS)
X_test_b  = sum(weights[l] * scaled_test_parts[l]  for l in LAYERS)

C_values = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
configs  = [
    ("scale after weight", X_train_a, X_test_a),
    ("scale before weight", X_train_b, X_test_b),
]

print(f"\n{'Scaling':<22} {'C':<8} {'Test Acc':>10}")
print("-" * 46)

best_acc    = 0
best_config = None
all_results = []

for config_name, X_tr, X_te in configs:
    for C in C_values:
        clf = LogisticRegression(C=C, solver='lbfgs', max_iter=1000, random_state=42)
        clf.fit(X_tr, train_labels)
        acc = accuracy_score(test_labels, clf.predict(X_te))
        marker = " <-- best" if acc > best_acc else ""
        if acc > best_acc:
            best_acc = acc
            best_config = {"scaling": config_name, "C": C}
        print(f"  {config_name:<22} C={C:<6} {acc*100:>8.2f}%{marker}", flush=True)
        all_results.append({"scaling": config_name, "C": C, "test_acc": acc})
    print()

print(f"\nBEST: scaling={best_config['scaling']}, C={best_config['C']}  ->  {best_acc*100:.2f}%")

out_dir = os.path.join(PROBE_DIR, "summary", "hyperparam_search")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "shapley_search_results.json")
with open(out_path, "w") as f:
    json.dump({"best": {**best_config, "test_acc": best_acc}, "all": all_results}, f, indent=2)
print(f"Saved to {out_path}")
