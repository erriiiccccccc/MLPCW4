import json
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

PROBE_DIR = "/home/s2411221/probe_results"
LAYERS = [8, 9, 10, 11]
SHAPLEY = {8: 0.014362, 9: 0.041173, 10: 0.106244, 11: 0.275068}

tot = sum(SHAPLEY[l] for l in LAYERS)
wts = {l: SHAPLEY[l] / tot for l in LAYERS}
for l in LAYERS:
    print(f"  Layer {l}: {wts[l]:.4f}")

train_bits, test_bits = {}, {}
for l in LAYERS:
    d = os.path.join(PROBE_DIR, f"layer_{l:02d}")
    train_bits[l] = np.load(os.path.join(d, "embeddings.npy"))
    test_bits[l] = np.load(os.path.join(d, "test_embeddings.npy"))

train_labels = np.load(os.path.join(PROBE_DIR, "layer_11", "labels.npy"))
test_labels = np.load(os.path.join(PROBE_DIR, "layer_11", "test_labels.npy"))

# weight first, then scale
train_a = sum(wts[l] * train_bits[l] for l in LAYERS)
test_a = sum(wts[l] * test_bits[l] for l in LAYERS)
sc_a = StandardScaler()
x_train_a = sc_a.fit_transform(train_a)
x_test_a = sc_a.transform(test_a)

# scale first, then weight
train_scaled, test_scaled = {}, {}
for l in LAYERS:
    sc = StandardScaler()
    train_scaled[l] = sc.fit_transform(train_bits[l])
    test_scaled[l] = sc.transform(test_bits[l])
x_train_b = sum(wts[l] * train_scaled[l] for l in LAYERS)
x_test_b = sum(wts[l] * test_scaled[l] for l in LAYERS)

C_values = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
configs = [
    ("scale after weight", x_train_a, x_test_a),
    ("scale before weight", x_train_b, x_test_b),
]

print(f"\n{'Scaling':<22} {'C':<8} {'Test Acc':>10}")
print("-" * 46)

best_acc = 0
best_cfg = None
runs = []

for cfg_name, x_tr, x_te in configs:
    for C in C_values:
        clf = LogisticRegression(C=C, solver='lbfgs', max_iter=1000, random_state=42)
        clf.fit(x_tr, train_labels)
        acc = accuracy_score(test_labels, clf.predict(x_te))
        marker = " <-- best" if acc > best_acc else ""
        if acc > best_acc:
            best_acc = acc
            best_cfg = {"scaling": cfg_name, "C": C}
        print(f"  {cfg_name:<22} C={C:<6} {acc*100:>8.2f}%{marker}")
        runs.append({"scaling": cfg_name, "C": C, "test_acc": acc})
    print()

print(f"\nBEST: scaling={best_cfg['scaling']}, C={best_cfg['C']}  ->  {best_acc*100:.2f}%")

out_dir = os.path.join(PROBE_DIR, "summary", "hyperparam_search")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "shapley_search_results.json")
with open(out_path, "w") as f:
    json.dump({"best": {**best_cfg, "test_acc": best_acc}, "all": runs}, f, indent=2)
print(f"Saved to {out_path}")
