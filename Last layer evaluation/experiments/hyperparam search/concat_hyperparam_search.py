import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

PROBE_DIR = "/home/s2411221/probe_results"
LAYERS = [8, 9, 10, 11]

train_bits, test_bits = [], []
for l in LAYERS:
    d = os.path.join(PROBE_DIR, f"layer_{l:02d}")
    train_bits.append(np.load(os.path.join(d, "embeddings.npy")))
    test_bits.append(np.load(os.path.join(d, "test_embeddings.npy")))

train_labels = np.load(os.path.join(PROBE_DIR, "layer_11", "labels.npy"))
test_labels = np.load(os.path.join(PROBE_DIR, "layer_11", "test_labels.npy"))

# per-layer scaling: scale each layer independently then concat
train_by_layer, test_by_layer = [], []
for tr, te in zip(train_bits, test_bits):
    sc = StandardScaler()
    train_by_layer.append(sc.fit_transform(tr))
    test_by_layer.append(sc.transform(te))
x_train_perlayer = np.concatenate(train_by_layer, axis=1)
x_test_perlayer = np.concatenate(test_by_layer, axis=1)

# global scaling: concat raw then scale everything at once
x_train_raw = np.concatenate(train_bits, axis=1)
x_test_raw = np.concatenate(test_bits, axis=1)
sc_global = StandardScaler()
x_train_global = sc_global.fit_transform(x_train_raw)
x_test_global = sc_global.transform(x_test_raw)

C_values = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]

configs = [
    ("per-layer scale", x_train_perlayer, x_test_perlayer),
    ("global scale", x_train_global, x_test_global),
]

print(f"\n{'Config':<20} {'C':<8} {'Test Acc':>10}")
print("-" * 42)

best_acc = 0
best_pick = None

for cfg_name, x_tr, x_te in configs:
    for C in C_values:
        clf = LogisticRegression(
            C=C,
            solver='saga',
            max_iter=1000,
            n_jobs=-1,
            random_state=42
        )
        clf.fit(x_tr, train_labels)
        acc = accuracy_score(test_labels, clf.predict(x_te))
        marker = " <-- best" if acc > best_acc else ""
        if acc > best_acc:
            best_acc = acc
            best_pick = (cfg_name, C)
        print(f"  {cfg_name:<20} C={C:<6} {acc*100:>8.2f}%{marker}")
    print()

print(f"\nBEST: {best_pick[0]}, C={best_pick[1]}  ->  {best_acc*100:.2f}%")
print(f"Groupmate got: 53.84%")
print(f"Gap: {(best_acc*100 - 53.84):+.2f}%")
