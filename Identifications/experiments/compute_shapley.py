#!/usr/bin/env python3
"""
Approximate Shapley Values for TimeSformer Linear Probes
=========================================================
Loads pre-saved layer embeddings from probe_results/ and computes
approximate Shapley values via random permutation sampling.

No GPU needed — runs on CPU using saved .npy files.

Method: For each random permutation of 12 layers, evaluate the marginal
contribution of each layer as it is added to the coalition. Average marginal
contributions across all permutations = approximate Shapley value.

Coalition values are cached so shared sub-coalitions are not re-evaluated.

Usage:
    python compute_shapley.py \
        --probe_dir ./MLPCW4/probe_results \
        --n_permutations 50 \
        --n_samples 3000
"""

import os
import json
import time
import argparse
import numpy as np
from math import factorial
from itertools import combinations

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD SAVED EMBEDDINGS
# ============================================================
def load_embeddings(probe_dir, num_layers=12):
    """Load all layer embeddings and labels from disk."""
    layer_results = {}
    labels = None

    for i in range(num_layers):
        layer_dir = os.path.join(probe_dir, f'layer_{i:02d}')
        emb_path = os.path.join(layer_dir, 'embeddings.npy')
        lbl_path = os.path.join(layer_dir, 'labels.npy')

        if not os.path.exists(emb_path):
            print(f"  WARNING: Missing embeddings for layer {i}, skipping")
            continue

        emb = np.load(emb_path)
        layer_results[i] = {'embeddings': emb}

        if labels is None and os.path.exists(lbl_path):
            labels = np.load(lbl_path)

        print(f"  Layer {i:2d}: {emb.shape}")

    if labels is None:
        raise RuntimeError("Could not find labels.npy in any layer directory.")

    return layer_results, labels


# ============================================================
# 2. COALITION VALUE FUNCTION
# ============================================================
def coalition_value(layer_results, labels, coalition,
                    test_size=0.3, random_state=42):
    """
    Accuracy of a linear probe on concatenated features from coalition layers.
    Empty coalition = majority class baseline.
    """
    if len(coalition) == 0:
        unique, counts = np.unique(labels, return_counts=True)
        return float(counts.max()) / len(labels)

    feats = np.concatenate(
        [layer_results[i]['embeddings'] for i in sorted(coalition)], axis=1
    )

    X_train, X_test, y_train, y_test = train_test_split(
        feats, labels, test_size=test_size,
        random_state=random_state
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=200, C=1.0, solver='lbfgs')
    clf.fit(X_train_s, y_train)
    return float(clf.score(X_test_s, y_test))


# ============================================================
# 3. APPROXIMATE SHAPLEY (PERMUTATION SAMPLING + CACHE)
# ============================================================
def approx_shapley(layer_results, labels, num_layers, n_permutations,
                   random_state=0):
    """
    Estimate Shapley values via random permutation sampling.

    For each permutation, walk through players in order and compute
    marginal contribution of each player as it joins the coalition.
    Cache coalition values to avoid redundant LR fits.
    """
    rng = np.random.RandomState(random_state)
    players = list(layer_results.keys())
    n = len(players)

    shapley = {i: 0.0 for i in players}
    cache = {}
    total_fits = 0
    cache_hits = 0

    start = time.time()
    print(f"\nApproximate Shapley: {n_permutations} permutations, "
          f"{n} players")
    print(f"Max unique coalitions possible: {2**n} "
          f"(exact would need all of them)")

    for perm_idx in range(n_permutations):
        perm = rng.permutation(players).tolist()
        coalition = []
        prev_val = None

        for step, player in enumerate(perm):
            # Value without player (prev step)
            key_before = frozenset(coalition)
            if key_before in cache:
                v_before = cache[key_before]
                cache_hits += 1
            else:
                v_before = coalition_value(layer_results, labels,
                                           list(key_before))
                cache[key_before] = v_before
                total_fits += 1

            # Value with player
            coalition_with = coalition + [player]
            key_after = frozenset(coalition_with)
            if key_after in cache:
                v_after = cache[key_after]
                cache_hits += 1
            else:
                v_after = coalition_value(layer_results, labels,
                                          coalition_with)
                cache[key_after] = v_after
                total_fits += 1

            marginal = v_after - v_before
            shapley[player] += marginal / n_permutations

            coalition.append(player)

        elapsed = time.time() - start
        rate = (perm_idx + 1) / elapsed
        remaining = (n_permutations - perm_idx - 1) / rate if rate > 0 else 0

        print(f"  Permutation {perm_idx+1:3d}/{n_permutations}  |  "
              f"cache size: {len(cache):4d}  |  "
              f"fits: {total_fits:4d}  hits: {cache_hits:4d}  |  "
              f"elapsed: {elapsed/60:.1f}m  "
              f"remaining: {remaining/60:.1f}m")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} minutes")
    print(f"Total LR fits: {total_fits}  (cache saved {cache_hits} fits)")

    return shapley, cache


# ============================================================
# 4. SAVE + PLOT
# ============================================================
def load_probe_accuracies(probe_dir, num_layers):
    """Load per-layer test accuracies from saved probe_accuracy.json files."""
    accs = {}
    for i in range(num_layers):
        path = os.path.join(probe_dir, f'layer_{i:02d}', 'probe_accuracy.json')
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            accs[i] = d.get('test_acc', 0.0)
    return accs


def save_and_plot(shapley, cache, probe_dir, n_permutations, n_samples):
    summary_dir = os.path.join(probe_dir, 'summary')
    os.makedirs(summary_dir, exist_ok=True)

    players = sorted(shapley.keys())
    num_layers = len(players)
    probe_accs = load_probe_accuracies(probe_dir, num_layers)

    # Print ranking
    print(f"\n{'='*60}")
    print("SHAPLEY VALUE RANKING")
    print(f"{'='*60}")
    ranked = sorted(shapley.items(), key=lambda x: x[1], reverse=True)
    for rank, (layer, sv) in enumerate(ranked):
        acc = probe_accs.get(layer, float('nan'))
        print(f"  #{rank+1:2d}  Layer {layer:2d}  |  "
              f"Shapley: {sv:+.6f}  |  Probe acc: {acc:.4f}")

    # Save JSON
    shapley_export = {
        'method': 'permutation_sampling',
        'n_permutations': n_permutations,
        'n_samples': n_samples,
        'shapley_values': {str(k): float(v) for k, v in shapley.items()},
        'ranking': [str(k) for k, v in ranked],
        'description': (
            'Approximate Shapley value via random permutation sampling. '
            'Value = average marginal contribution of each layer across '
            f'{n_permutations} random orderings.'
        ),
    }
    sv_path = os.path.join(summary_dir, 'shapley_values.json')
    with open(sv_path, 'w') as f:
        json.dump(shapley_export, f, indent=2)
    print(f"\nSaved: {sv_path}")

    # Save coalition cache
    coalition_export = {}
    for coalition_set, value in cache.items():
        key = ','.join(str(x) for x in sorted(coalition_set)) or 'empty'
        coalition_export[key] = value
    cc_path = os.path.join(summary_dir, 'coalition_values.json')
    with open(cc_path, 'w') as f:
        json.dump(coalition_export, f, indent=2)
    print(f"Saved: {cc_path}")

    # Plot
    layers = players
    sv_vals = [shapley[l] for l in layers]
    acc_vals = [probe_accs.get(l, 0) for l in layers]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Probe accuracy per layer
    axes[0].plot(layers, acc_vals, 'r-o', markersize=6)
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Test Accuracy')
    axes[0].set_title('Linear Probe Accuracy per Layer')
    axes[0].set_xticks(layers)
    axes[0].grid(True, alpha=0.3)

    # Shapley values
    colors = ['green' if v >= 0 else 'red' for v in sv_vals]
    axes[1].bar(layers, sv_vals, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Shapley Value')
    axes[1].set_title(f'Shapley Value per Layer\n({n_permutations} permutations)')
    axes[1].set_xticks(layers)
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].grid(True, alpha=0.3)

    # Marginal accuracy gain
    gains = [acc_vals[0]] + [acc_vals[i] - acc_vals[i-1]
                              for i in range(1, len(acc_vals))]
    colors_g = ['green' if g >= 0 else 'red' for g in gains]
    axes[2].bar(layers, gains, color=colors_g, alpha=0.7, edgecolor='black')
    axes[2].set_xlabel('Layer')
    axes[2].set_ylabel('Accuracy Gain')
    axes[2].set_title('Marginal Accuracy Gain (vs Previous Layer)')
    axes[2].set_xticks(layers)
    axes[2].axhline(y=0, color='black', linewidth=0.5)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(summary_dir, 'layer_comparison_plot.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Approximate Shapley values from saved layer embeddings'
    )
    parser.add_argument('--probe_dir', type=str, required=True,
                        help='Path to probe_results/ directory')
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--n_permutations', type=int, default=50,
                        help='Random permutations to sample (default: 50)')
    parser.add_argument('--n_samples', type=int, default=3000,
                        help='Subsample size for coalition evaluation (default: 3000)')
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("APPROXIMATE SHAPLEY VALUE COMPUTATION")
    print("=" * 60)
    print(f"  probe_dir:      {args.probe_dir}")
    print(f"  n_permutations: {args.n_permutations}")
    print(f"  n_samples:      {args.n_samples}")
    print(f"  num_layers:     {args.num_layers}")

    # Load embeddings
    print("\nLoading saved embeddings...")
    layer_results, labels = load_embeddings(args.probe_dir, args.num_layers)
    print(f"Loaded {len(layer_results)} layers, {len(labels)} samples")

    # Subsample for speed
    if args.n_samples < len(labels):
        rng = np.random.RandomState(args.random_state)
        idx = rng.choice(len(labels), args.n_samples, replace=False)
        layer_results_sub = {
            i: {'embeddings': v['embeddings'][idx]}
            for i, v in layer_results.items()
        }
        labels_sub = labels[idx]
        print(f"Subsampled to {args.n_samples} videos for coalition evaluation")
    else:
        layer_results_sub = layer_results
        labels_sub = labels

    # Estimate time
    est_unique = min(args.n_permutations * args.num_layers * 2,
                     2 ** args.num_layers)
    print(f"\nEstimated unique coalition evaluations: ~{est_unique}")
    print(f"Estimated time: {est_unique * 25 / 3600:.1f}–"
          f"{est_unique * 60 / 3600:.1f} hours (25–60s per LR fit)")

    # Run approximate Shapley
    shapley, cache = approx_shapley(
        layer_results_sub, labels_sub,
        num_layers=len(layer_results_sub),
        n_permutations=args.n_permutations,
        random_state=args.random_state
    )

    # Save and plot
    save_and_plot(shapley, cache, args.probe_dir,
                  args.n_permutations, args.n_samples)

    print("\nDone!")


if __name__ == '__main__':
    main()
