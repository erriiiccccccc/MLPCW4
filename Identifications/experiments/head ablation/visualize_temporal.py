import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram, linkage


DPI = 150


def plot_shapley_heatmap(shapley_df, output_dir, num_layers=12, num_heads=12):
    fig, ax = plt.subplots(figsize=(14, 8))

    pivot = shapley_df.pivot(index="layer", columns="head", values="shapley_value")
    pivot = pivot.reindex(index=range(num_layers), columns=range(num_heads), fill_value=0)

    vmax = pivot.abs().values.max()
    sns.heatmap(
        pivot, ax=ax, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
        annot=True, fmt=".3f", linewidths=0.5,
        cbar_kws={"label": "Shapley Value"},
        xticklabels=[f"H{i}" for i in range(num_heads)],
        yticklabels=[f"L{i}" for i in range(num_layers)],
    )
    ax.set_title("Temporal Head Shapley Values", fontsize=14)
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Layer Index")

    plt.tight_layout()
    path = output_dir / "shapley_heatmap.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Shapley heatmap saved to {path}")


def plot_attention_gallery(shapley_df, attention_dir, output_dir, top_n=5):
    layer_attns = {}
    for npy_file in sorted(attention_dir.glob("attention_L*.npy")):
        layer_idx = int(npy_file.stem.split("_L")[1])
        layer_attns[layer_idx] = np.load(npy_file)

    if not layer_attns:
        print("  Attention gallery skipped -- no .npy files found")
        return

    sorted_df = shapley_df.sort_values("shapley_value", ascending=False)
    available = sorted_df[sorted_df["layer"].isin(layer_attns.keys())]
    top_heads = available.head(top_n)
    bottom_heads = available.tail(top_n)

    all_heads = pd.concat([top_heads, bottom_heads])
    if len(all_heads) == 0:
        print("  Attention gallery skipped -- no matching heads")
        return

    fig, axes = plt.subplots(2, top_n, figsize=(3 * top_n, 7))
    if top_n == 1:
        axes = axes.reshape(2, 1)

    for idx, (group_name, group_df) in enumerate([
        ("Top (most important)", top_heads),
        ("Bottom (least important)", bottom_heads),
    ]):
        for col, (_, row) in enumerate(group_df.iterrows()):
            if col >= top_n:
                break
            layer = int(row["layer"])
            head = int(row["head"])
            if layer not in layer_attns:
                continue

            mean_attn = layer_attns[layer].mean(axis=0)[head]

            ax = axes[idx, col]
            sns.heatmap(
                mean_attn, ax=ax, cmap="viridis", vmin=0,
                annot=True, fmt=".2f", cbar=False, square=True,
                xticklabels=[f"F{t}" for t in range(mean_attn.shape[0])],
                yticklabels=[f"F{t}" for t in range(mean_attn.shape[1])],
            )
            ax.set_title(f"L{layer}-H{head}\nphi={row['shapley_value']:.4f}", fontsize=9)
            if col == 0:
                ax.set_ylabel(group_name, fontsize=10)

    plt.suptitle("Temporal Attention Patterns: Top vs Bottom Heads", fontsize=13)
    plt.tight_layout()
    path = output_dir / "attention_patterns.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Attention gallery saved to {path}")


def plot_head_dendrogram(attention_dir, output_dir):
    layer_attns = {}
    for npy_file in sorted(attention_dir.glob("attention_L*.npy")):
        layer_idx = int(npy_file.stem.split("_L")[1])
        layer_attns[layer_idx] = np.load(npy_file)

    if not layer_attns:
        print("  Dendrogram skipped -- no .npy files found")
        return

    labels = []
    features = []
    for layer_idx in sorted(layer_attns.keys()):
        mean_attn = layer_attns[layer_idx].mean(axis=0)
        for h in range(mean_attn.shape[0]):
            labels.append(f"L{layer_idx}-H{h}")
            features.append(mean_attn[h].flatten())

    feature_matrix = np.stack(features)
    Z = linkage(feature_matrix, method="ward", metric="euclidean")

    fig, ax = plt.subplots(figsize=(max(16, len(labels) * 0.3), 8))
    dendrogram(
        Z, labels=labels, ax=ax,
        leaf_rotation=90, leaf_font_size=7,
        color_threshold=Z[-4, 2] if len(Z) >= 4 else 0,
    )
    ax.set_title("Temporal Head Clustering (Ward Linkage)", fontsize=14)
    ax.set_xlabel("Head")
    ax.set_ylabel("Distance")

    plt.tight_layout()
    path = output_dir / "head_clustering.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Dendrogram saved to {path}")


def plot_semantic_importance(per_class_path, output_dir):
    if not per_class_path.exists():
        print("  Semantic importance skipped -- per_class_importance.csv not found")
        return

    df = pd.read_csv(per_class_path)
    if len(df) == 0 or "category" not in df.columns:
        print("  Semantic importance skipped -- empty or missing category column")
        return

    cat_importance = df.groupby("category")["flip_rate"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("Set2", len(cat_importance))
    bars = ax.barh(range(len(cat_importance)), cat_importance.values, color=colors)
    ax.set_yticks(range(len(cat_importance)))
    ax.set_yticklabels(cat_importance.index, fontsize=10)
    ax.set_xlabel("Mean Flip Rate (Head Importance)", fontsize=12)
    ax.set_title("Temporal Head Importance by SSv2 Action Category", fontsize=14)
    ax.invert_yaxis()

    for bar, val in zip(bars, cat_importance.values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    path = output_dir / "semantic_importance.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Semantic importance saved to {path}")


def plot_redundancy_matrices(output_dir, num_heads=12):
    sim_files = sorted(output_dir.glob("sim_matrix_L*.npy"))
    if not sim_files:
        print("  Redundancy heatmaps skipped -- no sim_matrix files found")
        return

    n_layers = len(sim_files)
    cols = min(4, n_layers)
    rows = (n_layers + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    if n_layers == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)

    for idx, npy_file in enumerate(sim_files):
        layer_idx = int(npy_file.stem.split("_L")[1])
        sim = np.load(npy_file)

        r, c = divmod(idx, cols)
        ax = axes[r, c]
        sns.heatmap(
            sim, ax=ax, cmap="YlOrRd", vmin=0, vmax=1,
            annot=True, fmt=".2f", linewidths=0.3, square=True, cbar=False,
            xticklabels=[f"H{i}" for i in range(sim.shape[0])],
            yticklabels=[f"H{i}" for i in range(sim.shape[0])],
        )
        ax.set_title(f"Layer {layer_idx}", fontsize=11)

    for idx in range(n_layers, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].set_visible(False)

    plt.suptitle("Temporal Head Redundancy (Cosine Similarity)", fontsize=14)
    plt.tight_layout()
    path = output_dir / "redundancy_matrices.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Redundancy matrices saved to {path}")


def plot_propagation_heatmap(propagation_df, output_dir, metric="l2_distance"):
    pivot = propagation_df.groupby(
        ["source_layer", "target_layer"]
    )[metric].mean().reset_index()
    heatmap_data = pivot.pivot(
        index="source_layer", columns="target_layer", values=metric,
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        heatmap_data, ax=ax, cmap="YlOrRd",
        annot=True, fmt=".3f", linewidths=0.5,
        cbar_kws={"label": metric.replace("_", " ").title()},
        xticklabels=[f"L{c}" for c in heatmap_data.columns],
        yticklabels=[f"L{r}" for r in heatmap_data.index],
    )
    ax.set_title(f"Propagation Effect: {metric.replace('_', ' ').title()}", fontsize=14)
    ax.set_xlabel("Target Layer (downstream)")
    ax.set_ylabel("Source Layer (ablated)")

    plt.tight_layout()
    path = output_dir / "propagation_heatmap.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Propagation heatmap saved to {path}")


def plot_propagation_curves(propagation_df, output_dir):
    source_layers = sorted(propagation_df["source_layer"].unique())
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("tab10", len(source_layers))

    for idx, sl in enumerate(source_layers):
        sl_df = propagation_df[propagation_df["source_layer"] == sl]
        downstream = sl_df[sl_df["target_layer"] >= sl]
        if len(downstream) == 0:
            continue
        curve = downstream.groupby("target_layer")["l2_distance"].mean()
        ax.plot(curve.index, curve.values,
                marker="o", linewidth=2, markersize=6,
                label=f"Ablate L{sl}", color=colors[idx])

    ax.set_xlabel("Target Layer", fontsize=12)
    ax.set_ylabel("Relative L2 Distance (normalized)", fontsize=12)
    ax.set_title("Propagation Curves: Ablation Effect Through Network", fontsize=14)
    ax.set_xticks(range(12))
    ax.set_xticklabels([f"L{i}" for i in range(12)])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "propagation_curves.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Propagation curves saved to {path}")


def plot_cka_heatmap(propagation_df, output_dir):
    """CKA near 1.0 = representation geometry preserved despite ablation"""
    if "cka" not in propagation_df.columns:
        print("  CKA heatmap skipped -- no 'cka' column")
        return

    pivot = propagation_df.groupby(
        ["source_layer", "target_layer"]
    )["cka"].mean().reset_index()
    heatmap_data = pivot.pivot(
        index="source_layer", columns="target_layer", values="cka",
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        heatmap_data, ax=ax, cmap="RdYlGn", vmin=0.9, vmax=1.0,
        annot=True, fmt=".3f", linewidths=0.5,
        cbar_kws={"label": "CKA Score"},
        xticklabels=[f"L{c}" for c in heatmap_data.columns],
        yticklabels=[f"L{r}" for r in heatmap_data.index],
    )
    ax.set_title("Representation Similarity After Ablation (Linear CKA)", fontsize=14)
    ax.set_xlabel("Target Layer (downstream)")
    ax.set_ylabel("Source Layer (ablated)")

    plt.tight_layout()
    path = output_dir / "propagation_cka.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  CKA heatmap saved to {path}")


def run_all_visualizations(output_dir="outputs/temporal_analysis"):
    output_path = Path(output_dir)

    print("Temporal Analysis Visualizations")

    shapley_path = output_path / "shapley_values.csv"
    if shapley_path.exists():
        shapley_df = pd.read_csv(shapley_path)
        print("\n1. Shapley heatmap")
        plot_shapley_heatmap(shapley_df, output_path)
    else:
        print("\n1. Shapley heatmap -- SKIPPED (no shapley_values.csv)")
        shapley_df = None

    if shapley_df is not None:
        print("\n2. Attention pattern gallery")
        plot_attention_gallery(shapley_df, output_path, output_path)
    else:
        print("\n2. Attention gallery -- SKIPPED")

    print("\n3. Head clustering dendrogram")
    plot_head_dendrogram(output_path, output_path)

    print("\n4. Semantic importance by category")
    plot_semantic_importance(output_path / "per_class_importance.csv", output_path)

    print("\n5. Redundancy matrix heatmaps")
    plot_redundancy_matrices(output_path)

    propagation_path = output_path / "propagation_effects.csv"
    if propagation_path.exists():
        prop_df = pd.read_csv(propagation_path)
        print("\n6. Propagation heatmap")
        plot_propagation_heatmap(prop_df, output_path)
        print("\n7. Propagation curves")
        plot_propagation_curves(prop_df, output_path)
        print("\n8. CKA heatmap")
        plot_cka_heatmap(prop_df, output_path)
    else:
        print("\n6-8. Propagation visualizations -- SKIPPED (no propagation_effects.csv)")

    print("\nVISUALIZATIONS done")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs/temporal_analysis")
    args = parser.parse_args()

    run_all_visualizations(output_dir=args.output_dir)
