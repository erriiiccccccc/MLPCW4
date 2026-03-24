"""visualization stuff for head ablation results - heatmaps, scatter plots, etc."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
from config import AblationConfig


def plot_ablation_heatmaps(df: pd.DataFrame, config: AblationConfig,
                           metric: str = "acc_drop",
                           save: bool = True) -> None:
    """plots side-by-side heatmaps of accuracy drop per head (temporal + spatial).
    layers on y-axis, heads on x-axis.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for idx, attn_type in enumerate(["temporal", "spatial"]):
        subset = df[df["attn_type"] == attn_type]
        pivot = subset.pivot(index="layer", columns="head", values=metric)

        # fill missing values with 0 so the heatmap doesnt crash
        full_index = range(config.num_layers)
        full_cols = range(config.num_heads)
        pivot = pivot.reindex(index=full_index, columns=full_cols, fill_value=0)

        ax = axes[idx]
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="RdYlGn_r",
            center=0,
            annot=True,
            fmt=".3f",
            linewidths=0.5,
            cbar_kws={"label": "Accuracy Drop"},
            xticklabels=[f"H{i}" for i in range(config.num_heads)],
            yticklabels=[f"L{i}" for i in range(config.num_layers)],
        )
        ax.set_title(f"{attn_type.capitalize()} Attention Heads", fontsize=14)
        ax.set_xlabel("Head Index")
        ax.set_ylabel("Layer Index")

    plt.suptitle("Head Ablation: Accuracy Drop per Head", fontsize=16, y=1.02)
    plt.tight_layout()

    if save:
        path = config.output_dir / "ablation_heatmaps.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Heatmap saved to {path}")

    plt.close(fig)


def plot_importance_heatmaps(df: pd.DataFrame, config: AblationConfig,
                             save: bool = True) -> None:
    """Plot gradient-based importance score heatmaps."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for idx, attn_type in enumerate(["temporal", "spatial"]):
        subset = df[df["attn_type"] == attn_type]
        pivot = subset.pivot(index="layer", columns="head", values="importance")

        full_index = range(config.num_layers)
        full_cols = range(config.num_heads)
        pivot = pivot.reindex(index=full_index, columns=full_cols, fill_value=0)

        ax = axes[idx]
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="YlOrRd",
            annot=True,
            fmt=".4f",
            linewidths=0.5,
            cbar_kws={"label": "Importance Score"},
            xticklabels=[f"H{i}" for i in range(config.num_heads)],
            yticklabels=[f"L{i}" for i in range(config.num_layers)],
        )
        ax.set_title(f"{attn_type.capitalize()} Attention - Gradient Importance",
                      fontsize=14)
        ax.set_xlabel("Head Index")
        ax.set_ylabel("Layer Index")

    plt.suptitle("Gradient-Based Head Importance Scores", fontsize=16, y=1.02)
    plt.tight_layout()

    if save:
        path = config.output_dir / "gradient_importance_heatmaps.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Importance heatmap saved to {path}")

    plt.close(fig)


def plot_cumulative_pruning(df: pd.DataFrame, config: AblationConfig,
                            baseline_acc: float,
                            save: bool = True) -> None:
    """Plot cumulative pruning curve.

    Sorts heads by importance (least to most), iteratively ablates,
    and plots accuracy vs percentage of heads removed.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Overall pruning curve
    sorted_df = df.sort_values("acc_drop", ascending=True)
    x_pct = np.arange(1, len(sorted_df) + 1) / len(sorted_df) * 100
    cumulative_drop = sorted_df["acc_drop"].cumsum().values

    ax.plot(x_pct, baseline_acc - cumulative_drop, label="All heads", linewidth=2)

    # Temporal-only curve
    temporal_df = df[df["attn_type"] == "temporal"].sort_values("acc_drop", ascending=True)
    if len(temporal_df) > 0:
        x_t = np.arange(1, len(temporal_df) + 1) / len(temporal_df) * 100
        cum_t = temporal_df["acc_drop"].cumsum().values
        ax.plot(x_t, baseline_acc - cum_t, label="Temporal only",
                linewidth=2, linestyle="--")

    # Spatial-only curve
    spatial_df = df[df["attn_type"] == "spatial"].sort_values("acc_drop", ascending=True)
    if len(spatial_df) > 0:
        x_s = np.arange(1, len(spatial_df) + 1) / len(spatial_df) * 100
        cum_s = spatial_df["acc_drop"].cumsum().values
        ax.plot(x_s, baseline_acc - cum_s, label="Spatial only",
                linewidth=2, linestyle=":")

    ax.axhline(y=baseline_acc, color="gray", linestyle="-.", alpha=0.5,
               label=f"Baseline ({baseline_acc:.4f})")
    ax.set_xlabel("Percentage of Heads Pruned (%)", fontsize=12)
    ax.set_ylabel("Estimated Accuracy", fontsize=12)
    ax.set_title("Cumulative Head Pruning Curve", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        path = config.output_dir / "cumulative_pruning.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Pruning curve saved to {path}")

    plt.close(fig)


def plot_layer_importance(df: pd.DataFrame, config: AblationConfig,
                          save: bool = True) -> None:
    """Plot per-layer average importance comparing temporal vs spatial."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for attn_type, marker in [("temporal", "o"), ("spatial", "s")]:
        subset = df[df["attn_type"] == attn_type]
        layer_means = subset.groupby("layer")["acc_drop"].mean()
        ax.plot(layer_means.index, layer_means.values,
                marker=marker, linewidth=2, markersize=8,
                label=f"{attn_type.capitalize()} (mean drop)")

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Mean Accuracy Drop", fontsize=12)
    ax.set_title("Average Head Importance by Layer", fontsize=14)
    ax.set_xticks(range(config.num_layers))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        path = config.output_dir / "layer_importance.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Layer importance plot saved to {path}")

    plt.close(fig)


def plot_temporal_vs_spatial_scatter(df: pd.DataFrame, config: AblationConfig,
                                     save: bool = True) -> None:
    """Scatter plot: average spatial vs temporal importance per layer."""
    fig, ax = plt.subplots(figsize=(8, 8))

    temporal_means = (df[df["attn_type"] == "temporal"]
                      .groupby("layer")["acc_drop"].mean())
    spatial_means = (df[df["attn_type"] == "spatial"]
                     .groupby("layer")["acc_drop"].mean())

    layers = sorted(set(temporal_means.index) & set(spatial_means.index))

    for layer in layers:
        ax.scatter(spatial_means[layer], temporal_means[layer],
                   s=100, zorder=5)
        ax.annotate(f"L{layer}",
                    (spatial_means[layer], temporal_means[layer]),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=10)

    # Diagonal line
    lim_max = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim_max], [0, lim_max], "k--", alpha=0.3, label="Equal importance")

    ax.set_xlabel("Mean Spatial Head Acc Drop", fontsize=12)
    ax.set_ylabel("Mean Temporal Head Acc Drop", fontsize=12)
    ax.set_title("Temporal vs Spatial Head Importance by Layer", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        path = config.output_dir / "temporal_vs_spatial_scatter.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Scatter plot saved to {path}")

    plt.close(fig)


def generate_all_plots(ablation_df: pd.DataFrame,
                       importance_df: Optional[pd.DataFrame],
                       config: AblationConfig,
                       baseline_acc: float) -> None:
    """Generate all visualization plots."""
    print("\n=== Generating Visualizations ===")

    plot_ablation_heatmaps(ablation_df, config)
    plot_cumulative_pruning(ablation_df, config, baseline_acc)
    plot_layer_importance(ablation_df, config)
    plot_temporal_vs_spatial_scatter(ablation_df, config)

    if importance_df is not None:
        plot_importance_heatmaps(importance_df, config)

    print(f"\nAll plots saved to {config.output_dir}/")
