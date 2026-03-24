"""Run head ablation on SSv2 dataset.

Uses the actual SSv2 labels for accuracy measurement alongside
prediction consistency metrics (flip rate, KL divergence, logit correlation).
"""

import json
import time
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from config import AblationConfig
from baseline import load_model_and_processor
from real_video_loader import create_dataloader_from_config
from ablation import make_ablation_hook
from gradient_importance import compute_gradient_importance
from visualize import (plot_ablation_heatmaps, plot_importance_heatmaps,
                       plot_cumulative_pruning, plot_layer_importance,
                       plot_temporal_vs_spatial_scatter)


@torch.no_grad()
def get_baseline_predictions(model, dataloader, device):
    """Get baseline predictions and logits for all videos."""
    all_logits = []
    all_preds = []
    model.eval()

    for pixel_values, labels in tqdm(dataloader, desc="Baseline predictions"):
        pixel_values = pixel_values.to(device)
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits.cpu()
        preds = logits.argmax(dim=-1)
        all_logits.append(logits)
        all_preds.append(preds)

    return torch.cat(all_logits, dim=0), torch.cat(all_preds, dim=0)


@torch.no_grad()
def evaluate_with_consistency(model, dataloader, baseline_preds,
                              baseline_logits, device):
    """Evaluate ablated model using prediction consistency metrics."""
    all_logits = []
    all_preds = []
    model.eval()

    for pixel_values, labels in dataloader:
        pixel_values = pixel_values.to(device)
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits.cpu()
        preds = logits.argmax(dim=-1)
        all_logits.append(logits)
        all_preds.append(preds)

    ablated_logits = torch.cat(all_logits, dim=0)
    ablated_preds = torch.cat(all_preds, dim=0)

    n = len(baseline_preds)

    # Prediction flip rate
    flips = (ablated_preds != baseline_preds).float().mean().item()

    # KL divergence (baseline || ablated)
    baseline_probs = F.softmax(baseline_logits, dim=-1)
    ablated_probs = F.softmax(ablated_logits, dim=-1)
    kl_div = F.kl_div(
        ablated_probs.log(), baseline_probs, reduction="batchmean"
    ).item()

    # Logit correlation (mean Pearson r across samples)
    bl = baseline_logits.numpy()
    ab = ablated_logits.numpy()
    corrs = []
    for i in range(n):
        r = np.corrcoef(bl[i], ab[i])[0, 1]
        corrs.append(r if not np.isnan(r) else 1.0)
    mean_corr = np.mean(corrs)

    return {
        "flip_rate": flips,
        "kl_divergence": kl_div,
        "logit_correlation": mean_corr,
    }


def run_ssv2_ablation(ssv2_root_dir: str, data_split: str = "val",
                      num_videos: int = 100, batch_size: int = 2,
                      model_name: str = None):
    """Full ablation pipeline on SSv2 videos."""
    print("=" * 60)
    print("  TimeSformer Head Ablation on SSv2")
    print("=" * 60)

    config_kwargs = dict(
        ssv2_root_dir=ssv2_root_dir,
        data_split=data_split,
        num_eval_videos=num_videos,
        batch_size=batch_size,
        output_dir=Path("outputs/ssv2_ablation"),
        device="cuda" if torch.cuda.is_available() else "cpu",
        grad_num_batches=10,
    )
    if model_name:
        config_kwargs["model_name"] = model_name
    config = AblationConfig(**config_kwargs)

    start_time = time.time()

    # Load model
    model, processor = load_model_and_processor(config)

    # Load SSv2 data (automatically subset to num_videos via config)
    print(f"\nLoading SSv2 videos ({data_split} split)...")
    dataloader = create_dataloader_from_config(config, processor=processor)

    # -- Phase 1: Baseline Predictions --
    print("\n" + "=" * 40)
    print("PHASE 1: Getting Baseline Predictions")
    print("=" * 40)
    baseline_logits, baseline_preds = get_baseline_predictions(
        model, dataloader, config.device
    )
    n_videos = len(baseline_preds)
    print(f"Baseline predictions for {n_videos} videos captured")

    # Show what the model predicts
    unique_preds, counts = torch.unique(baseline_preds, return_counts=True)
    print(f"Unique SSv2 classes predicted: {len(unique_preds)}")
    top_preds = sorted(zip(counts.tolist(), unique_preds.tolist()), reverse=True)[:5]
    for count, cls_id in top_preds:
        label = model.config.id2label.get(cls_id, f"class_{cls_id}")
        print(f"  {label}: {count} videos")

    # -- Phase 2: Targeted Ablation (layers 0, 3, 6, 9, 11) --
    print("\n" + "=" * 40)
    print("PHASE 2: Head Ablation Sweep")
    print("=" * 40)

    target_layers = [0, 3, 6, 9, 11]  # early, mid, late
    results = []
    total = len(target_layers) * 2 * config.num_heads
    count = 0

    for layer_idx in target_layers:
        for attn_type in ["temporal", "spatial"]:
            for head_idx in range(config.num_heads):
                count += 1
                encoder_layer = model.timesformer.encoder.layer[layer_idx]

                if attn_type == "temporal":
                    target_module = encoder_layer.temporal_attention.attention
                else:
                    target_module = encoder_layer.attention.attention

                hook = target_module.register_forward_hook(
                    make_ablation_hook(head_idx, config.num_heads, config.head_dim)
                )

                metrics = evaluate_with_consistency(
                    model, dataloader, baseline_preds,
                    baseline_logits, config.device
                )

                hook.remove()

                result = {
                    "layer": layer_idx,
                    "attn_type": attn_type,
                    "head": head_idx,
                    "flip_rate": metrics["flip_rate"],
                    "kl_divergence": metrics["kl_divergence"],
                    "logit_correlation": metrics["logit_correlation"],
                    "acc_drop": metrics["flip_rate"],
                }
                results.append(result)

                if count % 12 == 0 or count <= 3:
                    print(f"[{count}/{total}] L{layer_idx}-"
                          f"{attn_type[0].upper()}-H{head_idx} | "
                          f"flip={metrics['flip_rate']:.3f} "
                          f"KL={metrics['kl_divergence']:.4f} "
                          f"corr={metrics['logit_correlation']:.4f}")

    df = pd.DataFrame(results)
    csv_path = config.output_dir / "ablation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # -- Summary --
    print("\n" + "=" * 40)
    print("ABLATION SUMMARY (SSv2)")
    print("=" * 40)

    for attn_type in ["temporal", "spatial"]:
        subset = df[df["attn_type"] == attn_type]
        print(f"\n{attn_type.upper()} heads:")
        print(f"  Mean flip rate:      {subset['flip_rate'].mean():.4f}")
        print(f"  Max flip rate:       {subset['flip_rate'].max():.4f}")
        print(f"  Mean KL divergence:  {subset['kl_divergence'].mean():.4f}")
        print(f"  Mean logit corr:     {subset['logit_correlation'].mean():.4f}")
        top3 = subset.nlargest(3, "flip_rate")
        print(f"  Top-3 most impactful heads:")
        for _, row in top3.iterrows():
            print(f"    L{int(row['layer'])}-H{int(row['head'])}: "
                  f"flip={row['flip_rate']:.3f}, "
                  f"KL={row['kl_divergence']:.4f}")

    # -- Phase 3: Gradient Importance --
    print("\n" + "=" * 40)
    print("PHASE 3: Gradient-Based Importance")
    print("=" * 40)
    importance_df = compute_gradient_importance(
        model, dataloader, config, num_batches=config.grad_num_batches
    )

    # -- Phase 4: Visualizations --
    print("\n" + "=" * 40)
    print("PHASE 4: Generating Visualizations")
    print("=" * 40)

    plot_ablation_heatmaps(df, config, metric="flip_rate")
    plot_layer_importance(df, config)
    plot_temporal_vs_spatial_scatter(df, config)
    plot_cumulative_pruning(df, config, baseline_acc=1.0)
    plot_importance_heatmaps(importance_df, config)

    # -- Done --
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"COMPLETE — {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Tested {len(df)} heads on {n_videos} SSv2 videos")
    print("=" * 60)

    summary = {
        "dataset": f"SSv2 ({data_split} split)",
        "num_videos": n_videos,
        "layers_tested": target_layers,
        "heads_tested": len(df),
        "elapsed_seconds": elapsed,
        "temporal_mean_flip": df[df["attn_type"] == "temporal"]["flip_rate"].mean(),
        "spatial_mean_flip": df[df["attn_type"] == "spatial"]["flip_rate"].mean(),
        "temporal_mean_kl": df[df["attn_type"] == "temporal"]["kl_divergence"].mean(),
        "spatial_mean_kl": df[df["attn_type"] == "spatial"]["kl_divergence"].mean(),
    }
    with open(config.output_dir / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nAll outputs in: {config.output_dir.resolve()}")
    for p in sorted(config.output_dir.iterdir()):
        print(f"  {p.name} ({p.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run TimeSformer head ablation on SSv2 dataset"
    )
    parser.add_argument("--ssv2_root_dir", type=str, required=True,
                        help="Path to evaluation_frames/ directory")
    parser.add_argument("--data_split", type=str, default="val",
                        help="Dataset split (train, val, test)")
    parser.add_argument("--num_videos", type=int, default=100,
                        help="Number of videos to evaluate")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model name or local path (default: config.py default)")
    args = parser.parse_args()

    run_ssv2_ablation(
        ssv2_root_dir=args.ssv2_root_dir,
        data_split=args.data_split,
        num_videos=args.num_videos,
        batch_size=args.batch_size,
        model_name=args.model_name,
    )
