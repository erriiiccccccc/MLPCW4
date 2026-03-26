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
    all_logits = []
    all_preds = []
    model.eval()

    for pixel_values, labels in tqdm(dataloader, desc="Baseline predictions"):
        pixel_values = pixel_values.to(device)
        logits = model(pixel_values=pixel_values).logits.cpu()
        all_logits.append(logits)
        all_preds.append(logits.argmax(dim=-1))

    return torch.cat(all_logits, dim=0), torch.cat(all_preds, dim=0)


@torch.no_grad()
def evaluate_with_consistency(model, dataloader, baseline_preds,
                              baseline_logits, device):
    all_logits = []
    all_preds = []
    model.eval()

    for pixel_values, labels in dataloader:
        pixel_values = pixel_values.to(device)
        logits = model(pixel_values=pixel_values).logits.cpu()
        all_logits.append(logits)
        all_preds.append(logits.argmax(dim=-1))

    ablated_logits = torch.cat(all_logits, dim=0)
    ablated_preds = torch.cat(all_preds, dim=0)

    n = len(baseline_preds)

    flips = (ablated_preds != baseline_preds).float().mean().item()

    baseline_probs = F.softmax(baseline_logits, dim=-1)
    ablated_probs = F.softmax(ablated_logits, dim=-1)
    kl_div = F.kl_div(
        ablated_probs.log(), baseline_probs, reduction="batchmean"
    ).item()

    bl = baseline_logits.numpy()
    ab = ablated_logits.numpy()
    corrs = []
    for i in range(n):
        r = np.corrcoef(bl[i], ab[i])[0, 1]
        corrs.append(r if not np.isnan(r) else 1.0)

    return {
        "flip_rate": flips,
        "kl_divergence": kl_div,
        "logit_correlation": np.mean(corrs),
    }


def run_ssv2_ablation(ssv2_root_dir, data_split="val",
                      num_videos=100, batch_size=2, model_name=None):
    print("TimeSformer Head Ablation on SSv2")

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

    model, processor = load_model_and_processor(config)

    print(f"\nLoading SSv2 videos ({data_split} split)...")
    dataloader = create_dataloader_from_config(config, processor=processor)

    print("\nPHASE 1: Getting Baseline Predictions")
    baseline_logits, baseline_preds = get_baseline_predictions(
        model, dataloader, config.device
    )
    n_videos = len(baseline_preds)
    print(f"Baseline predictions for {n_videos} videos captured")

    unique_preds, counts = torch.unique(baseline_preds, return_counts=True)
    print(f"Unique SSv2 classes predicted: {len(unique_preds)}")
    top_preds = sorted(zip(counts.tolist(), unique_preds.tolist()), reverse=True)[:5]
    for count, cls_id in top_preds:
        label = model.config.id2label.get(cls_id, f"class_{cls_id}")
        print(f"  {label}: {count} videos")

    print("\nPHASE 2: Head Ablation Sweep")

    target_layers = [0, 3, 6, 9, 11]
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

                results.append({
                    "layer": layer_idx,
                    "attn_type": attn_type,
                    "head": head_idx,
                    "flip_rate": metrics["flip_rate"],
                    "kl_divergence": metrics["kl_divergence"],
                    "logit_correlation": metrics["logit_correlation"],
                    "acc_drop": metrics["flip_rate"],
                })

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

    print("\nABLATION SUMMARY (SSv2)")

    for attn_type in ["temporal", "spatial"]:
        subset = df[df["attn_type"] == attn_type]
        print(f"\n{attn_type.upper()} heads:")
        print(f"  Mean flip rate:      {subset['flip_rate'].mean():.4f}")
        print(f"  Max flip rate:       {subset['flip_rate'].max():.4f}")
        print(f"  Mean KL divergence:  {subset['kl_divergence'].mean():.4f}")
        print(f"  Mean logit corr:     {subset['logit_correlation'].mean():.4f}")
        top3 = subset.nlargest(3, "flip_rate")
        print(f"  Top-3 most impactful:")
        for _, row in top3.iterrows():
            print(f"    L{int(row['layer'])}-H{int(row['head'])}: "
                  f"flip={row['flip_rate']:.3f}, "
                  f"KL={row['kl_divergence']:.4f}")

    print("\nPHASE 3: Gradient-Based Importance")
    importance_df = compute_gradient_importance(
        model, dataloader, config, num_batches=config.grad_num_batches
    )

    print("\nPHASE 4: Generating Visualizations")

    plot_ablation_heatmaps(df, config, metric="flip_rate")
    plot_layer_importance(df, config)
    plot_temporal_vs_spatial_scatter(df, config)
    plot_cumulative_pruning(df, config, baseline_acc=1.0)
    plot_importance_heatmaps(importance_df, config)

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Tested {len(df)} heads on {n_videos} SSv2 videos")

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssv2_root_dir", type=str, required=True)
    parser.add_argument("--data_split", type=str, default="val")
    parser.add_argument("--num_videos", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--model_name", type=str, default=None)
    args = parser.parse_args()

    run_ssv2_ablation(
        ssv2_root_dir=args.ssv2_root_dir,
        data_split=args.data_split,
        num_videos=args.num_videos,
        batch_size=args.batch_size,
        model_name=args.model_name,
    )
