import json
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional
from sklearn.cluster import KMeans

from config import AblationConfig
from baseline import load_model_and_processor
from real_video_loader import create_dataloader_from_config
from ablation import make_ablation_hook


def _force_temporal_output_attentions(model, num_layers):
    originals = []
    for i in range(num_layers):
        module = model.timesformer.encoder.layer[i].temporal_attention.attention
        orig_forward = module.forward

        def _make_patched(orig):
            def patched_forward(hidden_states, output_attentions=False):
                return orig(hidden_states, output_attentions=True)
            return patched_forward

        originals.append((module, orig_forward))
        module.forward = _make_patched(orig_forward)
    return originals


def _restore_temporal_forwards(originals):
    for module, orig_forward in originals:
        module.forward = orig_forward


@torch.no_grad()
def extract_temporal_attention(model, dataloader, config):
    """returns dict: layer_idx -> attention array (N, H, T, T) avg over spatial locs"""
    num_layers = config.num_layers
    num_heads = config.num_heads
    T = config.num_frames

    layer_attns = {i: [] for i in range(num_layers)}
    hooks = []
    attn_store = {}

    def make_attn_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) > 1:
                attn_store[layer_idx] = output[1].detach().cpu()
        return hook_fn

    patched = _force_temporal_output_attentions(model, num_layers)

    for i in range(num_layers):
        layer = model.timesformer.encoder.layer[i]
        hooks.append(layer.temporal_attention.attention.register_forward_hook(
            make_attn_hook(i)
        ))

    model.eval()

    try:
        for pixel_values, _labels in tqdm(dataloader, desc="Extracting attention"):
            pixel_values = pixel_values.to(config.device)
            attn_store.clear()
            _ = model(pixel_values=pixel_values)

            batch_size = pixel_values.shape[0]

            for layer_idx in range(num_layers):
                if layer_idx not in attn_store:
                    continue
                attn = attn_store[layer_idx]
                num_spatial = attn.shape[0] // batch_size
                T_actual = attn.shape[2]
                attn = attn.view(batch_size, num_spatial, num_heads, T_actual, T_actual)
                layer_attns[layer_idx].append(attn.mean(dim=1).numpy())
    finally:
        for h in hooks:
            h.remove()
        _restore_temporal_forwards(patched)

    result = {}
    for layer_idx in range(num_layers):
        if layer_attns[layer_idx]:
            result[layer_idx] = np.concatenate(layer_attns[layer_idx], axis=0)

    if not result:
        print("  WARNING: No temporal attention captured. Check model compatibility.")

    return result


def cluster_attention_patterns(layer_attns, n_clusters=4):
    rows = []

    for layer_idx in sorted(layer_attns.keys()):
        attn = layer_attns[layer_idx]
        N, H, T, _ = attn.shape

        mean_attn = attn.mean(axis=0)
        features = mean_attn.reshape(H, -1)
        n_clust = min(n_clusters, H)
        km = KMeans(n_clusters=n_clust, n_init=10, random_state=42)
        labels = km.fit_predict(features)

        for head_idx in range(H):
            rows.append({
                "layer": layer_idx,
                "head": head_idx,
                "cluster": int(labels[head_idx]),
                "pattern_type": _classify_attention_pattern(mean_attn[head_idx]),
            })

    return pd.DataFrame(rows)


def _classify_attention_pattern(attn_map):
    T = attn_map.shape[0]

    diag_mean = np.diag(attn_map).mean()
    off_diag_mask = ~np.eye(T, dtype=bool)
    off_diag_mean = attn_map[off_diag_mask].mean()
    diag_ratio = diag_mean / (off_diag_mean + 1e-8)

    if diag_ratio > 3.0:
        return "diagonal"

    offsets = np.arange(T)[None, :] - np.arange(T)[:, None]
    mean_offset = (attn_map * offsets).sum(axis=1).mean()

    local_mask = np.abs(offsets) <= 1
    local_weight = attn_map[local_mask].sum() / attn_map.sum()

    flat = attn_map.flatten()
    flat = flat / (flat.sum() + 1e-8)
    entropy = -np.sum(flat * np.log(flat + 1e-8))
    max_entropy = np.log(T * T)
    norm_entropy = entropy / max_entropy

    if norm_entropy > 0.95:
        return "uniform"
    if local_weight > 0.7:
        return "local"
    if mean_offset > 0.5:
        return "forward"
    if mean_offset < -0.5:
        return "backward"
    return "long_range"


def compute_temporal_receptive_field(layer_attns):
    rows = []

    for layer_idx in sorted(layer_attns.keys()):
        attn = layer_attns[layer_idx]
        T = attn.shape[2]
        mean_attn = attn.mean(axis=0)

        for head_idx in range(mean_attn.shape[0]):
            head_attn = mean_attn[head_idx]

            # column-wise mean: how much each frame is attended to
            col_attn = head_attn.mean(axis=0)

            row = {"layer": layer_idx, "head": head_idx}
            for t in range(T):
                row[f"frame_{t}"] = float(col_attn[t])

            row["peak_frame"] = int(np.argmax(col_attn))

            frames = np.arange(T, dtype=float)
            col_norm = col_attn / (col_attn.sum() + 1e-8)
            mean_frame = (col_norm * frames).sum()
            var_frame = (col_norm * (frames - mean_frame) ** 2).sum()
            row["attention_span"] = float(np.sqrt(var_frame))
            row["mean_attended_frame"] = float(mean_frame)

            rows.append(row)

    return pd.DataFrame(rows)


def compute_specialization_scores(layer_attns):
    """low entropy = specialized, high entropy = distributed"""
    rows = []

    for layer_idx in sorted(layer_attns.keys()):
        attn = layer_attns[layer_idx]
        T = attn.shape[2]
        mean_attn = attn.mean(axis=0)
        max_entropy = np.log(T)

        for head_idx in range(mean_attn.shape[0]):
            head_attn = mean_attn[head_idx]

            entropies = []
            for t in range(T):
                row_dist = head_attn[t]
                row_dist = row_dist / (row_dist.sum() + 1e-8)
                h = -np.sum(row_dist * np.log(row_dist + 1e-8))
                entropies.append(h)
            mean_entropy = np.mean(entropies)
            norm_entropy = mean_entropy / max_entropy

            diag_mean = np.diag(head_attn).mean()
            off_diag_mask = ~np.eye(T, dtype=bool)
            off_diag_mean = head_attn[off_diag_mask].mean()

            rows.append({
                "layer": layer_idx,
                "head": head_idx,
                "entropy": mean_entropy,
                "normalized_entropy": norm_entropy,
                "specialization_score": 1.0 - norm_entropy,
                "diagonal_dominance": float(diag_mean / (off_diag_mean + 1e-8)),
            })

    return pd.DataFrame(rows)


@torch.no_grad()
def compute_per_class_importance(
    model, dataloader, baseline_preds, config,
    taxonomy_path="research/ssv2_class_taxonomy.json",
):
    with open(taxonomy_path) as f:
        taxonomy = json.load(f)

    class_to_cat = {}
    for cat_name, cat_data in taxonomy["categories"].items():
        for cid in cat_data["class_ids"]:
            class_to_cat[cid] = cat_name

    rows = []
    target_layers = [0, 3, 6, 9, 11]

    for layer_idx in target_layers:
        for head_idx in range(config.num_heads):
            encoder_layer = model.timesformer.encoder.layer[layer_idx]
            target_module = encoder_layer.temporal_attention.attention

            hook = target_module.register_forward_hook(
                make_ablation_hook(head_idx, config.num_heads, config.head_dim)
            )

            ablated_preds_list = []
            model.eval()

            try:
                for pixel_values, _labels in dataloader:
                    pixel_values = pixel_values.to(config.device)
                    preds = model(pixel_values=pixel_values).logits.argmax(dim=-1).cpu()
                    ablated_preds_list.append(preds)
            finally:
                hook.remove()

            ablated_preds = torch.cat(ablated_preds_list, dim=0)
            flips = (ablated_preds != baseline_preds)

            cat_flips = {}
            for i in range(len(baseline_preds)):
                pred_class = baseline_preds[i].item()
                cat = class_to_cat.get(pred_class, "unknown")
                if cat not in cat_flips:
                    cat_flips[cat] = []
                cat_flips[cat].append(flips[i].item())

            for cat, flip_list in cat_flips.items():
                rows.append({
                    "layer": layer_idx,
                    "head": head_idx,
                    "category": cat,
                    "flip_rate": np.mean(flip_list),
                    "n_samples": len(flip_list),
                })

    return pd.DataFrame(rows)


def compute_temporal_distance(layer_attns):
    """measures whether heads attend to nearby or distant frames"""
    rows = []

    for layer_idx in sorted(layer_attns.keys()):
        attn = layer_attns[layer_idx]
        T = attn.shape[2]
        mean_attn = attn.mean(axis=0)

        dist_matrix = np.abs(
            np.arange(T)[:, None] - np.arange(T)[None, :]
        ).astype(float)
        local_mask = dist_matrix <= 1.0

        for head_idx in range(mean_attn.shape[0]):
            head_attn = mean_attn[head_idx]
            head_norm = head_attn / (head_attn.sum(axis=1, keepdims=True) + 1e-8)

            mean_dist = (head_norm * dist_matrix).sum() / T
            max_dist = float((head_norm * dist_matrix).sum(axis=1).max())
            local_ratio = float((head_norm * local_mask).sum() / T)

            rows.append({
                "layer": layer_idx,
                "head": head_idx,
                "mean_distance": float(mean_dist),
                "max_distance": max_dist,
                "local_ratio": local_ratio,
            })

    return pd.DataFrame(rows)


def run_temporal_semantics(
    num_videos=20, batch_size=2,
    output_dir="outputs/temporal_analysis",
    ssv2_root_dir=None, data_split="val",
    model_name=None,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config_kwargs = dict(
        num_eval_videos=num_videos,
        batch_size=batch_size,
        output_dir=output_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        ssv2_root_dir=ssv2_root_dir,
        data_split=data_split,
    )
    if model_name:
        config_kwargs["model_name"] = model_name
    config = AblationConfig(**config_kwargs)

    print("Temporal Semantic Analysis")

    start_time = time.time()

    model, processor = load_model_and_processor(config)
    dataloader = create_dataloader_from_config(config, processor=processor)

    print("\nCollecting baseline predictions...")
    all_preds = []
    model.eval()
    with torch.no_grad():
        for pixel_values, _labels in tqdm(dataloader, desc="Baseline"):
            pixel_values = pixel_values.to(config.device)
            all_preds.append(model(pixel_values=pixel_values).logits.argmax(dim=-1).cpu())
    baseline_preds = torch.cat(all_preds, dim=0)

    print("\nExtracting temporal attention patterns...")
    layer_attns = extract_temporal_attention(model, dataloader, config)
    for layer_idx, attn in layer_attns.items():
        print(f"  L{layer_idx}: {attn.shape}")
        np.save(output_path / f"attention_L{layer_idx}.npy", attn)
    print(f"  Saved .npy attention files to {output_path}")

    results = {}

    print("\n--- Attention Pattern Clustering ---")
    cluster_df = cluster_attention_patterns(layer_attns)
    cluster_df.to_csv(output_path / "attention_clusters.csv", index=False)
    results["clusters"] = cluster_df
    for ptype, count in cluster_df["pattern_type"].value_counts().items():
        print(f"  {ptype}: {count} heads")

    print("\n--- Temporal Receptive Field ---")
    trf_df = compute_temporal_receptive_field(layer_attns)
    trf_df.to_csv(output_path / "temporal_receptive_field.csv", index=False)
    results["receptive_field"] = trf_df
    print(f"  Peak frame distribution: {trf_df['peak_frame'].value_counts().to_dict()}")

    print("\n--- Head Specialization Scores ---")
    spec_df = compute_specialization_scores(layer_attns)
    spec_df.to_csv(output_path / "specialization_scores.csv", index=False)
    results["specialization"] = spec_df
    most_specialized = spec_df.nlargest(5, "specialization_score")
    print("  Top-5 most specialized:")
    for _, row in most_specialized.iterrows():
        print(f"    L{int(row['layer'])}-H{int(row['head'])}: "
              f"spec={row['specialization_score']:.3f}, "
              f"diag={row['diagonal_dominance']:.2f}")

    print("\n--- Per-Class Head Importance ---")
    taxonomy_path = Path("research/ssv2_class_taxonomy.json")
    if taxonomy_path.exists():
        class_df = compute_per_class_importance(
            model, dataloader, baseline_preds, config, str(taxonomy_path),
        )
        class_df.to_csv(output_path / "per_class_importance.csv", index=False)
        results["per_class"] = class_df
        print(f"  Categories: {class_df['category'].nunique()}")
    else:
        print("  Skipped -- taxonomy file not found")

    print("\n--- Attention Temporal Distance ---")
    dist_df = compute_temporal_distance(layer_attns)
    dist_df.to_csv(output_path / "temporal_distance.csv", index=False)
    results["distance"] = dist_df
    print(f"  Mean distance range: "
          f"{dist_df['mean_distance'].min():.2f} - "
          f"{dist_df['mean_distance'].max():.2f}")

    combined = spec_df.merge(
        dist_df, on=["layer", "head"]
    ).merge(
        cluster_df[["layer", "head", "cluster", "pattern_type"]],
        on=["layer", "head"],
    ).merge(
        trf_df[["layer", "head", "peak_frame", "attention_span", "mean_attended_frame"]],
        on=["layer", "head"],
    )
    combined.to_csv(output_path / "temporal_semantics.csv", index=False)
    results["combined"] = combined

    elapsed = time.time() - start_time
    print(f"\nTEMPORAL SEMANTICS done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_videos", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="outputs/temporal_analysis")
    parser.add_argument("--ssv2_root_dir", type=str, required=True)
    parser.add_argument("--data_split", type=str, default="val")
    parser.add_argument("--model_name", type=str, default=None)
    args = parser.parse_args()

    run_temporal_semantics(
        num_videos=args.num_videos,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        ssv2_root_dir=args.ssv2_root_dir,
        data_split=args.data_split,
        model_name=args.model_name,
    )
