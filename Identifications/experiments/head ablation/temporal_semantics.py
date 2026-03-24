"""temporal semantic analysis of attention heads in TimeSformer.

extracts temporal attn patterns and tries to understand what each head is
actually doing - does it focus on early frames? late? is it specialised at all?

analyses:
    1. attn pattern clustering (kmeans on flattened attention maps)
    2. temporal receptive field (which frames each head attends to)
    3. head specialization score (entropy - low means specialised)
    4. per-class importance grouped by SSv2 action categories
    5. temporal attention distance
"""

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


# ---------------------------------------------------------------------------
# Temporal attention extraction
# ---------------------------------------------------------------------------

def _force_temporal_output_attentions(model, num_layers: int):
    """monkey-patch temporal attn to always output attn probs.

    huggingface doesnt pass output_attentions to temporal_attention (only spatial),
    so hooks cant see attn weights. we wrap each module's forward to force it.
    returns list of (module, orig_forward) so you can undo it after.
    """
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
    """Undo the monkey-patches applied by _force_temporal_output_attentions."""
    for module, orig_forward in originals:
        module.forward = orig_forward


@torch.no_grad()
def extract_temporal_attention(
    model,
    dataloader,
    config: AblationConfig,
) -> Dict[int, np.ndarray]:
    """Extract temporal attention weights from all layers across all videos.

    Returns:
        Dict mapping layer_idx -> attention array of shape
        (num_videos_total, num_heads, T, T) averaged over spatial locations.
    """
    num_layers = config.num_layers
    num_heads = config.num_heads
    T = config.num_frames  # 8

    # Storage: accumulate across batches
    layer_attns: Dict[int, List[np.ndarray]] = {i: [] for i in range(num_layers)}
    hooks = []
    attn_store: Dict[int, torch.Tensor] = {}

    def make_attn_hook(layer_idx: int):
        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) > 1:
                # output[1] = attention probs: (B*num_spatial, num_heads, T, T)
                attn_store[layer_idx] = output[1].detach().cpu()
        return hook_fn

    # Force temporal attention modules to output attention probs
    patched = _force_temporal_output_attentions(model, num_layers)

    # Register hooks
    for i in range(num_layers):
        layer = model.timesformer.encoder.layer[i]
        h = layer.temporal_attention.attention.register_forward_hook(
            make_attn_hook(i)
        )
        hooks.append(h)

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
                # attn shape: (batch*num_spatial, num_heads, T, T)
                attn = attn_store[layer_idx]
                # Compute num_spatial from actual tensor shape
                num_spatial = attn.shape[0] // batch_size
                # Infer T from actual attention shape (robust to CLS token)
                T_actual = attn.shape[2]
                # Reshape to (batch, num_spatial, num_heads, T, T)
                attn = attn.view(batch_size, num_spatial, num_heads, T_actual, T_actual)
                # Average over spatial locations -> (batch, num_heads, T, T)
                attn_avg = attn.mean(dim=1).numpy()
                layer_attns[layer_idx].append(attn_avg)
    finally:
        # Always clean up: remove hooks and restore original forwards
        for h in hooks:
            h.remove()
        _restore_temporal_forwards(patched)

    # Concatenate across batches
    result = {}
    for layer_idx in range(num_layers):
        if layer_attns[layer_idx]:
            result[layer_idx] = np.concatenate(layer_attns[layer_idx], axis=0)

    if not result:
        print("  WARNING: No temporal attention captured. Check model compatibility.")

    return result


# ---------------------------------------------------------------------------
# Analysis 1: Attention pattern clustering
# ---------------------------------------------------------------------------

def cluster_attention_patterns(
    layer_attns: Dict[int, np.ndarray],
    n_clusters: int = 4,
) -> pd.DataFrame:
    """Cluster temporal heads by their attention patterns using K-means.

    For each head, the feature vector is the mean attention map (TxT)
    flattened. Clustering reveals groups of heads with similar behaviour.

    Returns:
        DataFrame with columns: layer, head, cluster, pattern_type.
    """
    rows = []
    T = None

    for layer_idx in sorted(layer_attns.keys()):
        attn = layer_attns[layer_idx]  # (N, H, T, T)
        N, H, T, _ = attn.shape

        # Mean attention per head -> (H, T, T)
        mean_attn = attn.mean(axis=0)

        # Flatten for clustering -> (H, T*T)
        features = mean_attn.reshape(H, -1)

        # K-means clustering
        n_clust = min(n_clusters, H)
        km = KMeans(n_clusters=n_clust, n_init=10, random_state=42)
        labels = km.fit_predict(features)

        for head_idx in range(H):
            pattern_type = _classify_attention_pattern(mean_attn[head_idx])
            rows.append({
                "layer": layer_idx,
                "head": head_idx,
                "cluster": int(labels[head_idx]),
                "pattern_type": pattern_type,
            })

    return pd.DataFrame(rows)


def _classify_attention_pattern(attn_map: np.ndarray) -> str:
    """Classify a TxT attention pattern into a semantic type.

    Types (inspired by Kovaleva et al. 2019 for temporal domain):
        - diagonal: attends primarily to the same frame index
        - forward: attends to later frames
        - backward: attends to earlier frames
        - uniform: roughly equal attention across frames
        - local: attends to adjacent frames (±1)
        - long_range: attends to distant frames
    """
    T = attn_map.shape[0]

    # Diagonal dominance: mean of diagonal vs mean of off-diagonal
    diag_mean = np.diag(attn_map).mean()
    off_diag_mask = ~np.eye(T, dtype=bool)
    off_diag_mean = attn_map[off_diag_mask].mean()
    diag_ratio = diag_mean / (off_diag_mean + 1e-8)

    if diag_ratio > 3.0:
        return "diagonal"

    # Compute mean temporal offset per query
    offsets = np.arange(T)[None, :] - np.arange(T)[:, None]  # (T, T)
    mean_offset = (attn_map * offsets).sum(axis=1).mean()

    # Adjacent frame dominance (±1)
    local_mask = np.abs(offsets) <= 1
    local_weight = attn_map[local_mask].sum() / attn_map.sum()

    # Entropy of attention (uniform → high entropy)
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


# ---------------------------------------------------------------------------
# Analysis 2: Temporal receptive field
# ---------------------------------------------------------------------------

def compute_temporal_receptive_field(
    layer_attns: Dict[int, np.ndarray],
) -> pd.DataFrame:
    """For each head, compute which frames it attends to most.

    Returns:
        DataFrame with columns: layer, head, frame_0..frame_T-1 (column attention),
        peak_frame, attention_span.
    """
    rows = []

    for layer_idx in sorted(layer_attns.keys()):
        attn = layer_attns[layer_idx]  # (N, H, T, T)
        T = attn.shape[2]

        # Mean attention per head: (H, T, T)
        mean_attn = attn.mean(axis=0)

        for head_idx in range(mean_attn.shape[0]):
            head_attn = mean_attn[head_idx]  # (T, T)

            # Column-wise mean: how much each frame is attended to (averaged
            # across all query frames)
            col_attn = head_attn.mean(axis=0)  # (T,)

            row = {
                "layer": layer_idx,
                "head": head_idx,
            }
            for t in range(T):
                row[f"frame_{t}"] = float(col_attn[t])

            row["peak_frame"] = int(np.argmax(col_attn))

            # Attention span: std of the attention distribution across frames
            # (higher = more spread out)
            frames = np.arange(T, dtype=float)
            col_norm = col_attn / (col_attn.sum() + 1e-8)
            mean_frame = (col_norm * frames).sum()
            var_frame = (col_norm * (frames - mean_frame) ** 2).sum()
            row["attention_span"] = float(np.sqrt(var_frame))
            row["mean_attended_frame"] = float(mean_frame)

            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis 3: Head specialization score (entropy)
# ---------------------------------------------------------------------------

def compute_specialization_scores(
    layer_attns: Dict[int, np.ndarray],
) -> pd.DataFrame:
    """Compute specialization score for each temporal head.

    Low entropy = specialized (attends to specific frames).
    High entropy = distributed (uniform attention).

    Returns:
        DataFrame with columns: layer, head, entropy, normalized_entropy,
        specialization_score, diagonal_dominance.
    """
    rows = []

    for layer_idx in sorted(layer_attns.keys()):
        attn = layer_attns[layer_idx]  # (N, H, T, T)
        T = attn.shape[2]
        mean_attn = attn.mean(axis=0)  # (H, T, T)

        max_entropy = np.log(T)  # max entropy for T-class distribution

        for head_idx in range(mean_attn.shape[0]):
            head_attn = mean_attn[head_idx]  # (T, T)

            # Compute entropy per query frame, then average
            entropies = []
            for t in range(T):
                row_dist = head_attn[t]
                row_dist = row_dist / (row_dist.sum() + 1e-8)
                h = -np.sum(row_dist * np.log(row_dist + 1e-8))
                entropies.append(h)
            mean_entropy = np.mean(entropies)
            norm_entropy = mean_entropy / max_entropy

            # Diagonal dominance
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


# ---------------------------------------------------------------------------
# Analysis 4: Per-class head importance
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_per_class_importance(
    model,
    dataloader,
    baseline_preds: torch.Tensor,
    config: AblationConfig,
    taxonomy_path: str = "research/ssv2_class_taxonomy.json",
) -> pd.DataFrame:
    """Compute head importance grouped by SSv2 action categories.

    For each (layer, head), ablate and measure flip rate. Group flipped
    samples by predicted class category.

    Returns:
        DataFrame with columns: layer, head, category, flip_rate, n_samples.
    """
    # Load taxonomy
    with open(taxonomy_path) as f:
        taxonomy = json.load(f)

    # Build class_id -> category mapping
    class_to_cat = {}
    for cat_name, cat_data in taxonomy["categories"].items():
        for cid in cat_data["class_ids"]:
            class_to_cat[cid] = cat_name

    rows = []
    target_layers = [0, 3, 6, 9, 11]  # representative layers

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
                    outputs = model(pixel_values=pixel_values)
                    preds = outputs.logits.argmax(dim=-1).cpu()
                    ablated_preds_list.append(preds)
            finally:
                hook.remove()

            ablated_preds = torch.cat(ablated_preds_list, dim=0)
            flips = (ablated_preds != baseline_preds)

            # Group by predicted category
            cat_flips: Dict[str, List[bool]] = {}
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


# ---------------------------------------------------------------------------
# Analysis 5: Attention temporal distance
# ---------------------------------------------------------------------------

def compute_temporal_distance(
    layer_attns: Dict[int, np.ndarray],
) -> pd.DataFrame:
    """Compute average temporal distance of attention for each head.

    Measures whether heads attend to nearby (local motion) or distant
    (long-range temporal) frames.

    Returns:
        DataFrame with columns: layer, head, mean_distance, max_distance,
        local_ratio (fraction within ±1 frame).
    """
    rows = []

    for layer_idx in sorted(layer_attns.keys()):
        attn = layer_attns[layer_idx]  # (N, H, T, T)
        T = attn.shape[2]
        mean_attn = attn.mean(axis=0)  # (H, T, T)

        # Distance matrix: |i - j| for frames i, j
        dist_matrix = np.abs(
            np.arange(T)[:, None] - np.arange(T)[None, :]
        ).astype(float)

        local_mask = dist_matrix <= 1.0

        for head_idx in range(mean_attn.shape[0]):
            head_attn = mean_attn[head_idx]  # (T, T)
            # Normalize per query
            head_norm = head_attn / (head_attn.sum(axis=1, keepdims=True) + 1e-8)

            mean_dist = (head_norm * dist_matrix).sum() / T
            max_dist = float(
                (head_norm * dist_matrix).sum(axis=1).max()
            )
            local_ratio = float(
                (head_norm * local_mask).sum() / T
            )

            rows.append({
                "layer": layer_idx,
                "head": head_idx,
                "mean_distance": float(mean_dist),
                "max_distance": max_dist,
                "local_ratio": local_ratio,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Combined analysis pipeline
# ---------------------------------------------------------------------------

def run_temporal_semantics(
    num_videos: int = 20,
    batch_size: int = 2,
    output_dir: str = "outputs/temporal_analysis",
    ssv2_root_dir: Optional[str] = None,
    data_split: str = "val",
    model_name: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Run all temporal semantic analyses.

    Returns:
        Dict mapping analysis name -> results DataFrame.
    """
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

    print("=" * 60)
    print("  Temporal Semantic Analysis — TimeSformer")
    print("=" * 60)

    start_time = time.time()

    # Load model + data (automatically subset to num_videos via config)
    model, processor = load_model_and_processor(config)
    dataloader = create_dataloader_from_config(config, processor=processor)

    # Baseline predictions (for per-class importance)
    print("\nCollecting baseline predictions...")
    all_preds = []
    model.eval()
    with torch.no_grad():
        for pixel_values, _labels in tqdm(dataloader, desc="Baseline"):
            pixel_values = pixel_values.to(config.device)
            outputs = model(pixel_values=pixel_values)
            all_preds.append(outputs.logits.argmax(dim=-1).cpu())
    baseline_preds = torch.cat(all_preds, dim=0)

    # Extract attention weights
    print("\nExtracting temporal attention patterns...")
    layer_attns = extract_temporal_attention(model, dataloader, config)
    for layer_idx, attn in layer_attns.items():
        print(f"  L{layer_idx}: {attn.shape}")
        # Save raw attention patterns for downstream phases (viz, recommendations)
        np.save(output_path / f"attention_L{layer_idx}.npy", attn)
    print(f"  Saved .npy attention files to {output_path}")

    results: Dict[str, pd.DataFrame] = {}

    # Analysis 1: Clustering
    print("\n--- Attention Pattern Clustering ---")
    cluster_df = cluster_attention_patterns(layer_attns)
    cluster_df.to_csv(output_path / "attention_clusters.csv", index=False)
    results["clusters"] = cluster_df
    for ptype, count in cluster_df["pattern_type"].value_counts().items():
        print(f"  {ptype}: {count} heads")

    # Analysis 2: Temporal receptive field
    print("\n--- Temporal Receptive Field ---")
    trf_df = compute_temporal_receptive_field(layer_attns)
    trf_df.to_csv(output_path / "temporal_receptive_field.csv", index=False)
    results["receptive_field"] = trf_df
    print(f"  Peak frame distribution: {trf_df['peak_frame'].value_counts().to_dict()}")

    # Analysis 3: Specialization scores
    print("\n--- Head Specialization Scores ---")
    spec_df = compute_specialization_scores(layer_attns)
    spec_df.to_csv(output_path / "specialization_scores.csv", index=False)
    results["specialization"] = spec_df
    most_specialized = spec_df.nlargest(5, "specialization_score")
    print("  Top-5 most specialized heads:")
    for _, row in most_specialized.iterrows():
        print(f"    L{int(row['layer'])}-H{int(row['head'])}: "
              f"spec={row['specialization_score']:.3f}, "
              f"diag={row['diagonal_dominance']:.2f}")

    # Analysis 4: Per-class importance
    print("\n--- Per-Class Head Importance ---")
    taxonomy_path = Path("research/ssv2_class_taxonomy.json")
    if taxonomy_path.exists():
        class_df = compute_per_class_importance(
            model, dataloader, baseline_preds, config,
            str(taxonomy_path),
        )
        class_df.to_csv(output_path / "per_class_importance.csv", index=False)
        results["per_class"] = class_df
        print(f"  Categories: {class_df['category'].nunique()}")
    else:
        print("  Skipped — taxonomy file not found")

    # Analysis 5: Temporal distance
    print("\n--- Attention Temporal Distance ---")
    dist_df = compute_temporal_distance(layer_attns)
    dist_df.to_csv(output_path / "temporal_distance.csv", index=False)
    results["distance"] = dist_df
    print(f"  Mean distance range: "
          f"{dist_df['mean_distance'].min():.2f} - "
          f"{dist_df['mean_distance'].max():.2f}")

    # Combined temporal semantics CSV
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
    print(f"\n{'=' * 60}")
    print(f"TEMPORAL SEMANTICS COMPLETE — {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 60}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Temporal semantic analysis of TimeSformer attention heads"
    )
    parser.add_argument("--num_videos", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output_dir", type=str,
                        default="outputs/temporal_analysis")
    parser.add_argument("--ssv2_root_dir", type=str, required=True,
                        help="Path to evaluation_frames/ directory")
    parser.add_argument("--data_split", type=str, default="val")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model name or local path (default: config.py default)")
    args = parser.parse_args()

    run_temporal_semantics(
        num_videos=args.num_videos,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        ssv2_root_dir=args.ssv2_root_dir,
        data_split=args.data_split,
        model_name=args.model_name,
    )
