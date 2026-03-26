"""Measure how an ablated head changes downstream layers."""

import json
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

from config import AblationConfig
from baseline import load_model_and_processor
from real_video_loader import create_dataloader_from_config
from ablation import make_ablation_hook


def _make_hidden_state_hook(store: Dict[int, torch.Tensor], layer_idx: int):
    """Capture hidden states at a given layer."""
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            store[layer_idx] = output[0].detach().cpu()
        else:
            store[layer_idx] = output.detach().cpu()
    return hook_fn


def _make_attn_capture_hook(store: Dict[int, torch.Tensor], layer_idx: int):
    """Capture temporal attention probabilities from a self-attention module.

    Hooks on model.timesformer.encoder.layer[L].temporal_attention.attention.
    Output[1] contains the attention probs when output_attentions=True.
    """
    def hook_fn(module, input, output):
        if isinstance(output, tuple) and len(output) > 1:
            store[layer_idx] = output[1].detach().cpu()
    return hook_fn


def _force_temporal_output_attentions(model, num_layers: int):
    """Patch temporal attention modules to always return attention probs."""
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
    """Undo monkey-patches from _force_temporal_output_attentions."""
    for module, orig_forward in originals:
        module.forward = orig_forward


@torch.no_grad()
def _capture_forward_pass(
    model,
    pixel_values: torch.Tensor,
    num_layers: int,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor],
           Dict[int, torch.Tensor]]:
    """Run a forward pass and capture hidden states plus attention tensors."""
    hidden_store: Dict[int, torch.Tensor] = {}
    temporal_attn_store: Dict[int, torch.Tensor] = {}
    spatial_attn_store: Dict[int, torch.Tensor] = {}
    hooks = []

    try:
        for i in range(num_layers):
            layer = model.timesformer.encoder.layer[i]
            h = layer.register_forward_hook(
                _make_hidden_state_hook(hidden_store, i)
            )
            hooks.append(h)
            h2 = layer.temporal_attention.attention.register_forward_hook(
                _make_attn_capture_hook(temporal_attn_store, i)
            )
            hooks.append(h2)
            h3 = layer.attention.attention.register_forward_hook(
                _make_attn_capture_hook(spatial_attn_store, i)
            )
            hooks.append(h3)

        model.eval()
        _ = model(pixel_values=pixel_values)
    finally:
        for h in hooks:
            h.remove()

    return hidden_store, temporal_attn_store, spatial_attn_store


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute linear CKA between two representation matrices."""
    X = X - X.mean(dim=0)
    Y = Y - Y.mean(dim=0)

    hsic_xy = (X.T @ Y).pow(2).sum()
    hsic_xx = (X.T @ X).pow(2).sum()
    hsic_yy = (Y.T @ Y).pow(2).sum()

    denom = hsic_xx.sqrt() * hsic_yy.sqrt()
    if denom < 1e-10:
        return 1.0
    return (hsic_xy / denom).item()


def attention_jsd(
    bl_attn: torch.Tensor,
    ab_attn: torch.Tensor,
) -> float:
    """Compute Jensen-Shannon divergence between attention distributions."""
    bl = bl_attn.reshape(-1, bl_attn.shape[-1]).float().clamp(min=1e-8)
    ab = ab_attn.reshape(-1, ab_attn.shape[-1]).float().clamp(min=1e-8)
    bl = bl / bl.sum(dim=-1, keepdim=True)
    ab = ab / ab.sum(dim=-1, keepdim=True)

    m = 0.5 * (bl + ab)
    kl_bl_m = F.kl_div(m.log(), bl, reduction="batchmean").item()
    kl_ab_m = F.kl_div(m.log(), ab, reduction="batchmean").item()
    jsd = 0.5 * kl_bl_m + 0.5 * kl_ab_m
    return max(0.0, jsd / np.log(2))


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_propagation_metrics(
    baseline_hidden: Dict[int, torch.Tensor],
    ablated_hidden: Dict[int, torch.Tensor],
    baseline_temporal_attn: Dict[int, torch.Tensor],
    ablated_temporal_attn: Dict[int, torch.Tensor],
    baseline_spatial_attn: Dict[int, torch.Tensor],
    ablated_spatial_attn: Dict[int, torch.Tensor],
    source_layer: int,
) -> List[Dict]:
    """Compute divergence metrics for each layer.

    Includes per-position decomposition (CLS vs patch tokens) and
    cross-type attention analysis (temporal + spatial JSD).

    Args:
        baseline_hidden: Per-layer baseline hidden states.
        ablated_hidden: Per-layer ablated hidden states.
        baseline_temporal_attn: Per-layer baseline temporal attention probs.
        ablated_temporal_attn: Per-layer ablated temporal attention probs.
        baseline_spatial_attn: Per-layer baseline spatial attention probs.
        ablated_spatial_attn: Per-layer ablated spatial attention probs.
        source_layer: The layer where ablation was applied.

    Returns:
        List of metric dicts, one per layer.
    """
    results = []
    all_layers = sorted(baseline_hidden.keys())

    for target_layer in all_layers:
        if target_layer not in ablated_hidden:
            continue

        bl_h = baseline_hidden[target_layer].float()
        ab_h = ablated_hidden[target_layer].float()

        # Flatten to (total_tokens, hidden_dim) for global metrics
        bl_flat = bl_h.reshape(-1, bl_h.shape[-1])
        ab_flat = ab_h.reshape(-1, ab_h.shape[-1])

        # 1. Global cosine similarity
        cos_sim = F.cosine_similarity(bl_flat, ab_flat, dim=-1).mean().item()

        # 2. Global normalized L2 distance
        l2_dist = torch.norm(bl_flat - ab_flat, p=2, dim=-1).mean().item()
        bl_norm = torch.norm(bl_flat, p=2, dim=-1).mean().item()
        norm_l2 = l2_dist / (bl_norm + 1e-8)

        # 3. Feature norm ratio
        ab_norm = torch.norm(ab_flat, p=2, dim=-1).mean().item()
        norm_ratio = ab_norm / (bl_norm + 1e-8)

        # 4. Linear CKA
        cka = linear_cka(bl_flat, ab_flat)

        # 5. Per-position decomposition: CLS token (index 0) vs patch tokens
        # Hidden state shape is typically (batch_dim, seq_len, hidden_dim)
        # where seq_len includes CLS at index 0
        cls_cos, cls_l2, patch_cos, patch_l2 = 0.0, 0.0, 0.0, 0.0
        if bl_h.dim() >= 2 and bl_h.shape[-2] > 1:
            # CLS token: take index 0 along the sequence dimension
            bl_cls = bl_h[..., 0:1, :].reshape(-1, bl_h.shape[-1])
            ab_cls = ab_h[..., 0:1, :].reshape(-1, ab_h.shape[-1])
            cls_cos = F.cosine_similarity(bl_cls, ab_cls, dim=-1).mean().item()
            cls_l2_raw = torch.norm(bl_cls - ab_cls, p=2, dim=-1).mean().item()
            cls_norm = torch.norm(bl_cls, p=2, dim=-1).mean().item()
            cls_l2 = cls_l2_raw / (cls_norm + 1e-8)

            # Patch tokens: indices 1 onwards
            bl_patch = bl_h[..., 1:, :].reshape(-1, bl_h.shape[-1])
            ab_patch = ab_h[..., 1:, :].reshape(-1, ab_h.shape[-1])
            patch_cos = F.cosine_similarity(bl_patch, ab_patch, dim=-1).mean().item()
            patch_l2_raw = torch.norm(bl_patch - ab_patch, p=2, dim=-1).mean().item()
            patch_norm = torch.norm(bl_patch, p=2, dim=-1).mean().item()
            patch_l2 = patch_l2_raw / (patch_norm + 1e-8)

        # 6. Temporal attention KL and JSD
        attn_kl = 0.0
        attn_jsd_val = 0.0
        if (target_layer in baseline_temporal_attn and
                target_layer in ablated_temporal_attn):
            bl_a = baseline_temporal_attn[target_layer]
            ab_a = ablated_temporal_attn[target_layer]
            bl_a_flat = bl_a.reshape(-1, bl_a.shape[-1]).float().clamp(min=1e-8)
            ab_a_flat = ab_a.reshape(-1, ab_a.shape[-1]).float().clamp(min=1e-8)
            bl_a_flat = bl_a_flat / bl_a_flat.sum(dim=-1, keepdim=True)
            ab_a_flat = ab_a_flat / ab_a_flat.sum(dim=-1, keepdim=True)
            kl = F.kl_div(
                ab_a_flat.log(), bl_a_flat, reduction="batchmean"
            ).item()
            attn_kl = max(0.0, kl)
            attn_jsd_val = attention_jsd(bl_a, ab_a)

        # 7. Spatial attention JSD (cross-type coupling)
        spatial_attn_jsd_val = 0.0
        if (target_layer in baseline_spatial_attn and
                target_layer in ablated_spatial_attn):
            bl_sa = baseline_spatial_attn[target_layer]
            ab_sa = ablated_spatial_attn[target_layer]
            spatial_attn_jsd_val = attention_jsd(bl_sa, ab_sa)

        results.append({
            "target_layer": target_layer,
            "cosine_sim": cos_sim,
            "l2_distance": norm_l2,
            "attention_kl": attn_kl,
            "attention_jsd": attn_jsd_val,
            "spatial_attention_jsd": spatial_attn_jsd_val,
            "norm_ratio": norm_ratio,
            "cka": cka,
            "cls_cosine_sim": cls_cos,
            "cls_l2_distance": cls_l2,
            "patch_cosine_sim": patch_cos,
            "patch_l2_distance": patch_l2,
        })

    return results


# ---------------------------------------------------------------------------
# Main propagation analysis
# ---------------------------------------------------------------------------

@torch.no_grad()
def analyze_propagation(
    model,
    dataloader,
    config: AblationConfig,
    source_layers: List[int],
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Analyze how ablating each temporal head propagates downstream.

    For each source_layer, runs the baseline forward pass once and caches it,
    then runs ablated passes for each of the 12 heads. This halves the number
    of forward passes compared to recomputing baseline per head.

    Args:
        model: TimeSformer model.
        dataloader: Evaluation DataLoader.
        config: AblationConfig.
        source_layers: Which layers to ablate from.

    Returns:
        (propagation_df, propagation_matrices):
            propagation_df: DataFrame with columns source_layer, source_head,
                target_layer, cosine_sim, l2_distance, attention_kl,
                attention_jsd, spatial_attention_jsd, norm_ratio, cka,
                cls_cosine_sim, cls_l2_distance, patch_cosine_sim,
                patch_l2_distance
            propagation_matrices: Dict mapping metric_name -> array of shape
                (len(source_layers), num_heads, num_layers)
    """
    num_layers = config.num_layers
    num_heads = config.num_heads

    # Monkey-patch temporal attention for attention capture
    patched = _force_temporal_output_attentions(model, num_layers)

    all_rows = []
    total_combos = len(source_layers) * num_heads
    combo_idx = 0

    try:
        for source_layer in source_layers:
            print(f"\n  Source layer {source_layer}: capturing baseline...")

            # Cache baseline across all batches for this source layer
            baseline_cache: List[Tuple[torch.Tensor, Dict[int, torch.Tensor],
                                       Dict[int, torch.Tensor],
                                       Dict[int, torch.Tensor]]] = []
            for pixel_values, _labels in dataloader:
                pixel_values = pixel_values.to(config.device)
                bl_hidden, bl_t_attn, bl_s_attn = _capture_forward_pass(
                    model, pixel_values, num_layers,
                )
                baseline_cache.append((pixel_values, bl_hidden, bl_t_attn,
                                       bl_s_attn))

            # Now ablate each head, reusing cached baseline
            for head_idx in range(num_heads):
                combo_idx += 1
                batch_metrics: Dict[int, List[Dict]] = {}

                for pixel_values, bl_hidden, bl_t_attn, bl_s_attn in baseline_cache:
                    # Ablated forward pass
                    target_module = (
                        model.timesformer.encoder.layer[source_layer]
                        .temporal_attention.attention
                    )
                    abl_hook = target_module.register_forward_hook(
                        make_ablation_hook(head_idx, num_heads, config.head_dim)
                    )
                    try:
                        ab_hidden, ab_t_attn, ab_s_attn = _capture_forward_pass(
                            model, pixel_values, num_layers,
                        )
                    finally:
                        abl_hook.remove()

                    # Compute metrics
                    batch_results = compute_propagation_metrics(
                        bl_hidden, ab_hidden,
                        bl_t_attn, ab_t_attn,
                        bl_s_attn, ab_s_attn,
                        source_layer,
                    )
                    for entry in batch_results:
                        tl = entry["target_layer"]
                        if tl not in batch_metrics:
                            batch_metrics[tl] = []
                        batch_metrics[tl].append(entry)

                # Average across batches
                for target_layer in sorted(batch_metrics.keys()):
                    entries = batch_metrics[target_layer]
                    avg_row = {
                        "source_layer": source_layer,
                        "source_head": head_idx,
                        "target_layer": target_layer,
                    }
                    for key in ["cosine_sim", "l2_distance", "attention_kl",
                                "attention_jsd", "spatial_attention_jsd",
                                "norm_ratio", "cka",
                                "cls_cosine_sim", "cls_l2_distance",
                                "patch_cosine_sim", "patch_l2_distance"]:
                        avg_row[key] = np.mean([e[key] for e in entries])
                    all_rows.append(avg_row)

                if combo_idx % 6 == 0 or combo_idx <= 2:
                    print(f"  [{combo_idx}/{total_combos}] "
                          f"L{source_layer}-H{head_idx} done")

    finally:
        _restore_temporal_forwards(patched)

    df = pd.DataFrame(all_rows)

    # Build propagation matrices: (len(source_layers), num_heads, num_layers)
    all_metrics = ["cosine_sim", "l2_distance", "attention_kl",
                   "attention_jsd", "spatial_attention_jsd", "norm_ratio",
                   "cka", "cls_cosine_sim", "cls_l2_distance",
                   "patch_cosine_sim", "patch_l2_distance"]
    matrices = {}
    for metric in all_metrics:
        mat = np.zeros((len(source_layers), num_heads, num_layers))
        for i, sl in enumerate(source_layers):
            for h in range(num_heads):
                subset = df[(df["source_layer"] == sl) & (df["source_head"] == h)]
                for _, row in subset.iterrows():
                    tl = int(row["target_layer"])
                    if tl < num_layers:
                        mat[i, h, tl] = row[metric]
        matrices[metric] = mat

    return df, matrices


# ---------------------------------------------------------------------------
# Amplification / dampening classification
# ---------------------------------------------------------------------------

def classify_propagation(df: pd.DataFrame) -> pd.DataFrame:
    """Classify each (source_layer, source_head) as amplifying or dampening.

    Compares relative L2 at the first downstream layer vs the last layer (11).
    Ratio > 1 means the effect grows (amplifies); < 1 means it shrinks (dampens).

    Returns:
        DataFrame with columns: source_layer, source_head, l2_first_downstream,
        l2_last_layer, amplification_ratio, classification.
    """
    rows = []
    for (sl, sh), group in df.groupby(["source_layer", "source_head"]):
        downstream = group[group["target_layer"] > sl].sort_values("target_layer")
        if len(downstream) < 2:
            continue
        first = downstream.iloc[0]
        last = downstream.iloc[-1]

        l2_first = first["l2_distance"]
        l2_last = last["l2_distance"]
        ratio = l2_last / (l2_first + 1e-8)

        if ratio > 1.2:
            classification = "amplifying"
        elif ratio < 0.8:
            classification = "dampening"
        else:
            classification = "stable"

        rows.append({
            "source_layer": int(sl),
            "source_head": int(sh),
            "l2_first_downstream": l2_first,
            "l2_last_layer": l2_last,
            "amplification_ratio": ratio,
            "classification": classification,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_downstream_propagation(
    num_videos: int = 20,
    source_layers: Optional[List[int]] = None,
    batch_size: int = 2,
    output_dir: str = "outputs/temporal_analysis",
    ssv2_root_dir: Optional[str] = None,
    data_split: str = "val",
    model_name: Optional[str] = None,
) -> pd.DataFrame:
    """Run the full downstream propagation analysis.

    Args:
        num_videos: Number of videos to evaluate on.
        source_layers: Layers to ablate from (default: [0, 3, 6, 9, 11]).
        batch_size: DataLoader batch size.
        output_dir: Directory for output files.
        ssv2_root_dir: Path to evaluation_frames/ directory.
        data_split: Dataset split to use.
        model_name: Model name or local path.

    Returns:
        DataFrame with propagation effects.
    """
    if source_layers is None:
        source_layers = [0, 3, 6, 9, 11]

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
    print("  Downstream Propagation Analysis — TimeSformer")
    print("=" * 60)
    print(f"  Videos: {num_videos}")
    print(f"  Source layers: {source_layers}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {config.device}")
    print(f"  Data: SSv2 ({ssv2_root_dir})")
    print(f"  Output: {output_path}")
    print()

    start_time = time.time()

    # Load model + data (automatically subset to num_videos via config)
    model, processor = load_model_and_processor(config)
    dataloader = create_dataloader_from_config(config, processor=processor)

    # Run analysis
    print(f"\nAnalyzing propagation for {len(source_layers)} source layers "
          f"x {config.num_heads} heads = "
          f"{len(source_layers) * config.num_heads} combos...")
    df, matrices = analyze_propagation(model, dataloader, config, source_layers)

    # Save CSV
    csv_path = output_path / "propagation_effects.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nPropagation effects saved to {csv_path}")

    # Save per-metric matrices as NPY
    for metric_name, mat in matrices.items():
        npy_path = output_path / f"propagation_{metric_name}.npy"
        np.save(npy_path, mat)
    print(f"Propagation matrices saved ({len(matrices)} metrics)")

    # Amplification / dampening classification
    prop_class = classify_propagation(df)
    if len(prop_class) > 0:
        class_path = output_path / "propagation_classification.csv"
        prop_class.to_csv(class_path, index=False)
        print(f"\nPropagation classification:")
        for cls, count in prop_class["classification"].value_counts().items():
            print(f"  {cls}: {count} heads")

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"DOWNSTREAM PROPAGATION COMPLETE — {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 60}")

    # Per-source-layer summary
    for sl in source_layers:
        sl_df = df[df["source_layer"] == sl]
        downstream = sl_df[sl_df["target_layer"] > sl]
        if len(downstream) > 0:
            print(f"  L{sl}: downstream mean cos={downstream['cosine_sim'].mean():.4f}, "
                  f"l2={downstream['l2_distance'].mean():.4f}, "
                  f"cka={downstream['cka'].mean():.4f}, "
                  f"t_jsd={downstream['attention_jsd'].mean():.4f}, "
                  f"s_jsd={downstream['spatial_attention_jsd'].mean():.4f}")

    # Identify strongest propagation effects
    downstream_only = df[df["target_layer"] > df["source_layer"]]
    if len(downstream_only) > 0:
        worst = downstream_only.nsmallest(5, "cosine_sim")
        print(f"\n  Strongest effects (lowest cosine similarity):")
        for _, row in worst.iterrows():
            print(f"    L{int(row['source_layer'])}-H{int(row['source_head'])} "
                  f"-> L{int(row['target_layer'])}: "
                  f"cos={row['cosine_sim']:.4f}, cka={row['cka']:.4f}")

    # Save summary JSON
    summary = {
        "num_videos": num_videos,
        "source_layers": source_layers,
        "total_combos": len(source_layers) * config.num_heads,
        "elapsed_seconds": round(elapsed, 1),
        "metrics": ["cosine_sim", "l2_distance", "attention_kl",
                     "attention_jsd", "spatial_attention_jsd", "norm_ratio",
                     "cka", "cls_cosine_sim", "cls_l2_distance",
                     "patch_cosine_sim", "patch_l2_distance"],
    }
    if len(prop_class) > 0:
        summary["classification"] = prop_class["classification"].value_counts().to_dict()
    summary_path = output_path / "propagation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze downstream propagation of temporal head ablation"
    )
    parser.add_argument("--ssv2_root_dir", type=str, required=True,
                        help="Path to evaluation_frames/ directory")
    parser.add_argument("--num_videos", type=int, default=20,
                        help="Number of videos to evaluate on")
    parser.add_argument("--source_layers", type=int, nargs="*",
                        default=None,
                        help="Layers to ablate from (default: 0 3 6 9 11)")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output_dir", type=str,
                        default="outputs/temporal_analysis")
    parser.add_argument("--data_split", type=str, default="val")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model name or local path (default: config.py default)")
    args = parser.parse_args()

    run_downstream_propagation(
        num_videos=args.num_videos,
        source_layers=args.source_layers,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        ssv2_root_dir=args.ssv2_root_dir,
        data_split=args.data_split,
        model_name=args.model_name,
    )
