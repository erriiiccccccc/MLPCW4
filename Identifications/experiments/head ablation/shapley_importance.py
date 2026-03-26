"""Monte Carlo Shapley values for temporal attention heads."""

import json
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Set, Tuple

from config import AblationConfig
from baseline import load_model_and_processor
from real_video_loader import create_dataloader_from_config


def make_multi_ablation_hook(
    heads_to_ablate: Set[int],
    num_heads: int = 12,
    head_dim: int = 64,
):
    """Create a hook that zeros an arbitrary subset of heads."""
    ablate_list = sorted(heads_to_ablate)

    def hook_fn(module, input, output):
        context = output[0]  # (B, N, C)
        B, N, C = context.shape
        context = context.view(B, N, num_heads, head_dim)
        for h in ablate_list:
            context[:, :, h, :] = 0.0
        context = context.view(B, N, C)
        if len(output) > 1:
            return (context,) + output[1:]
        return (context,)

    return hook_fn


@torch.no_grad()
def evaluate_coalition(
    model,
    dataloader,
    layer_idx: int,
    heads_to_ablate: Set[int],
    baseline_logits: torch.Tensor,
    baseline_preds: torch.Tensor,
    config: AblationConfig,
    verbose: bool = False,
) -> float:
    """Evaluate the model with a specific set of temporal heads ablated."""
    if not heads_to_ablate:
        if verbose:
            print(f"    eval L{layer_idx} ablate={{}} → shortcut (no ablation)", flush=True)
        return 0.5 * 1.0 + 0.5 * 0.0

    if verbose:
        t0 = time.time()

    encoder_layer = model.timesformer.encoder.layer[layer_idx]
    target_module = encoder_layer.temporal_attention.attention

    hook = target_module.register_forward_hook(
        make_multi_ablation_hook(heads_to_ablate, config.num_heads, config.head_dim)
    )

    all_logits = []
    all_preds = []
    model.eval()

    try:
        for batch_idx, (pixel_values, _labels) in enumerate(dataloader):
            pixel_values = pixel_values.to(config.device)
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits.cpu()
            all_logits.append(logits)
            all_preds.append(logits.argmax(dim=-1))
    finally:
        hook.remove()

    ablated_logits = torch.cat(all_logits, dim=0)
    ablated_preds = torch.cat(all_preds, dim=0)

    flip_rate = (ablated_preds != baseline_preds).float().mean().item()
    flip_val = 1.0 - flip_rate

    baseline_probs = F.softmax(baseline_logits, dim=-1)
    ablated_probs = F.softmax(ablated_logits, dim=-1)
    ablated_log_probs = torch.clamp(ablated_probs, min=1e-8).log()
    kl_div = F.kl_div(
        ablated_log_probs, baseline_probs, reduction="batchmean"
    ).item()
    kl_val = -kl_div

    if verbose:
        elapsed = time.time() - t0
        print(f"    eval L{layer_idx} ablate={sorted(heads_to_ablate)} "
              f"→ flip={flip_rate:.3f} kl={kl_div:.4f} v={0.5*flip_val+0.5*kl_val:.4f} "
              f"({elapsed:.2f}s)", flush=True)

    return 0.5 * flip_val + 0.5 * kl_val


@torch.no_grad()
def get_baseline_predictions(model, dataloader, device: str):
    """Collect baseline logits and predicted class ids."""
    all_logits, all_preds = [], []
    model.eval()
    for pixel_values, _labels in tqdm(dataloader, desc="Baseline predictions"):
        pixel_values = pixel_values.to(device)
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits.cpu()
        all_logits.append(logits)
        all_preds.append(logits.argmax(dim=-1))
    return torch.cat(all_logits, dim=0), torch.cat(all_preds, dim=0)


def compute_shapley_from_value_fn(
    value_fn,
    num_heads: int,
    num_permutations: int = 200,
) -> Dict[int, float]:
    """Compute Shapley values from a generic value function."""
    shapley = np.zeros(num_heads)
    all_heads = set(range(num_heads))

    num_pairs = max(num_permutations // 2, 1)

    for _ in range(num_pairs):
        perm_fwd = np.random.permutation(num_heads)
        perm_rev = perm_fwd[::-1].copy()

        for perm in [perm_fwd, perm_rev]:
            coalition = set()
            v_prev = value_fn(coalition)

            for head in perm:
                coalition.add(head)
                v_curr = value_fn(coalition)
                shapley[head] += v_curr - v_prev
                v_prev = v_curr

    total_perms = num_pairs * 2
    shapley /= total_perms

    return {h: shapley[h] for h in range(num_heads)}


def compute_shapley_layer(
    model,
    dataloader,
    layer_idx: int,
    baseline_logits: torch.Tensor,
    baseline_preds: torch.Tensor,
    config: AblationConfig,
    num_permutations: int = 200,
    convergence_threshold: float = 0.01,
    convergence_check_every: int = 10,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Shapley values for the temporal heads in one layer.
    Args:
        model: TimeSformer model.
        dataloader: Evaluation DataLoader.
        layer_idx: Transformer layer index (0-11).
        baseline_logits: Baseline logits tensor.
        baseline_preds: Baseline prediction ids.
        config: AblationConfig.
        num_permutations: Total permutations (includes both forward + reverse).
        convergence_threshold: Stop early when max relative change < threshold.
        convergence_check_every: Check convergence every N antithetic pairs.
        verbose: If True, print per-evaluation progress logs.

    Returns:
        (shapley_values, shapley_stderrs): arrays of shape (num_heads,).
    """
    num_heads = config.num_heads
    all_heads = set(range(num_heads))

    # cache results so we dont recompute the same coalition twice
    cache: Dict[frozenset, float] = {}
    eval_count = [0]

    def cached_evaluate(heads_to_ablate: Set[int]) -> float:
        key = frozenset(heads_to_ablate)
        if key not in cache:
            eval_count[0] += 1
            if verbose:
                print(f"  [eval #{eval_count[0]}] L{layer_idx} "
                      f"ablating {len(heads_to_ablate)}/{num_heads} heads "
                      f"(cache miss, {len(cache)} cached)", flush=True)
            cache[key] = evaluate_coalition(
                model, dataloader, layer_idx, heads_to_ablate,
                baseline_logits, baseline_preds, config,
                verbose=verbose,
            )
        elif verbose:
            print(f"  [cache hit] L{layer_idx} "
                  f"ablating {len(heads_to_ablate)} heads", flush=True)
        return cache[key]

    # track marginal contributions per permutation, keep all of em for std later
    marginal_contributions: List[np.ndarray] = []

    # antithetic sampling - run fwd perm and its reverse, halves variance
    num_pairs = num_permutations // 2
    effective_permutations = num_pairs * 2

    print(f"  Layer {layer_idx}: {effective_permutations} permutations "
          f"({num_pairs} antithetic pairs)", flush=True)

    prev_shapley = None
    converged = False
    layer_start = time.time()

    for pair_idx in tqdm(range(num_pairs), desc=f"  L{layer_idx} Shapley",
                         leave=False):
        pair_start = time.time()
        perm_fwd = np.random.permutation(num_heads)
        perm_rev = perm_fwd[::-1].copy()

        for perm_i, perm in enumerate([perm_fwd, perm_rev]):
            mc = np.zeros(num_heads)
            coalition = set()
            # v(empty coalition active) = all heads ablated
            v_prev = cached_evaluate(all_heads - coalition)  # ablate everything

            for head in perm:
                coalition.add(head)
                heads_to_ablate = all_heads - coalition
                v_curr = cached_evaluate(heads_to_ablate)
                mc[head] = v_curr - v_prev
                v_prev = v_curr

            marginal_contributions.append(mc)

        pair_elapsed = time.time() - pair_start
        total_elapsed = time.time() - layer_start
        print(f"  L{layer_idx} pair {pair_idx+1}/{num_pairs} done "
              f"({pair_elapsed:.1f}s this pair, {total_elapsed:.1f}s total, "
              f"{len(cache)} unique coalitions cached)", flush=True)

        # check if we've converged yet (max relative change across all heads)
        if (convergence_check_every > 0
                and (pair_idx + 1) % convergence_check_every == 0
                and len(marginal_contributions) >= 4):
            mc_array_so_far = np.stack(marginal_contributions, axis=0)
            curr_shapley = mc_array_so_far.mean(axis=0)

            if prev_shapley is not None:
                abs_change = np.abs(curr_shapley - prev_shapley)
                denom = np.maximum(np.abs(curr_shapley), 1e-6)
                max_rel_change = (abs_change / denom).max()
                if max_rel_change < convergence_threshold:
                    print(f"  L{layer_idx}: converged at {len(marginal_contributions)} "
                          f"permutations (max rel change={max_rel_change:.4f})", flush=True)
                    converged = True
                    break

            prev_shapley = curr_shapley.copy()

    # aggregate across all permutations
    mc_array = np.stack(marginal_contributions, axis=0)  # (perms, heads)
    shapley_values = mc_array.mean(axis=0)
    shapley_stderrs = mc_array.std(axis=0) / np.sqrt(len(mc_array))

    cache_size = len(cache)
    total_evals = len(marginal_contributions) * (num_heads + 1)
    hit_rate = 1.0 - cache_size / max(total_evals, 1)
    status = "CONVERGED" if converged else f"{len(marginal_contributions)} perms"
    print(f"  L{layer_idx}: {status}, cache size={cache_size}, "
          f"hit rate={hit_rate:.1%}, "
          f"max |phi|={np.abs(shapley_values).max():.4f}", flush=True)

    return shapley_values, shapley_stderrs


def compute_layer_shapley_values(
    model,
    dataloader,
    layer_idx: int,
    num_permutations: int,
    baseline_preds: torch.Tensor,
    config: AblationConfig,
    baseline_logits: Optional[torch.Tensor] = None,
    verbose: bool = False,
) -> Dict[int, float]:
    """Convenience wrapper matching the test-expected API.

    Computes Shapley values for a single layer and returns a dict
    mapping head_idx -> shapley_value (without stderr).

    If baseline_logits is not provided, computes a dummy version from
    baseline_preds (one-hot) to keep the value function operational.
    """
    if baseline_logits is None:
        # Approximate: create one-hot logits from predictions
        num_classes = config.num_classes
        baseline_logits = F.one_hot(
            baseline_preds, num_classes
        ).float() * 10.0  # sharp logits

    sv, _ = compute_shapley_layer(
        model, dataloader, layer_idx,
        baseline_logits, baseline_preds,
        config, num_permutations,
        verbose=verbose,
    )
    return {h: float(sv[h]) for h in range(config.num_heads)}


def compute_all_shapley(
    model,
    dataloader,
    baseline_logits: torch.Tensor,
    baseline_preds: torch.Tensor,
    config: AblationConfig,
    num_permutations: int = 200,
    layers: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Compute Shapley values for temporal heads across all (or selected) layers.

    Args:
        model: TimeSformer model.
        dataloader: Evaluation DataLoader.
        baseline_logits: (N, C) tensor of baseline logits.
        baseline_preds: (N,) tensor of baseline predicted class ids.
        config: AblationConfig.
        num_permutations: Number of permutations per layer.
        layers: Specific layers to compute (default: all 12).

    Returns:
        DataFrame with columns: layer, head, shapley_value, stderr, ci_lower, ci_upper.
    """
    layers = layers or list(range(config.num_layers))
    results = []

    print(f"\nComputing Shapley values for {len(layers)} layers, "
          f"{num_permutations} permutations each")
    print(f"Value function: 0.5*(1-flip_rate) + 0.5*(-KL_divergence)")
    print(f"Variance reduction: antithetic sampling\n")

    for layer_idx in layers:
        sv, se = compute_shapley_layer(
            model, dataloader, layer_idx,
            baseline_logits, baseline_preds,
            config, num_permutations,
        )
        for head_idx in range(config.num_heads):
            results.append({
                "layer": layer_idx,
                "head": head_idx,
                "shapley_value": sv[head_idx],
                "stderr": se[head_idx],
                "ci_lower": sv[head_idx] - 1.96 * se[head_idx],
                "ci_upper": sv[head_idx] + 1.96 * se[head_idx],
            })

    df = pd.DataFrame(results)
    return df


# ---------------------------------------------------------------------------
# Efficiency check: Shapley values should sum to v(N) - v(empty)
# ---------------------------------------------------------------------------

def verify_efficiency(
    df: pd.DataFrame,
    model,
    dataloader,
    baseline_logits: torch.Tensor,
    baseline_preds: torch.Tensor,
    config: AblationConfig,
    layers: Optional[List[int]] = None,
) -> Dict[int, Dict[str, float]]:
    """Verify the Shapley efficiency property per layer.

    sum(phi_i) should approximately equal v(all heads) - v(no heads).

    Returns:
        Dict mapping layer_idx -> {sum_phi, v_all, v_empty, v_diff, error}.
    """
    layers = layers or sorted(df["layer"].unique())
    all_heads = set(range(config.num_heads))
    checks = {}

    for layer_idx in layers:
        layer_df = df[df["layer"] == layer_idx]
        sum_phi = layer_df["shapley_value"].sum()

        v_all = evaluate_coalition(
            model, dataloader, layer_idx, set(),
            baseline_logits, baseline_preds, config,
        )
        v_empty = evaluate_coalition(
            model, dataloader, layer_idx, all_heads,
            baseline_logits, baseline_preds, config,
        )
        v_diff = v_all - v_empty

        checks[layer_idx] = {
            "sum_phi": sum_phi,
            "v_all": v_all,
            "v_empty": v_empty,
            "v_diff": v_diff,
            "error": abs(sum_phi - v_diff),
        }
        print(f"  L{layer_idx}: sum(phi)={sum_phi:.4f}, "
              f"v(N)-v(0)={v_diff:.4f}, "
              f"error={abs(sum_phi - v_diff):.4f}")

    return checks


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_shapley(
    num_videos: int = 20,
    num_permutations: int = 200,
    layers: Optional[List[int]] = None,
    batch_size: int = 2,
    output_dir: str = "outputs/temporal_analysis",
    ssv2_root_dir: Optional[str] = None,
    data_split: str = "val",
    model_name: Optional[str] = None,
) -> pd.DataFrame:
    """Run the full Shapley importance pipeline.

    Args:
        num_videos: Number of videos to evaluate on.
        num_permutations: Permutations per layer for Monte Carlo Shapley.
        layers: Specific layers (default: all 12).
        batch_size: DataLoader batch size.
        output_dir: Directory for output files.
        ssv2_root_dir: Path to evaluation_frames/ directory.
        data_split: Dataset split to use.

    Returns:
        DataFrame with Shapley values.
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
    print("  Monte Carlo Shapley Value Computation")
    print("  Temporal Attention Heads — TimeSformer")
    print("=" * 60)
    print(f"  Videos: {num_videos}")
    print(f"  Permutations per layer: {num_permutations}")
    print(f"  Layers: {layers or 'all 12'}")
    print(f"  Device: {config.device}")
    print(f"  Data: SSv2 ({ssv2_root_dir})")
    print(f"  Output: {output_path}")
    print()

    start_time = time.time()

    # Load model
    model, processor = load_model_and_processor(config)

    # Load data (automatically subset to num_videos via config)
    print("\nLoading videos...")
    dataloader = create_dataloader_from_config(config, processor=processor)

    # Baseline
    print("\nCollecting baseline predictions...")
    baseline_logits, baseline_preds = get_baseline_predictions(
        model, dataloader, config.device
    )
    n_videos = len(baseline_preds)
    print(f"Baseline: {n_videos} videos, "
          f"{len(torch.unique(baseline_preds))} unique predictions")

    # Shapley computation
    df = compute_all_shapley(
        model, dataloader,
        baseline_logits, baseline_preds,
        config, num_permutations, layers,
    )

    # Save CSV
    csv_path = output_path / "shapley_values.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nShapley values saved to {csv_path}")

    # Efficiency verification
    print("\nVerifying Shapley efficiency property...")
    eff_checks = verify_efficiency(
        df, model, dataloader,
        baseline_logits, baseline_preds,
        config, layers,
    )

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"SHAPLEY COMPUTATION COMPLETE — {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 60}")

    # Per-layer summary
    for layer_idx in sorted(df["layer"].unique()):
        layer_df = df[df["layer"] == layer_idx]
        top_head = layer_df.loc[layer_df["shapley_value"].abs().idxmax()]
        print(f"  L{layer_idx}: max |phi| = L{layer_idx}-H{int(top_head['head'])} "
              f"({top_head['shapley_value']:+.4f} ± {top_head['stderr']:.4f})")

    # Save summary JSON
    summary = {
        "num_videos": n_videos,
        "num_permutations": num_permutations,
        "layers_computed": sorted(df["layer"].unique().tolist()),
        "elapsed_seconds": elapsed,
        "value_function": "0.5*(1-flip_rate) + 0.5*(-KL_divergence)",
        "variance_reduction": "antithetic_sampling",
        "efficiency_checks": {
            str(k): {kk: round(vv, 6) for kk, vv in v.items()}
            for k, v in eff_checks.items()
        },
    }
    summary_path = output_path / "shapley_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute Monte Carlo Shapley values for temporal heads"
    )
    parser.add_argument("--num_videos", type=int, default=20,
                        help="Number of videos to evaluate on")
    parser.add_argument("--num_permutations", type=int, default=200,
                        help="Permutations per layer")
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="Specific layers (default: all 12)")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output_dir", type=str,
                        default="outputs/temporal_analysis")
    parser.add_argument("--ssv2_root_dir", type=str, required=True,
                        help="Path to evaluation_frames/ directory")
    parser.add_argument("--data_split", type=str, default="val")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model name or local path (default: config.py default)")
    args = parser.parse_args()

    run_shapley(
        num_videos=args.num_videos,
        num_permutations=args.num_permutations,
        layers=args.layers,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        ssv2_root_dir=args.ssv2_root_dir,
        data_split=args.data_split,
        model_name=args.model_name,
    )
