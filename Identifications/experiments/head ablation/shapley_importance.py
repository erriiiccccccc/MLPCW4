# monte carlo shapley values for temporal attention heads

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


def make_multi_ablation_hook(heads_to_ablate, num_heads=12, head_dim=64):
    ablate_list = sorted(heads_to_ablate)

    def hook_fn(module, input, output):
        context = output[0]
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
    model, dataloader, layer_idx, heads_to_ablate,
    baseline_logits, baseline_preds, config, verbose=False,
):
    if not heads_to_ablate:
        if verbose:
            print(f"    eval L{layer_idx} ablate={{}} -> shortcut (no ablation)")
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
            logits = model(pixel_values=pixel_values).logits.cpu()
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
              f"-> flip={flip_rate:.3f} kl={kl_div:.4f} v={0.5*flip_val+0.5*kl_val:.4f} "
              f"({elapsed:.2f}s)")

    return 0.5 * flip_val + 0.5 * kl_val


@torch.no_grad()
def get_baseline_predictions(model, dataloader, device):
    all_logits, all_preds = [], []
    model.eval()
    for pixel_values, _labels in tqdm(dataloader, desc="Baseline predictions"):
        pixel_values = pixel_values.to(device)
        logits = model(pixel_values=pixel_values).logits.cpu()
        all_logits.append(logits)
        all_preds.append(logits.argmax(dim=-1))
    return torch.cat(all_logits, dim=0), torch.cat(all_preds, dim=0)


def compute_shapley_from_value_fn(value_fn, num_heads, num_permutations=200):
    shapley = np.zeros(num_heads)

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

    shapley /= (num_pairs * 2)
    return {h: shapley[h] for h in range(num_heads)}


def compute_shapley_layer(
    model, dataloader, layer_idx,
    baseline_logits, baseline_preds,
    config, num_permutations=200,
    convergence_threshold=0.01,
    convergence_check_every=10,
    verbose=False,
):
    num_heads = config.num_heads
    all_heads = set(range(num_heads))

    # cache coalition results to avoid recomputation
    cache = {}
    eval_count = [0]

    def cached_evaluate(heads_to_ablate):
        key = frozenset(heads_to_ablate)
        if key not in cache:
            eval_count[0] += 1
            if verbose:
                print(f"  [eval #{eval_count[0]}] L{layer_idx} "
                      f"ablating {len(heads_to_ablate)}/{num_heads} heads "
                      f"(cache miss, {len(cache)} cached)")
            cache[key] = evaluate_coalition(
                model, dataloader, layer_idx, heads_to_ablate,
                baseline_logits, baseline_preds, config, verbose=verbose,
            )
        elif verbose:
            print(f"  [cache hit] L{layer_idx} ablating {len(heads_to_ablate)} heads")
        return cache[key]

    marginal_contributions = []

    # antithetic sampling: fwd perm + reverse, halves variance
    num_pairs = num_permutations // 2
    effective_permutations = num_pairs * 2

    print(f"  Layer {layer_idx}: {effective_permutations} permutations "
          f"({num_pairs} antithetic pairs)")

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
            v_prev = cached_evaluate(all_heads - coalition)

            for head in perm:
                coalition.add(head)
                v_curr = cached_evaluate(all_heads - coalition)
                mc[head] = v_curr - v_prev
                v_prev = v_curr

            marginal_contributions.append(mc)

        pair_elapsed = time.time() - pair_start
        total_elapsed = time.time() - layer_start
        print(f"  L{layer_idx} pair {pair_idx+1}/{num_pairs} done "
              f"({pair_elapsed:.1f}s this pair, {total_elapsed:.1f}s total, "
              f"{len(cache)} unique coalitions cached)")

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
                          f"permutations (max rel change={max_rel_change:.4f})")
                    converged = True
                    break

            prev_shapley = curr_shapley.copy()

    mc_array = np.stack(marginal_contributions, axis=0)
    shapley_values = mc_array.mean(axis=0)
    shapley_stderrs = mc_array.std(axis=0) / np.sqrt(len(mc_array))

    cache_size = len(cache)
    total_evals = len(marginal_contributions) * (num_heads + 1)
    hit_rate = 1.0 - cache_size / max(total_evals, 1)
    status = "CONVERGED" if converged else f"{len(marginal_contributions)} perms"
    print(f"  L{layer_idx}: {status}, cache size={cache_size}, "
          f"hit rate={hit_rate:.1%}, "
          f"max |phi|={np.abs(shapley_values).max():.4f}")

    return shapley_values, shapley_stderrs


def compute_layer_shapley_values(
    model, dataloader, layer_idx, num_permutations,
    baseline_preds, config, baseline_logits=None, verbose=False,
):
    """Convenience wrapper matching the test-expected API."""
    if baseline_logits is None:
        # approximate: one-hot logits from predictions
        baseline_logits = F.one_hot(
            baseline_preds, config.num_classes
        ).float() * 10.0

    sv, _ = compute_shapley_layer(
        model, dataloader, layer_idx,
        baseline_logits, baseline_preds,
        config, num_permutations, verbose=verbose,
    )
    return {h: float(sv[h]) for h in range(config.num_heads)}


def compute_all_shapley(
    model, dataloader,
    baseline_logits, baseline_preds,
    config, num_permutations=200, layers=None,
):
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

    return pd.DataFrame(results)


def verify_efficiency(
    df, model, dataloader,
    baseline_logits, baseline_preds,
    config, layers=None,
):
    """sum(phi_i) should approx equal v(all heads) - v(no heads)"""
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


def run_shapley(
    num_videos=20, num_permutations=200,
    layers=None, batch_size=2,
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

    print(f"Monte Carlo Shapley Value Computation")
    print(f"  Videos: {num_videos}, Perms/layer: {num_permutations}")
    print(f"  Layers: {layers or 'all 12'}, Device: {config.device}")
    print(f"  Data: SSv2 ({ssv2_root_dir}), Output: {output_path}")

    start_time = time.time()

    model, processor = load_model_and_processor(config)

    print("\nLoading videos...")
    dataloader = create_dataloader_from_config(config, processor=processor)

    print("\nCollecting baseline predictions...")
    baseline_logits, baseline_preds = get_baseline_predictions(
        model, dataloader, config.device
    )
    n_videos = len(baseline_preds)
    print(f"Baseline: {n_videos} videos, "
          f"{len(torch.unique(baseline_preds))} unique predictions")

    df = compute_all_shapley(
        model, dataloader,
        baseline_logits, baseline_preds,
        config, num_permutations, layers,
    )

    csv_path = output_path / "shapley_values.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nShapley values saved to {csv_path}")

    print("\nVerifying Shapley efficiency property...")
    eff_checks = verify_efficiency(
        df, model, dataloader,
        baseline_logits, baseline_preds,
        config, layers,
    )

    elapsed = time.time() - start_time
    print(f"\nSHAPLEY COMPUTATION done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    for layer_idx in sorted(df["layer"].unique()):
        layer_df = df[df["layer"] == layer_idx]
        top_head = layer_df.loc[layer_df["shapley_value"].abs().idxmax()]
        print(f"  L{layer_idx}: max |phi| = L{layer_idx}-H{int(top_head['head'])} "
              f"({top_head['shapley_value']:+.4f} +/- {top_head['stderr']:.4f})")

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_videos", type=int, default=20)
    parser.add_argument("--num_permutations", type=int, default=200)
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="outputs/temporal_analysis")
    parser.add_argument("--ssv2_root_dir", type=str, required=True)
    parser.add_argument("--data_split", type=str, default="val")
    parser.add_argument("--model_name", type=str, default=None)
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
