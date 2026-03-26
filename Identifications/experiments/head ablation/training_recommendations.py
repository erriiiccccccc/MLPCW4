import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
from scipy.spatial.distance import cosine


def compute_head_redundancy(layer_attns):
    rows = []
    sim_matrices = {}

    for layer_idx in sorted(layer_attns.keys()):
        attn = layer_attns[layer_idx]
        H = attn.shape[1]

        mean_attn = attn.mean(axis=0)
        features = mean_attn.reshape(H, -1)

        sim_matrix = np.zeros((H, H))
        for i in range(H):
            for j in range(H):
                if i == j:
                    sim_matrix[i, j] = 1.0
                else:
                    sim_matrix[i, j] = 1.0 - cosine(features[i], features[j])

        sim_matrices[layer_idx] = sim_matrix

        for i in range(H):
            for j in range(i + 1, H):
                rows.append({
                    "layer": layer_idx,
                    "head_a": i,
                    "head_b": j,
                    "cosine_similarity": float(sim_matrix[i, j]),
                })

    return pd.DataFrame(rows), sim_matrices


def compute_layer_contributions(shapley_df, ablation_df=None):
    rows = []

    for layer_idx in sorted(shapley_df["layer"].unique()):
        layer_shapley = shapley_df[shapley_df["layer"] == layer_idx]
        total_shapley = layer_shapley["shapley_value"].sum()
        mean_shapley = layer_shapley["shapley_value"].mean()
        max_shapley = layer_shapley["shapley_value"].max()
        min_shapley = layer_shapley["shapley_value"].min()

        row = {
            "layer": layer_idx,
            "temporal_total_shapley": total_shapley,
            "temporal_mean_shapley": mean_shapley,
            "temporal_max_shapley": max_shapley,
            "temporal_min_shapley": min_shapley,
            "temporal_shapley_range": max_shapley - min_shapley,
        }

        if ablation_df is not None:
            temporal_abl = ablation_df[
                (ablation_df["layer"] == layer_idx) &
                (ablation_df["attn_type"] == "temporal")
            ]
            spatial_abl = ablation_df[
                (ablation_df["layer"] == layer_idx) &
                (ablation_df["attn_type"] == "spatial")
            ]
            if len(temporal_abl) > 0:
                row["temporal_mean_flip"] = temporal_abl["flip_rate"].mean()
            if len(spatial_abl) > 0:
                row["spatial_mean_flip"] = spatial_abl["flip_rate"].mean()
            if len(temporal_abl) > 0 and len(spatial_abl) > 0:
                row["temporal_spatial_ratio"] = (
                    temporal_abl["flip_rate"].mean() /
                    (spatial_abl["flip_rate"].mean() + 1e-8)
                )

        rows.append(row)

    return pd.DataFrame(rows)


def simulate_gate_values(shapley_df):
    """simulate what optimal temporal gate values would look like based on shapley"""
    layer_totals = shapley_df.groupby("layer")["shapley_value"].sum()
    max_total = layer_totals.abs().max()

    rows = []
    for layer_idx, total in layer_totals.items():
        normalized = total / (max_total + 1e-8)
        suggested_gate = max(0.0, min(1.0, float(normalized)))

        rows.append({
            "layer": int(layer_idx),
            "total_shapley": float(total),
            "simulated_gate": suggested_gate,
            "gate_recommendation": (
                "increase" if suggested_gate > 0.7
                else "decrease" if suggested_gate < 0.3
                else "neutral"
            ),
        })

    return pd.DataFrame(rows)


def compute_pruning_recommendations(shapley_df, threshold_pct=1.0):
    df = shapley_df.copy()
    df["abs_shapley"] = df["shapley_value"].abs()

    df["layer_rank"] = df.groupby("layer")["abs_shapley"].rank(ascending=False)

    layer_totals = df.groupby("layer")["shapley_value"].sum()
    df["layer_total"] = df["layer"].map(layer_totals)

    df["contribution_pct"] = (
        df["shapley_value"].abs() / (df["layer_total"].abs() + 1e-8) * 100
    )

    df["prunable"] = df["contribution_pct"] < threshold_pct
    df["confidence"] = "high" if "stderr" not in df.columns else None
    if "stderr" in df.columns:
        df["confidence"] = np.where(
            df["abs_shapley"] < 2 * df["stderr"],
            "high",   # confidently near zero = safe to prune
            "low"     # significantly nonzero = risky
        )

    return df[["layer", "head", "shapley_value", "abs_shapley", "layer_rank",
               "contribution_pct", "prunable", "confidence"]].sort_values(
        ["layer", "abs_shapley"], ascending=[True, True]
    )


def compute_strengthening_recommendations(
    shapley_df, semantics_df=None, redundancy_df=None,
):
    """find weak-but-unique heads that could benefit from strengthening"""
    df = shapley_df[["layer", "head", "shapley_value"]].copy()
    df["abs_shapley"] = df["shapley_value"].abs()

    median_shapley = df["abs_shapley"].median()
    df["is_weak"] = df["abs_shapley"] < median_shapley

    if redundancy_df is not None and len(redundancy_df) > 0:
        uniqueness = []
        for _, row in df.iterrows():
            layer, head = int(row["layer"]), int(row["head"])
            layer_red = redundancy_df[redundancy_df["layer"] == layer]
            sims = layer_red[
                (layer_red["head_a"] == head) | (layer_red["head_b"] == head)
            ]["cosine_similarity"]
            mean_sim = sims.mean() if len(sims) > 0 else 0.5
            uniqueness.append(1.0 - mean_sim)
        df["uniqueness_score"] = uniqueness
    else:
        df["uniqueness_score"] = 0.5

    if semantics_df is not None and "specialization_score" in semantics_df.columns:
        spec_map = semantics_df.set_index(["layer", "head"])["specialization_score"]
        df["specialization"] = df.apply(
            lambda r: spec_map.get((r["layer"], r["head"]), 0.5), axis=1
        )
    else:
        df["specialization"] = 0.5

    df["strengthen_score"] = (
        df["is_weak"].astype(float) * 0.4
        + df["uniqueness_score"] * 0.3
        + df["specialization"] * 0.3
    )

    df["recommendation"] = np.where(
        (df["is_weak"]) & (df["uniqueness_score"] > 0.3),
        "strengthen",
        np.where(
            (df["is_weak"]) & (df["uniqueness_score"] <= 0.3),
            "consider_pruning",
            "keep"
        )
    )

    return df.sort_values("strengthen_score", ascending=False)


def run_training_recommendations(
    shapley_path="outputs/temporal_analysis/shapley_values.csv",
    semantics_path="outputs/temporal_analysis/temporal_semantics.csv",
    ablation_path="outputs/ssv2_ablation/ablation_results.csv",
    attention_dir="outputs/temporal_analysis",
    output_dir="outputs/temporal_analysis",
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Training Recommendations")

    start_time = time.time()
    results = {}

    shapley_path = Path(shapley_path)
    if not shapley_path.exists():
        print(f"ERROR: Shapley values not found at {shapley_path}")
        print("Run shapley_importance.py first.")
        return results
    shapley_df = pd.read_csv(shapley_path)
    print(f"\nLoaded Shapley values: {len(shapley_df)} entries, "
          f"layers {sorted(shapley_df['layer'].unique())}")

    semantics_df = None
    semantics_p = Path(semantics_path)
    if semantics_p.exists():
        semantics_df = pd.read_csv(semantics_p)
        print(f"Loaded semantics: {len(semantics_df)} entries")

    ablation_df = None
    ablation_p = Path(ablation_path)
    if ablation_p.exists():
        ablation_df = pd.read_csv(ablation_p)
        print(f"Loaded ablation results: {len(ablation_df)} entries")

    layer_attns = _load_attention_patterns(attention_dir)

    print("\n--- Head Redundancy Matrix ---")
    if layer_attns:
        redundancy_df, sim_matrices = compute_head_redundancy(layer_attns)
        redundancy_df.to_csv(output_path / "head_redundancy.csv", index=False)
        results["redundancy"] = redundancy_df

        for layer_idx, sim in sim_matrices.items():
            np.save(output_path / f"sim_matrix_L{layer_idx}.npy", sim)

        high_sim = redundancy_df[redundancy_df["cosine_similarity"] > 0.9]
        print(f"  Total pairs: {len(redundancy_df)}")
        print(f"  Highly redundant (sim > 0.9): {len(high_sim)}")
        if len(high_sim) > 0:
            for _, row in high_sim.head(5).iterrows():
                print(f"    L{int(row['layer'])}-H{int(row['head_a'])} <-> "
                      f"H{int(row['head_b'])}: {row['cosine_similarity']:.3f}")
    else:
        print("  Skipped -- no attention patterns found. Run temporal_semantics.py first.")
        redundancy_df = pd.DataFrame()

    print("\n--- Layer Contribution Analysis ---")
    layer_df = compute_layer_contributions(shapley_df, ablation_df)
    layer_df.to_csv(output_path / "layer_contributions.csv", index=False)
    results["layer_contributions"] = layer_df
    for _, row in layer_df.iterrows():
        line = f"  L{int(row['layer'])}: Shapley total={row['temporal_total_shapley']:.4f}"
        if "temporal_spatial_ratio" in row and pd.notna(row.get("temporal_spatial_ratio")):
            line += f", T/S ratio={row['temporal_spatial_ratio']:.2f}"
        print(line)

    print("\n--- Gate Value Simulation ---")
    gate_df = simulate_gate_values(shapley_df)
    gate_df.to_csv(output_path / "gate_simulation.csv", index=False)
    results["gate_simulation"] = gate_df
    for _, row in gate_df.iterrows():
        print(f"  L{int(row['layer'])}: gate={row['simulated_gate']:.3f} "
              f"({row['gate_recommendation']})")

    print("\n--- Pruning Recommendations ---")
    prune_df = compute_pruning_recommendations(shapley_df)
    prune_df.to_csv(output_path / "pruning_recommendations.csv", index=False)
    results["pruning"] = prune_df
    prunable = prune_df[prune_df["prunable"]]
    print(f"  Prunable heads (<1% contribution): {len(prunable)} / {len(prune_df)}")
    if len(prunable) > 0:
        for _, row in prunable.head(5).iterrows():
            print(f"    L{int(row['layer'])}-H{int(row['head'])}: "
                  f"|phi|={row['abs_shapley']:.4f}, "
                  f"contrib={row['contribution_pct']:.2f}%")

    print("\n--- Strengthening Recommendations ---")
    strengthen_df = compute_strengthening_recommendations(
        shapley_df, semantics_df, redundancy_df
    )
    strengthen_df.to_csv(output_path / "strengthening_recommendations.csv", index=False)
    results["strengthening"] = strengthen_df
    to_strengthen = strengthen_df[strengthen_df["recommendation"] == "strengthen"]
    print(f"  Candidates for strengthening: {len(to_strengthen)}")
    if len(to_strengthen) > 0:
        for _, row in to_strengthen.head(5).iterrows():
            print(f"    L{int(row['layer'])}-H{int(row['head'])}: "
                  f"score={row['strengthen_score']:.3f}, "
                  f"uniqueness={row['uniqueness_score']:.3f}")

    report = _generate_report(results, shapley_df, layer_attns)
    report_path = output_path / "training_recommendations.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")

    elapsed = time.time() - start_time
    print(f"\nTRAINING RECOMMENDATIONS done in {elapsed:.1f}s")

    return results


def _load_attention_patterns(attention_dir):
    attn_dir = Path(attention_dir)
    layer_attns = {}

    for npy_file in sorted(attn_dir.glob("attention_L*.npy")):
        layer_idx = int(npy_file.stem.split("_L")[1])
        layer_attns[layer_idx] = np.load(npy_file)

    if layer_attns:
        print(f"Loaded attention patterns: {len(layer_attns)} layers")

    return layer_attns


def _generate_report(results, shapley_df, layer_attns):
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "layers_analyzed": sorted(shapley_df["layer"].unique().tolist()),
        "total_heads_analyzed": len(shapley_df),
    }

    top5 = shapley_df.nlargest(5, "shapley_value")
    bottom5 = shapley_df.nsmallest(5, "shapley_value")
    report["top_5_heads"] = [
        {"layer": int(r["layer"]), "head": int(r["head"]),
         "shapley_value": round(r["shapley_value"], 6)}
        for _, r in top5.iterrows()
    ]
    report["bottom_5_heads"] = [
        {"layer": int(r["layer"]), "head": int(r["head"]),
         "shapley_value": round(r["shapley_value"], 6)}
        for _, r in bottom5.iterrows()
    ]

    if "pruning" in results:
        prune_df = results["pruning"]
        report["pruning"] = {
            "prunable_count": int(prune_df["prunable"].sum()),
            "total_heads": len(prune_df),
            "prunable_heads": [
                {"layer": int(r["layer"]), "head": int(r["head"]),
                 "contribution_pct": round(r["contribution_pct"], 4)}
                for _, r in prune_df[prune_df["prunable"]].iterrows()
            ],
        }

    if "strengthening" in results:
        str_df = results["strengthening"]
        to_str = str_df[str_df["recommendation"] == "strengthen"]
        report["strengthening"] = {
            "candidates_count": len(to_str),
            "candidates": [
                {"layer": int(r["layer"]), "head": int(r["head"]),
                 "strengthen_score": round(r["strengthen_score"], 4),
                 "uniqueness": round(r["uniqueness_score"], 4)}
                for _, r in to_str.iterrows()
            ],
        }

    if "gate_simulation" in results:
        gate_df = results["gate_simulation"]
        report["gate_recommendations"] = [
            {"layer": int(r["layer"]),
             "simulated_gate": round(r["simulated_gate"], 4),
             "recommendation": r["gate_recommendation"]}
            for _, r in gate_df.iterrows()
        ]

    if "redundancy" in results and len(results["redundancy"]) > 0:
        red_df = results["redundancy"]
        high_sim = red_df[red_df["cosine_similarity"] > 0.9]
        report["redundancy"] = {
            "total_pairs": len(red_df),
            "highly_redundant_pairs": len(high_sim),
            "mean_similarity": round(red_df["cosine_similarity"].mean(), 4),
        }

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--shapley_path", type=str,
                        default="outputs/temporal_analysis/shapley_values.csv")
    parser.add_argument("--semantics_path", type=str,
                        default="outputs/temporal_analysis/temporal_semantics.csv")
    parser.add_argument("--ablation_path", type=str,
                        default="outputs/ssv2_ablation/ablation_results.csv")
    parser.add_argument("--attention_dir", type=str,
                        default="outputs/temporal_analysis")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/temporal_analysis")
    args = parser.parse_args()

    run_training_recommendations(
        shapley_path=args.shapley_path,
        semantics_path=args.semantics_path,
        ablation_path=args.ablation_path,
        attention_dir=args.attention_dir,
        output_dir=args.output_dir,
    )
