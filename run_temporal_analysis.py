"""main entry point for the full temporal analysis pipeline.

runs all phases in order (or just the ones you specify via --phases):
    phase 1: shapley values (shapley_importance.py)
    phase 2: temporal attn semantics (temporal_semantics.py)
    phase 3: training recommendations (training_recommendations.py)
    phase 4: visualisations (visualize_temporal.py)
    phase 5: downstream propagation (downstream_propagation.py)
"""

import json
import time
import argparse
from pathlib import Path
from typing import List, Optional


def run_pipeline(
    phases: List[int],
    ssv2_root_dir: str,
    num_videos: int = 20,
    num_permutations: int = 200,
    layers: Optional[List[int]] = None,
    batch_size: int = 2,
    output_dir: str = "outputs/temporal_analysis",
    data_split: str = "val",
    model_name: Optional[str] = None,
) -> dict:
    """runs the full analysis pipeline for the specified phases.

    phases: list of ints 1-5 (which phases to run)
    ssv2_root_dir: path to evaluation_frames/ on the cluster
    num_videos: how many videos to eval on (20 is usually enough)
    num_permutations: permutations per layer for shapley (more = more accurate but slower)
    layers: specific layers to analyze, default is all 12
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # shared dataset kwargs - both phase 1 and 2 need these
    data_kwargs = dict(
        ssv2_root_dir=ssv2_root_dir,
        data_split=data_split,
    )
    if model_name:
        data_kwargs["model_name"] = model_name

    print("=" * 60)
    print("  TimeSformer Temporal Head Analysis Pipeline")
    print("=" * 60)
    print(f"  Phases: {phases}")
    print(f"  Videos: {num_videos}")
    print(f"  Permutations: {num_permutations}")
    print(f"  Layers: {layers or 'all 12'}")
    print(f"  Data: SSv2 ({ssv2_root_dir})")
    print(f"  Split: {data_split}")
    print(f"  Output: {output_path}")
    print()

    start_time = time.time()
    summary = {
        "phases_run": phases,
        "num_videos": num_videos,
        "num_permutations": num_permutations,
        "layers": layers or list(range(12)),
        "dataset": "ssv2",
        "ssv2_root_dir": ssv2_root_dir,
        "data_split": data_split,
    }

    # Phase 1: Shapley Values
    if 1 in phases:
        print("\n" + "=" * 60)
        print("  PHASE 1: Shapley Value Computation")
        print("=" * 60)
        from shapley_importance import run_shapley

        shapley_df = run_shapley(
            num_videos=num_videos,
            num_permutations=num_permutations,
            layers=layers,
            batch_size=batch_size,
            output_dir=output_dir,
            **data_kwargs,
        )
        summary["shapley"] = {
            "heads_computed": len(shapley_df),
            "layers": sorted(shapley_df["layer"].unique().tolist()),
            "max_shapley": float(shapley_df["shapley_value"].max()),
            "min_shapley": float(shapley_df["shapley_value"].min()),
        }

    # Phase 2: Temporal Semantics
    if 2 in phases:
        print("\n" + "=" * 60)
        print("  PHASE 2: Temporal Semantic Analysis")
        print("=" * 60)
        from temporal_semantics import run_temporal_semantics

        sem_results = run_temporal_semantics(
            num_videos=num_videos,
            batch_size=batch_size,
            output_dir=output_dir,
            **data_kwargs,
        )
        if "combined" in sem_results:
            combined = sem_results["combined"]
            summary["semantics"] = {
                "heads_analyzed": len(combined),
                "pattern_types": (
                    combined["pattern_type"].value_counts().to_dict()
                    if "pattern_type" in combined.columns else {}
                ),
                "mean_specialization": float(
                    combined["specialization_score"].mean()
                ) if "specialization_score" in combined.columns else None,
            }

    # Phase 3: Training Recommendations
    if 3 in phases:
        print("\n" + "=" * 60)
        print("  PHASE 3: Training Recommendations")
        print("=" * 60)
        from training_recommendations import run_training_recommendations

        rec_results = run_training_recommendations(
            shapley_path=str(output_path / "shapley_values.csv"),
            semantics_path=str(output_path / "temporal_semantics.csv"),
            ablation_path=str(output_path / "ablation_results.csv"),
            attention_dir=str(output_path),
            output_dir=output_dir,
        )
        if "pruning" in rec_results:
            prune_df = rec_results["pruning"]
            summary["recommendations"] = {
                "prunable_heads": int(prune_df["prunable"].sum()),
                "total_heads": len(prune_df),
            }
        if "strengthening" in rec_results:
            str_df = rec_results["strengthening"]
            summary.setdefault("recommendations", {})
            summary["recommendations"]["strengthen_candidates"] = int(
                (str_df["recommendation"] == "strengthen").sum()
            )

    # Phase 4: Visualizations
    if 4 in phases:
        print("\n" + "=" * 60)
        print("  PHASE 4: Visualizations")
        print("=" * 60)
        from visualize_temporal import run_all_visualizations

        run_all_visualizations(output_dir=output_dir)

    # Phase 5: Downstream Propagation
    if 5 in phases:
        print("\n" + "=" * 60)
        print("  PHASE 5: Downstream Propagation Analysis")
        print("=" * 60)
        from downstream_propagation import run_downstream_propagation

        prop_df = run_downstream_propagation(
            num_videos=num_videos,
            source_layers=layers,
            batch_size=batch_size,
            output_dir=output_dir,
            **data_kwargs,
        )
        downstream_only = prop_df[
            prop_df["target_layer"] > prop_df["source_layer"]
        ]
        summary["propagation"] = {
            "total_entries": len(prop_df),
            "mean_downstream_cosine": float(
                downstream_only["cosine_sim"].mean()
            ) if len(downstream_only) > 0 else None,
            "mean_downstream_cka": float(
                downstream_only["cka"].mean()
            ) if len(downstream_only) > 0 and "cka" in downstream_only.columns else None,
        }

    # Final summary
    elapsed = time.time() - start_time
    summary["elapsed_seconds"] = round(elapsed, 1)

    summary_path = output_path / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  PIPELINE COMPLETE — {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 60}")
    print(f"\nSummary saved to {summary_path}")

    # List output files
    print(f"\nOutput files in {output_path}:")
    for p in sorted(output_path.iterdir()):
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name} ({size_kb:.1f} KB)")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run TimeSformer temporal head analysis pipeline"
    )
    parser.add_argument("--ssv2_root_dir", type=str, required=True,
                        help="Path to evaluation_frames/ directory")
    parser.add_argument("--phases", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                        help="Phases to run: 1=Shapley, 2=Semantics, "
                             "3=Recommendations, 4=Visualizations, "
                             "5=Propagation")
    parser.add_argument("--num_videos", type=int, default=20)
    parser.add_argument("--num_permutations", type=int, default=200)
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output_dir", type=str,
                        default="outputs/temporal_analysis")
    parser.add_argument("--data_split", type=str, default="val",
                        help="Dataset split (test, val, train)")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model name or local path (default: config.py default)")
    args = parser.parse_args()

    run_pipeline(
        phases=args.phases,
        ssv2_root_dir=args.ssv2_root_dir,
        num_videos=args.num_videos,
        num_permutations=args.num_permutations,
        layers=args.layers,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        data_split=args.data_split,
        model_name=args.model_name,
    )
