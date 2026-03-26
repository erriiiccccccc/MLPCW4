import json
import time
import argparse
from pathlib import Path
from typing import List, Optional


def run_pipeline(
    phases, ssv2_root_dir,
    num_videos=20, num_permutations=200,
    layers=None, batch_size=2,
    output_dir="outputs/temporal_analysis",
    data_split="val", model_name=None,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data_kwargs = dict(ssv2_root_dir=ssv2_root_dir, data_split=data_split)
    if model_name:
        data_kwargs["model_name"] = model_name

    print(f"TimeSformer Temporal Head Analysis Pipeline")
    print(f"  Phases: {phases}, Videos: {num_videos}, Perms: {num_permutations}")
    print(f"  Layers: {layers or 'all 12'}, Data: SSv2 ({ssv2_root_dir})")
    print(f"  Split: {data_split}, Output: {output_path}")

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

    if 1 in phases:
        print("\nPHASE 1: Shapley Value Computation")
        from shapley_importance import run_shapley

        shapley_df = run_shapley(
            num_videos=num_videos,
            num_permutations=num_permutations,
            layers=layers, batch_size=batch_size,
            output_dir=output_dir, **data_kwargs,
        )
        summary["shapley"] = {
            "heads_computed": len(shapley_df),
            "layers": sorted(shapley_df["layer"].unique().tolist()),
            "max_shapley": float(shapley_df["shapley_value"].max()),
            "min_shapley": float(shapley_df["shapley_value"].min()),
        }

    if 2 in phases:
        print("\nPHASE 2: Temporal Semantic Analysis")
        from temporal_semantics import run_temporal_semantics

        sem_results = run_temporal_semantics(
            num_videos=num_videos, batch_size=batch_size,
            output_dir=output_dir, **data_kwargs,
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

    if 3 in phases:
        print("\nPHASE 3: Training Recommendations")
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

    if 4 in phases:
        print("\nPHASE 4: Visualizations")
        from visualize_temporal import run_all_visualizations
        run_all_visualizations(output_dir=output_dir)

    if 5 in phases:
        print("\nPHASE 5: Downstream Propagation Analysis")
        from downstream_propagation import run_downstream_propagation

        prop_df = run_downstream_propagation(
            num_videos=num_videos, source_layers=layers,
            batch_size=batch_size, output_dir=output_dir,
            **data_kwargs,
        )
        downstream_only = prop_df[prop_df["target_layer"] > prop_df["source_layer"]]
        summary["propagation"] = {
            "total_entries": len(prop_df),
            "mean_downstream_cosine": float(
                downstream_only["cosine_sim"].mean()
            ) if len(downstream_only) > 0 else None,
            "mean_downstream_cka": float(
                downstream_only["cka"].mean()
            ) if len(downstream_only) > 0 and "cka" in downstream_only.columns else None,
        }

    elapsed = time.time() - start_time
    summary["elapsed_seconds"] = round(elapsed, 1)

    summary_path = output_path / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPIPELINE done in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Summary saved to {summary_path}")

    print(f"\nOutput files in {output_path}:")
    for p in sorted(output_path.iterdir()):
        print(f"  {p.name} ({p.stat().st_size / 1024:.1f} KB)")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssv2_root_dir", type=str, required=True)
    parser.add_argument("--phases", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--num_videos", type=int, default=20)
    parser.add_argument("--num_permutations", type=int, default=200)
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="outputs/temporal_analysis")
    parser.add_argument("--data_split", type=str, default="val")
    parser.add_argument("--model_name", type=str, default=None)
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
