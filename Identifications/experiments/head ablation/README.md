# Temporal Head Ablation

runs the temporal head ablation pipeline on TimeSformer (SSv2 finetuned). we use divided space-time attention so temporal heads can be ablated independantly from spatial ones.

## Prerequisites

- Python 3.9+
- CUDA GPU (RTX 2080 Ti or better; full shapley run takes ~11h)
- SSv2 pre-extracted frames at `/disk/scratch/MLPG102/evaluation_frames/`
- `pip install -r requirements.txt`


## Offline model weights

`model_local/` has the finetuned TimeSformer weights (`model.safetensors`, config, preprocessor).
scripts will load from here if theres no network access.

## Running the pipeline

```bash
# step 1: run cpu tests first (no gpu needed)
bash run_tests.sh --cpu-only

# step 2: full shapley computation (~11h on the cluster)
sbatch run_shapley_full.sh

# step 3: semantics, ablation sweep, recommendations, viz, propagation
sbatch run_analysis.sh

# or just run everything at once:
python run_temporal_analysis.py --ssv2_root_dir /path/to/evaluation_frames --num_videos 20
```

## Key output files

outputs go to `outputs/temporal_analysis/`:

| File | Contents |
|---|---|
| `shapley_values.csv` | per-head shapley values (φ, SE, CI) for all 144 temporal heads |
| `shapley_heatmap.png` | 12x12 heatmap - layers vs heads |
| `propagation_classification.csv` | amplifying / stable / dampening per source layer |
| `propagation_effects.csv` | per (layer, head, target_layer) L2/CKA/JSD metrics |
| `head_redundancy.csv` | pairwise cosine similarity within each layer |
| `pruning_recommendations.csv` | heads safe to prune (<1% layer contribution) |

## Tests

```bash
pytest tests/ -v   # 141 cpu tests pass; 3 gpu-only tests will skip without a gpu
```
