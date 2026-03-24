#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=Teaching
#SBATCH --time=0-10:00:00

source /home/s2197197/miniconda3/bin/activate testenv
cd /disk/scratch/s2197197
. /home/htang2/toolchain-20251006/toolchain.rc

python /home/s2197197/timesformer/causal_tracing.py \
    --model_dir  /home/s2197197/timesformer/timesformer-model \
    --frames_dir ./evaluation_frames/frames \
    --val_csv    ./evaluation_frames/frame_lists/val.csv \
    --num_videos 200 \
    --output     causal_results.json