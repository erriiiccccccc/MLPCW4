#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=Teaching
#SBATCH --time=0-24:00:00

source /home/s2197197/miniconda3/bin/activate testenv
cd /disk/scratch/s2197197
. /home/htang2/toolchain-20251006/toolchain.rc

python -u /home/s2197197/timesformer/finetune_realloc.py \
    --model_dir  /home/s2197197/timesformer/timesformer-model \
    --frames_dir /disk/scratch/s2197197/evaluation_frames/frames \
    --train_csv  /disk/scratch/s2197197/evaluation_frames/frame_lists/train.csv \
    --output_dir /home/s2197197/timesformer/checkpoints_realloc \
    --batch_size 8 \
    --epochs 5 \
    --num_workers 0 \
    --calib_videos 50
