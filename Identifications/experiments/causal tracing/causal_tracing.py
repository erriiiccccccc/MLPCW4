#!/usr/bin/env python3
"""
Causal Tracing for TimeSformer on SSv2
=======================================
Identifies which attention layers are causally responsible for
temporal reasoning by patching clean activations into corrupted runs.

Protocol per video:
  1. Clean run    → save hidden states from all attention layers
  2. Corrupt run  → shuffle frames, record drop in correct class prob
  3. Patch loop   → restore one layer at a time, measure recovery

Causal effect score per layer:
  score = P(correct | patched at layer L) - P(correct | corrupted)

High score = that layer is causally responsible for temporal understanding.

Usage:
    python causal_tracing.py \
        --model_dir /disk/scratch/s2197197/timesformer/timesformer-model \
        --frames_dir /disk/scratch/s2197197/processed/frames \
        --val_csv    /disk/scratch/s2197197/processed/frame_lists/val.csv \
        --num_videos 200 \
        --output     causal_results.json
"""

import os
import json
import random
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, TimesformerForVideoClassification


# ─────────────────────────────────────────────
# Data loading  (mirrors step3_full_evaluation)
# ─────────────────────────────────────────────

def load_val_csv(val_csv, num_videos, seed=42):
    """Parse val.csv and sample num_videos entries."""
    samples = []
    with open(val_csv, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                samples.append((parts[0], int(parts[1]), int(parts[2])))

    random.seed(seed)
    if num_videos and num_videos < len(samples):
        samples = random.sample(samples, num_videos)

    print(f"Selected {len(samples)} videos for causal tracing")
    return samples


def load_frames(frames_dir, video_id, num_video_frames, num_frames=8):
    """Uniformly sample num_frames PIL images from a video's frame folder."""
    frame_dir = os.path.join(frames_dir, video_id)

    if num_video_frames >= num_frames:
        indices = np.linspace(0, num_video_frames - 1, num_frames, dtype=int)
    else:
        indices = list(range(num_video_frames))
        while len(indices) < num_frames:
            indices.append(num_video_frames - 1)
        indices = indices[:num_frames]

    frames = []
    for idx in indices:
        for fmt in [f"{idx+1:05d}.jpg", f"{idx+1:04d}.jpg", f"{idx+1}.jpg"]:
            fp = os.path.join(frame_dir, fmt)
            if os.path.exists(fp):
                frames.append(Image.open(fp).convert('RGB'))
                break
        else:
            if frames:
                frames.append(frames[-1].copy())
            else:
                raise FileNotFoundError(f"No frame found in {frame_dir}")

    return frames


def corrupt_frames(frames):
    """Temporal corruption: shuffle frame order."""
    shuffled = frames.copy()
    random.shuffle(shuffled)
    return shuffled


# ─────────────────────────────────────────────
# Hook utilities
# ─────────────────────────────────────────────

def get_attention_layer_names(model):
    """
    Return names of temporal attention modules in HuggingFace TimeSformer.
    Confirmed layer structure: timesformer.encoder.layer.N.temporal_attention
    These are TimeSformerAttention modules — one per encoder layer (12 total).
    """
    names = []
    for name, module in model.named_modules():
        if (
            "temporal_attention" in name
            and type(module).__name__ == "TimeSformerAttention"
        ):
            names.append(name)
    return names


class ActivationStore:
    """Holds saved activations for one forward pass."""
    def __init__(self):
        self.store = {}

    def save_hook(self, name):
        def hook(module, input, output):
            # output is typically a tuple; first element is the hidden state
            if isinstance(output, tuple):
                self.store[name] = output[0].detach().clone()
            else:
                self.store[name] = output.detach().clone()
        return hook

    def patch_hook(self, name):
        """Return a hook that replaces this layer's output with the saved clean version."""
        def hook(module, input, output):
            saved = self.store[name]
            if isinstance(output, tuple):
                return (saved,) + output[1:]
            return saved
        return hook


# ─────────────────────────────────────────────
# Core causal tracing logic
# ─────────────────────────────────────────────

def run_forward(model, processor, frames, device):
    """Run one forward pass, return probability vector."""
    inputs = processor(images=frames, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.softmax(logits, dim=-1)[0]   # shape: (num_classes,)


def causal_trace_single_video(model, processor, frames, true_label,
                               layer_names, device):
    """
    Run the full 3-phase causal tracing protocol on one video.

    Returns dict: {layer_name: causal_effect_score}
    Also returns clean_prob and corrupted_prob for reference.
    """
    store = ActivationStore()

    # ── Phase 1: Clean run — save all attention outputs ──────────────────
    handles = []
    for name in layer_names:
        module = dict(model.named_modules())[name]
        handles.append(module.register_forward_hook(store.save_hook(name)))

    clean_probs = run_forward(model, processor, frames, device)
    clean_prob  = clean_probs[true_label].item()

    for h in handles:
        h.remove()

    # ── Phase 2: Corrupt run — shuffled frames, no patching ──────────────
    shuffled = corrupt_frames(frames)
    corrupt_probs = run_forward(model, processor, shuffled, device)
    corrupt_prob  = corrupt_probs[true_label].item()

    # ── Phase 3: Patch loop — restore one layer at a time ────────────────
    causal_scores = {}

    for layer_name in layer_names:
        module = dict(model.named_modules())[layer_name]
        handle = module.register_forward_hook(store.patch_hook(layer_name))

        patched_probs = run_forward(model, processor, shuffled, device)
        patched_prob  = patched_probs[true_label].item()

        handle.remove()

        # Causal effect = how much did patching this layer recover correct prob
        causal_scores[layer_name] = patched_prob - corrupt_prob

    return causal_scores, clean_prob, corrupt_prob


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Causal tracing for TimeSformer on SSv2')
    parser.add_argument('--model_dir',  type=str, required=True,
                        help='Path to local HuggingFace model directory')
    parser.add_argument('--frames_dir', type=str, required=True,
                        help='Directory containing extracted frames')
    parser.add_argument('--val_csv',    type=str, required=True,
                        help='Path to val.csv')
    parser.add_argument('--num_videos', type=int, default=200,
                        help='Number of validation videos to trace (default: 200)')
    parser.add_argument('--num_frames', type=int, default=8,
                        help='Frames per video (default: 8, matches training)')
    parser.add_argument('--output',     type=str, default='causal_results.json',
                        help='Output JSON file for results')
    args = parser.parse_args()

    print("=" * 60)
    print("TimeSformer Causal Tracing")
    print("=" * 60)
    print(f"  Model:      {args.model_dir}")
    print(f"  Frames dir: {args.frames_dir}")
    print(f"  Val CSV:    {args.val_csv}")
    print(f"  Videos:     {args.num_videos}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device:     {device}")

    # Load model
    print("\nLoading model...")
    processor = AutoImageProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = TimesformerForVideoClassification.from_pretrained(
        args.model_dir, local_files_only=True
    )
    model = model.to(device)
    model.eval()
    print("Model loaded.")

    # Identify temporal attention layers
    layer_names = get_attention_layer_names(model)
    print(f"\nFound {len(layer_names)} attention layers to probe:")
    for n in layer_names:
        print(f"  {n}")

    if len(layer_names) == 0:
        print("\nERROR: No attention layers found.")
        print("Run this to inspect layer names:")
        print("  python -c \"from transformers import TimesformerForVideoClassification; "
              "m = TimesformerForVideoClassification.from_pretrained('<dir>'); "
              "[print(n) for n,_ in m.named_modules()]\"")
        return

    # Load video list
    samples = load_val_csv(args.val_csv, args.num_videos)

    # Accumulate causal scores across videos
    accumulated  = {name: [] for name in layer_names}
    clean_probs_all   = []
    corrupt_probs_all = []
    skipped = 0

    print(f"\nRunning causal tracing on {len(samples)} videos...")

    for video_id, num_video_frames, true_label in tqdm(samples):
        try:
            frames = load_frames(args.frames_dir, video_id,
                                 num_video_frames, args.num_frames)
        except FileNotFoundError as e:
            skipped += 1
            continue

        scores, clean_p, corrupt_p = causal_trace_single_video(
            model, processor, frames, true_label, layer_names, device
        )

        for name in layer_names:
            accumulated[name].append(scores[name])

        clean_probs_all.append(clean_p)
        corrupt_probs_all.append(corrupt_p)

    # ── Aggregate ──────────────────────────────────────────────────────────
    avg_scores = {name: float(np.mean(accumulated[name]))
                  for name in layer_names}

    # Rank layers by causal importance
    ranked = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 60)
    print("CAUSAL TRACING RESULTS")
    print("=" * 60)
    print(f"\nAvg P(correct) — clean:     {np.mean(clean_probs_all):.4f}")
    print(f"Avg P(correct) — corrupted: {np.mean(corrupt_probs_all):.4f}")
    print(f"Avg drop from corruption:   {np.mean(clean_probs_all) - np.mean(corrupt_probs_all):.4f}")
    print(f"\nSkipped (missing frames): {skipped}")

    print("\nLayer causal effect scores (ranked):")
    print(f"  {'Layer':<55} {'Score':>8}")
    print("  " + "-" * 65)
    for name, score in ranked:
        marker = "  ← TOP" if name == ranked[0][0] else ""
        print(f"  {name:<55} {score:>8.4f}{marker}")

    print(f"\nMost causally important layer: {ranked[0][0]}")
    print(f"Least causally important layer: {ranked[-1][0]}")

    # ── Save results ───────────────────────────────────────────────────────
    results = {
        "model_dir":             args.model_dir,
        "num_videos_traced":     len(samples) - skipped,
        "num_videos_skipped":    skipped,
        "avg_clean_prob":        float(np.mean(clean_probs_all)),
        "avg_corrupt_prob":      float(np.mean(corrupt_probs_all)),
        "avg_causal_scores":     avg_scores,
        "ranked_layers":         [{"layer": n, "score": s} for n, s in ranked],
        "top_causal_layer":      ranked[0][0],
        "per_video_scores":      accumulated,
    }

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nFull results saved to: {args.output}")
    print("\nNext step: use top_causal_layer result to decide which")
    print("attention layers to modify with your causal mask.")


if __name__ == '__main__':
    main()