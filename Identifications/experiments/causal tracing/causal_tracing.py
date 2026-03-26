import argparse
import json
import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoImageProcessor, TimesformerForVideoClassification


def load_val_csv(path, n_videos, seed=42):
    samples = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                samples.append((parts[0], int(parts[1]), int(parts[2])))
    random.seed(seed)
    if n_videos and n_videos < len(samples):
        samples = random.sample(samples, n_videos)
    print(f"Selected {len(samples)} videos")
    return samples


def load_frames(frames_dir, vid_id, n_total, n_frames=8):
    fdir = os.path.join(frames_dir, vid_id)
    if n_total >= n_frames:
        idxs = np.linspace(0, n_total - 1, n_frames, dtype=int)
    else:
        idxs = list(range(n_total))
        while len(idxs) < n_frames:
            idxs.append(n_total - 1)
        idxs = idxs[:n_frames]

    frames = []
    for i in idxs:
        for fmt in [f"{i+1:05d}.jpg", f"{i+1:04d}.jpg", f"{i+1}.jpg"]:
            fp = os.path.join(fdir, fmt)
            if os.path.exists(fp):
                frames.append(Image.open(fp).convert('RGB'))
                break
        else:
            if frames:
                frames.append(frames[-1].copy())
            else:
                raise FileNotFoundError(f"No frame found in {fdir}")
    return frames


def shuffle_frames(frames):
    out = frames.copy()
    random.shuffle(out)
    return out


def get_attn_layers(model):
    names = []
    for name, mod in model.named_modules():
        if "temporal_attention" in name and type(mod).__name__ == "TimeSformerAttention":
            names.append(name)
    return names


class ActStore:
    def __init__(self):
        self.data = {}

    def save_hook(self, name):
        def hook(mod, inp, out):
            if isinstance(out, tuple):
                self.data[name] = out[0].detach().clone()
            else:
                self.data[name] = out.detach().clone()
        return hook

    def patch_hook(self, name):
        def hook(mod, inp, out):
            saved = self.data[name]
            if isinstance(out, tuple):
                return (saved,) + out[1:]
            return saved
        return hook


def run_forward(model, proc, frames, dev):
    inputs = proc(images=frames, return_tensors="pt")
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.softmax(logits, dim=-1)[0]


def trace_video(model, proc, frames, label, layers, dev):
    store = ActStore()
    mods = dict(model.named_modules())

    # save clean activations
    handles = []
    for name in layers:
        handles.append(mods[name].register_forward_hook(store.save_hook(name)))
    probs = run_forward(model, proc, frames, dev)
    clean_p = probs[label].item()
    for h in handles:
        h.remove()

    # corrupted run (shuffled frames)
    shuffled = shuffle_frames(frames)
    probs = run_forward(model, proc, shuffled, dev)
    corrupt_p = probs[label].item()

    # patch each layer one at a time to see how much it recovers
    scores = {}
    for ln in layers:
        h = mods[ln].register_forward_hook(store.patch_hook(ln))
        probs = run_forward(model, proc, shuffled, dev)
        scores[ln] = probs[label].item() - corrupt_p
        h.remove()

    return scores, clean_p, corrupt_p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--frames_dir', type=str, required=True)
    parser.add_argument('--val_csv', type=str, required=True)
    parser.add_argument('--num_videos', type=int, default=200)
    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument('--output', type=str, default='causal_results.json')
    args = parser.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {dev}")

    proc = AutoImageProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = TimesformerForVideoClassification.from_pretrained(args.model_dir, local_files_only=True)
    model = model.to(dev)
    model.eval()

    layers = get_attn_layers(model)
    print(f"Found {len(layers)} temporal attention layers")
    if not layers:
        print("ERROR: no attention layers found")
        return

    samples = load_val_csv(args.val_csv, args.num_videos)

    all_scores = {name: [] for name in layers}
    clean_ps = []
    corrupt_ps = []
    skipped = 0

    for vid_id, n_frames, label in tqdm(samples):
        try:
            frames = load_frames(args.frames_dir, vid_id, n_frames, args.num_frames)
        except FileNotFoundError:
            skipped += 1
            continue

        scores, cp, crp = trace_video(model, proc, frames, label, layers, dev)
        for name in layers:
            all_scores[name].append(scores[name])
        clean_ps.append(cp)
        corrupt_ps.append(crp)

    avg = {name: float(np.mean(all_scores[name])) for name in layers}
    ranked = sorted(avg.items(), key=lambda x: x[1], reverse=True)

    print(f"\nAvg P(correct) clean: {np.mean(clean_ps):.4f}")
    print(f"Avg P(correct) corrupted: {np.mean(corrupt_ps):.4f}")
    print(f"Skipped: {skipped}")
    print("\nCausal scores (ranked):")
    for name, sc in ranked:
        print(f"  {name}: {sc:.4f}")

    results = {
        "model_dir": args.model_dir,
        "num_videos_traced": len(samples) - skipped,
        "num_videos_skipped": skipped,
        "avg_clean_prob": float(np.mean(clean_ps)),
        "avg_corrupt_prob": float(np.mean(corrupt_ps)),
        "avg_causal_scores": avg,
        "ranked_layers": [{"layer": n, "score": s} for n, s in ranked],
        "top_causal_layer": ranked[0][0],
        "per_video_scores": all_scores,
    }
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
