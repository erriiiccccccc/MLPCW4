"""
Shared utilities for Group B temporal fine-tuning experiments.
Dataset, evaluate (paper protocol), freeze helpers, result saving.
"""

import os, sys, json, time, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image, ImageFile
from transformers import TimesformerForVideoClassification, AutoImageProcessor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../setup'))
from step3_full_evaluation import SSv2Dataset as _SSv2Eval, collate_fn

ImageFile.LOAD_TRUNCATED_IMAGES = True

MODEL_DIR   = '/home/s2411221/MLPCW4/timesformer-model'
FRAMES_DIR  = '/disk/scratch/MLPG102/evaluation_frames/frames'
TRAIN_CSV   = '/disk/scratch/MLPG102/evaluation_frames/frame_lists/train.csv'
TEST_CSV    = '/disk/scratch/MLPG102/evaluation_frames/frame_lists/test.csv'
RESULTS_DIR = '/home/s2411221/temporal_experiments/results'
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training hypers (same for all Group B experiments)
TRAIN_SUBSET = 0.15
BATCH_TRAIN  = 8
BATCH_EVAL   = 8
EPOCHS       = 5
BASE_LR      = 1e-4
BASE_WD      = 5e-4
NUM_WORKERS  = 4
NUM_FRAMES   = 8
SEED         = 42

# Per-layer Shapley sums (Session 9, 12-layer run)
SHAPLEY_SUMS = {
    0: 0.056, 1: 0.057, 2: 0.045, 3: 0.077,
    4: 0.104, 5: 0.052, 6: 0.126, 7: 0.049,
    8: 0.077, 9: 0.170, 10: 0.194, 11: 0.449
}

os.makedirs(RESULTS_DIR, exist_ok=True)


# ── TRAINING DATASET (processor-based, faster for training loop) ──────────────
class SSv2TrainDataset(Dataset):
    def __init__(self, csv_path, frames_dir, processor, num_frames=8):
        self.frames_dir = frames_dir
        self.processor  = processor
        self.num_frames = num_frames
        self.samples    = []
        with open(csv_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    self.samples.append((parts[0], int(parts[1]), int(parts[2])))
        print(f"  Train dataset: {len(self.samples)} videos")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_id, n_frames, label = self.samples[idx]
        frame_dir = os.path.join(self.frames_dir, video_id)
        try:
            files = sorted(os.listdir(frame_dir))
        except Exception:
            files = []
        if not files:
            files = [f"{i:06d}.jpg" for i in range(self.num_frames)]
        if len(files) >= self.num_frames:
            idxs = np.linspace(0, len(files) - 1, self.num_frames, dtype=int)
            selected = [files[i] for i in idxs]
        else:
            selected = (files * (self.num_frames // len(files) + 1))[:self.num_frames]
        pil_frames = []
        for fname in selected:
            path = os.path.join(frame_dir, fname)
            try:
                img = Image.open(path).convert('RGB')
            except Exception:
                img = Image.new('RGB', (224, 224))
            pil_frames.append(img)
        inputs = self.processor(images=pil_frames, return_tensors='pt')
        return inputs['pixel_values'].squeeze(0), label


# ── MODEL LOADING ─────────────────────────────────────────────────────────────
def load_model():
    model = TimesformerForVideoClassification.from_pretrained(
        MODEL_DIR, local_files_only=True)
    model.to(DEVICE)
    return model


# ── FREEZE / UNFREEZE ─────────────────────────────────────────────────────────
def freeze_all(model):
    for p in model.parameters():
        p.requires_grad_(False)


def unfreeze_temporal(model):
    """Unfreeze only temporal_attention, temporal_dense, temporal_layernorm."""
    n = 0
    for block in model.timesformer.encoder.layer:
        for p in block.temporal_attention.parameters():
            p.requires_grad_(True); n += p.numel()
        for p in block.temporal_dense.parameters():
            p.requires_grad_(True); n += p.numel()
        for p in block.temporal_layernorm.parameters():
            p.requires_grad_(True); n += p.numel()
    print(f"  Unfrozen temporal parameters: {n:,}")
    return n


# ── EVALUATION (paper protocol: 3 spatial crops, averaged) ────────────────────
def evaluate(model, desc="eval"):
    dataset = _SSv2Eval(
        frames_dir=FRAMES_DIR, test_csv=TEST_CSV,
        num_frames=NUM_FRAMES, num_spatial_crops=3)
    loader = DataLoader(
        dataset, batch_size=BATCH_EVAL, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)

    model.eval()
    all_probs  = {}
    all_labels = {}
    t0 = time.time()

    with torch.no_grad():
        for i, (videos, labels, sidxs) in enumerate(loader):
            videos = videos.to(DEVICE)
            probs  = F.softmax(model(pixel_values=videos).logits, dim=-1)
            for prob, label, sidx in zip(probs, labels, sidxs):
                s = sidx.item()
                all_probs.setdefault(s, []).append(prob.cpu())
                all_labels[s] = label.item()
            if i % 500 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = (len(loader) - i - 1) / rate if rate > 0 else 0
                print(f"  [{desc}] {i}/{len(loader)} | {rate:.1f} b/s | ETA {remaining/60:.1f}m")

    correct_top1 = correct_top5 = 0
    total = len(all_probs)
    per_class_c = {}; per_class_t = {}

    for sidx, probs_list in all_probs.items():
        avg  = torch.stack(probs_list).mean(0)
        lbl  = all_labels[sidx]
        per_class_t[lbl] = per_class_t.get(lbl, 0) + 1
        if avg.argmax().item() == lbl:
            correct_top1 += 1
            per_class_c[lbl] = per_class_c.get(lbl, 0) + 1
        if lbl in avg.topk(5).indices.tolist():
            correct_top5 += 1

    top1 = correct_top1 / total
    top5 = correct_top5 / total
    per_class = {c: per_class_c.get(c, 0) / per_class_t[c] for c in per_class_t}
    print(f"  [{desc}] Top-1={top1*100:.2f}%  Top-5={top5*100:.2f}%  ({time.time()-t0:.0f}s)")
    return {'top1': top1, 'top5': top5, 'per_class_acc': per_class, 'n': total}


# ── TRAIN DATALOADER ──────────────────────────────────────────────────────────
def make_train_loader():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    processor = AutoImageProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
    full_ds   = SSv2TrainDataset(TRAIN_CSV, FRAMES_DIR, processor, NUM_FRAMES)
    n_subset  = int(len(full_ds) * TRAIN_SUBSET)
    indices   = random.sample(range(len(full_ds)), n_subset)
    subset    = Subset(full_ds, indices)
    print(f"  Training subset: {n_subset}/{len(full_ds)} videos ({TRAIN_SUBSET*100:.0f}%)")
    return DataLoader(subset, batch_size=BATCH_TRAIN, shuffle=True,
                      num_workers=NUM_WORKERS, pin_memory=True)


# ── SAVE RESULT ───────────────────────────────────────────────────────────────
def save_result(name, result, extra=None):
    record = {'experiment': name, **result}
    if extra:
        record.update(extra)
    record['per_class_acc'] = {str(k): v for k, v in record.get('per_class_acc', {}).items()}
    path = os.path.join(RESULTS_DIR, f'{name}.json')
    with open(path, 'w') as f:
        json.dump(record, f, indent=2)
    print(f"  Saved → {path}")
    return path
