#!/usr/bin/env python3
"""
Fine-tuning TimeSformer with Multi-Layer Feature Aggregation
=============================================================
Experiment B — Concatenation:
  CLS tokens from layers 8-11 concatenated into one vector
  Output dim: 3072 → linear → 174 classes

Backbone fully frozen. Only concat classifier trains.
Supports resume from latest checkpoint with --resume flag.

Usage:
    python finetune.py \
        --model_dir  /home/s2197197/timesformer/timesformer-model \
        --frames_dir /disk/scratch/s2197197/evaluation_frames/frames \
        --train_csv  /disk/scratch/s2197197/evaluation_frames/frame_lists/train.csv \
        --output_dir /home/s2197197/timesformer/checkpoints \
        --experiment concat \
        --batch_size 8 \
        --epochs 5 \
        --num_workers 0 \
        --resume
"""

import os
import json
import argparse
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import TimesformerForVideoClassification


# ─────────────────────────────────────────────
# Multi-Layer Aggregation Heads
# ─────────────────────────────────────────────

class WeightedSumHead(nn.Module):
    def __init__(self, hidden_size=768, num_classes=174):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.ones(4))
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states):
        cls_tokens = [h[:, 0, :] for h in hidden_states]
        weights = torch.softmax(self.layer_weights, dim=0)
        combined = sum(w * c for w, c in zip(weights, cls_tokens))
        return self.classifier(combined), weights.detach().cpu()


class ConcatHead(nn.Module):
    def __init__(self, hidden_size=768, num_layers=4, num_classes=174):
        super().__init__()
        self.classifier = nn.Linear(hidden_size * num_layers, num_classes)

    def forward(self, hidden_states):
        cls_tokens = [h[:, 0, :] for h in hidden_states]
        combined = torch.cat(cls_tokens, dim=-1)
        return self.classifier(combined), None


# ─────────────────────────────────────────────
# Full Model Wrapper
# ─────────────────────────────────────────────

class TimeSformerMultiLayer(nn.Module):
    def __init__(self, base_model, experiment='concat', num_classes=174):
        super().__init__()
        self.backbone = base_model
        self.experiment = experiment
        self.target_layers = [8, 9, 10, 11]

        if experiment == 'weighted_sum':
            self.head = WeightedSumHead(hidden_size=768, num_classes=num_classes)
        elif experiment == 'concat':
            self.head = ConcatHead(hidden_size=768, num_layers=4, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown experiment: {experiment}")

        self._freeze_layers()

    def _freeze_layers(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = True

        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = total - trainable
        print(f"\nParameter summary:")
        print(f"  Total:     {total:,}")
        print(f"  Trainable: {trainable:,}  ({100*trainable/total:.1f}%) — concat classifier only")
        print(f"  Frozen:    {frozen:,}  ({100*frozen/total:.1f}%) — entire backbone frozen", flush=True)

    def forward(self, pixel_values):
        outputs = self.backbone(
            pixel_values=pixel_values,
            output_hidden_states=True
        )
        target_hidden = [outputs.hidden_states[i + 1] for i in self.target_layers]
        logits, weights = self.head(target_hidden)
        return logits, weights


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class SSv2TrainDataset(Dataset):
    def __init__(self, frames_dir, train_csv, num_frames=8, crop_size=224):
        self.frames_dir = frames_dir
        self.num_frames = num_frames
        self.crop_size  = crop_size

        self.samples = []
        with open(train_csv, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    self.samples.append((parts[0], int(parts[1]), int(parts[2])))

        print(f"Loaded {len(self.samples)} training samples", flush=True)

        self.normalize = lambda t: (
            t - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        ) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def _load_frames(self, video_id, num_video_frames):
        frame_dir = os.path.join(self.frames_dir, video_id)

        if num_video_frames >= self.num_frames:
            indices = np.linspace(0, num_video_frames - 1, self.num_frames, dtype=int)
        else:
            indices = list(range(num_video_frames))
            while len(indices) < self.num_frames:
                indices.append(num_video_frames - 1)

        frames = []
        for idx in indices[:self.num_frames]:
            for fmt in [f"{idx+1:05d}.jpg", f"{idx+1:04d}.jpg", f"{idx+1}.jpg"]:
                fp = os.path.join(frame_dir, fmt)
                if os.path.exists(fp):
                    frames.append(Image.open(fp).convert('RGB'))
                    break
            else:
                frames.append(frames[-1].copy() if frames else Image.new('RGB', (224, 224)))

        return frames

    def _process_frame(self, img):
        w, h = img.size
        if w < h:
            img = img.resize((256, int(h * 256 / w)), Image.BILINEAR)
        else:
            img = img.resize((int(w * 256 / h), 256), Image.BILINEAR)

        w, h = img.size
        x1 = (w - self.crop_size) // 2
        y1 = (h - self.crop_size) // 2
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        t = torch.from_numpy(np.array(img)).float() / 255.0
        t = t.permute(2, 0, 1)
        return self.normalize(t)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_id, num_video_frames, label = self.samples[idx]
        frames = self._load_frames(video_id, num_video_frames)
        tensors = torch.stack([self._process_frame(f) for f in frames])
        return tensors, label


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train(model, dataloader, optimizer, scheduler, device, epoch, log_every=100):
    model.train()
    total_loss = 0
    correct = 0
    total   = 0
    num_batches = len(dataloader)

    print(f"Epoch {epoch} started — {num_batches} batches", flush=True)

    for step, (videos, labels) in enumerate(dataloader):
        videos = videos.to(device)
        labels = labels.to(device)

        logits, _ = model(pixel_values=videos)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        if (step + 1) % log_every == 0:
            avg_loss = total_loss / (step + 1)
            acc      = correct / total * 100
            print(f"  Epoch {epoch} Step {step+1}/{num_batches}: loss={avg_loss:.4f}  top-1={acc:.2f}%", flush=True)

    scheduler.step()
    return total_loss / len(dataloader), correct / total * 100


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Fine-tune TimeSformer with multi-layer aggregation')
    parser.add_argument('--model_dir',   type=str, required=True)
    parser.add_argument('--frames_dir',  type=str, required=True)
    parser.add_argument('--train_csv',   type=str, required=True)
    parser.add_argument('--output_dir',  type=str, required=True)
    parser.add_argument('--experiment',  type=str, default='concat',
                        choices=['weighted_sum', 'concat'])
    parser.add_argument('--batch_size',  type=int,   default=8)
    parser.add_argument('--epochs',      type=int,   default=5)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int,   default=0)
    parser.add_argument('--num_frames',  type=int,   default=8)
    parser.add_argument('--resume',      action='store_true',
                        help='Resume from latest checkpoint in output_dir')
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print(f"TimeSformer Fine-tuning — Experiment: {args.experiment}", flush=True)
    print("=" * 60, flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    # Load base model
    print("\nLoading base model...", flush=True)
    base_model = TimesformerForVideoClassification.from_pretrained(
        args.model_dir, local_files_only=True
    )

    # Wrap with multi-layer head
    model = TimeSformerMultiLayer(
        base_model,
        experiment=args.experiment,
        num_classes=174
    ).to(device)

    # ── Resume from checkpoint ──────────────────────────────────────────
    start_epoch = 1
    best_loss   = float('inf')

    if args.resume and os.path.exists(args.output_dir):
        ckpts = sorted([f for f in os.listdir(args.output_dir)
                        if f.startswith('epoch_') and f.endswith('.pt')])
        if ckpts:
            latest = os.path.join(args.output_dir, ckpts[-1])
            print(f"\nResuming from {latest}...", flush=True)
            ckpt = torch.load(latest, map_location=device)
            model.load_state_dict(ckpt['state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_loss   = ckpt['loss']
            print(f"Resuming from epoch {start_epoch}, best_loss so far: {best_loss:.4f}", flush=True)
        else:
            print("\nNo checkpoints found — starting from scratch", flush=True)
    else:
        print("\nStarting from scratch", flush=True)

    # ── Dataset + DataLoader ────────────────────────────────────────────
    dataset    = SSv2TrainDataset(args.frames_dir, args.train_csv, args.num_frames)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers,
                            pin_memory=True, drop_last=True)

    # ── Optimizer ───────────────────────────────────────────────────────
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.05
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training loop ───────────────────────────────────────────────────
    history = []

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*60}", flush=True)
        print(f"Epoch {epoch}/{args.epochs}", flush=True)
        print(f"{'='*60}", flush=True)

        loss, acc = train(model, dataloader, optimizer, scheduler, device, epoch)

        print(f"\nEpoch {epoch} complete: loss={loss:.4f}  top-1={acc:.2f}%", flush=True)

        # Save every epoch
        ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch}.pt")
        torch.save({
            'epoch':      epoch,
            'state_dict': model.state_dict(),
            'loss':       loss,
            'acc':        acc,
            'experiment': args.experiment,
        }, ckpt_path)
        print(f"  Saved {ckpt_path}", flush=True)

        # Save best
        if loss < best_loss:
            best_loss = loss
            best_path = os.path.join(args.output_dir, 'best.pt')
            torch.save({
                'epoch':      epoch,
                'state_dict': model.state_dict(),
                'loss':       loss,
                'acc':        acc,
                'experiment': args.experiment,
            }, best_path)
            print(f"  New best checkpoint saved (loss={best_loss:.4f})", flush=True)

        history.append({'epoch': epoch, 'loss': loss, 'acc': acc})

    # Save history
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best loss: {best_loss:.4f}", flush=True)
    print(f"Checkpoints saved to: {args.output_dir}", flush=True)


if __name__ == '__main__':
    main()
