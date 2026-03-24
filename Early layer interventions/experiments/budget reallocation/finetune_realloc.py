#!/usr/bin/env python3
"""
Attention Budget Reallocation Experiment
=========================================
Hypothesis:
  Early layers waste an attention head on temporal patterns that aren't
  useful yet. Reallocating that head's capacity to a semantically-focused
  head (highest attention entropy) gives the model a stronger starter in
  early layers.

Method (layers 0-3):
  1. Calibration pass on 50 videos to collect attention weights
     (computed directly from Q/K inside hooks — no output_attentions flag needed)
  2. In spatial attention: find head with HIGHEST entropy (semantic head)
  3. In temporal attention: find head with LOWEST entropy (most temporally
     focused, least useful in early layers)
  4. Copy semantic head Q/K/V weights into temporal head slot
     → temporal head now acts as a clone boosting semantic processing
  5. Fine-tune concat classifier head (same as experiment 1)

Comparisons:
  (a) Baseline          timesformer-finetuned-ssv2
  (b) Experiment 1      concat last 4 layers (finetune.py)
  (c) This experiment   reallocation + concat
  (d) Control           random head duplication (--random_control flag)

Usage:
    python finetune_realloc.py \
        --model_dir  /home/s2197197/timesformer/timesformer-model \
        --frames_dir /disk/scratch/s2197197/evaluation_frames/frames \
        --train_csv  /disk/scratch/s2197197/evaluation_frames/frame_lists/train.csv \
        --output_dir /home/s2197197/timesformer/checkpoints_realloc \
        --batch_size 8 \
        --epochs 5 \
        --num_workers 0

    # Control: random head duplication instead of semantic
    python finetune_realloc.py ... --random_control
"""

import os
import json
import random
import argparse
import numpy as np
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import TimesformerForVideoClassification

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ─────────────────────────────────────────────
# Dataset helpers (defined early — used in calibration)
# ─────────────────────────────────────────────

def _load_frames(frames_dir, video_id, num_video_frames, num_frames=8):
    frame_dir = os.path.join(frames_dir, video_id)
    if num_video_frames >= num_frames:
        indices = np.linspace(0, num_video_frames - 1, num_frames, dtype=int)
    else:
        indices = list(range(num_video_frames))
        while len(indices) < num_frames:
            indices.append(num_video_frames - 1)

    frames = []
    for idx in indices[:num_frames]:
        for fmt in [f"{idx+1:05d}.jpg", f"{idx+1:04d}.jpg", f"{idx+1}.jpg"]:
            fp = os.path.join(frame_dir, fmt)
            if os.path.exists(fp):
                frames.append(Image.open(fp).convert('RGB'))
                break
        else:
            frames.append(frames[-1].copy() if frames else Image.new('RGB', (224, 224)))
    return frames


def _process_frame(img, normalize, crop_size=224):
    w, h = img.size
    if w < h:
        img = img.resize((256, int(h * 256 / w)), Image.BILINEAR)
    else:
        img = img.resize((int(w * 256 / h), 256), Image.BILINEAR)
    w, h = img.size
    x1 = (w - crop_size) // 2
    y1 = (h - crop_size) // 2
    img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
    t = torch.from_numpy(np.array(img)).float() / 255.0
    t = t.permute(2, 0, 1)
    return normalize(t)


# ─────────────────────────────────────────────
# Step 1: Calibration — collect attention weights
# ─────────────────────────────────────────────

def collect_attention_weights(model, frames_dir, train_csv, device,
                               num_videos=50, num_frames=8, target_layers=[0,1,2,3]):
    """
    Compute attention weights directly from Q/K inside forward hooks.
    Does NOT rely on output_attentions=True (which HuggingFace TimeSformer
    doesn't propagate to individual SelfAttention modules).
    """
    print(f"\nCalibration: collecting attention weights from {num_videos} videos...", flush=True)

    samples = []
    with open(train_csv, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                samples.append((parts[0], int(parts[1]), int(parts[2])))
    random.seed(42)
    samples = random.sample(samples, min(num_videos, len(samples)))

    spatial_attn  = {l: [] for l in target_layers}
    temporal_attn = {l: [] for l in target_layers}
    hooks = []
    captured = {}

    def make_hook(layer_idx, attn_type):
        def hook(module, input, output):
            # Compute attention probs directly from Q and K
            # input[0] is hidden_states: (batch, seq, hidden)
            hidden = input[0]
            B, N, _ = hidden.shape
            hidden_size = module.qkv.weight.shape[1]
            num_heads = 12
            head_dim = hidden_size // num_heads

            qkv_out = module.qkv(hidden)  # (B, N, 3*hidden)
            q = qkv_out[:, :, :hidden_size]
            k = qkv_out[:, :, hidden_size:2*hidden_size]

            # Reshape to (B, heads, seq, head_dim)
            q = q.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
            k = k.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)

            attn = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
            attn = torch.softmax(attn, dim=-1)  # (B, heads, seq, seq)

            # Average over batch → (heads, seq, seq)
            captured[(layer_idx, attn_type)] = attn.detach().cpu().mean(0)
        return hook

    for l in target_layers:
        layer = model.timesformer.encoder.layer[l]
        h1 = layer.attention.attention.register_forward_hook(make_hook(l, 'spatial'))
        h2 = layer.temporal_attention.attention.register_forward_hook(make_hook(l, 'temporal'))
        hooks.extend([h1, h2])

    model.eval()
    normalize = lambda t: (t - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) \
                           / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    with torch.no_grad():
        for video_id, num_video_frames, _ in samples:
            try:
                frames = _load_frames(frames_dir, video_id, num_video_frames, num_frames)
                tensors = torch.stack([_process_frame(f, normalize) for f in frames])
                pixel_values = tensors.unsqueeze(0).to(device)

                model(pixel_values=pixel_values)

                for l in target_layers:
                    if (l, 'spatial') in captured:
                        spatial_attn[l].append(captured[(l, 'spatial')])
                    if (l, 'temporal') in captured:
                        temporal_attn[l].append(captured[(l, 'temporal')])
                captured.clear()
            except Exception as e:
                print(f"  Skipped video {video_id}: {e}", flush=True)
                continue

    for h in hooks:
        h.remove()

    print(f"  Captured spatial layers:  {[l for l in target_layers if spatial_attn[l]]}", flush=True)
    print(f"  Captured temporal layers: {[l for l in target_layers if temporal_attn[l]]}", flush=True)

    spatial_avg  = {l: torch.stack(spatial_attn[l]).mean(0)  for l in target_layers if spatial_attn[l]}
    temporal_avg = {l: torch.stack(temporal_attn[l]).mean(0) for l in target_layers if temporal_attn[l]}

    return spatial_avg, temporal_avg


def compute_entropy(attn_weights):
    """
    attn_weights: (num_heads, seq, seq)
    Returns: (num_heads,) entropy per head
    """
    attn = attn_weights.clamp(min=1e-9)
    entropy = -(attn * attn.log()).sum(dim=-1).mean(dim=-1)
    return entropy


# ─────────────────────────────────────────────
# Step 2: Identify heads
# ─────────────────────────────────────────────

def identify_heads(spatial_avg, temporal_avg, target_layers, random_control=False):
    head_map = {}
    print("\nHead identification:", flush=True)

    for l in target_layers:
        if l not in spatial_avg or l not in temporal_avg:
            print(f"  Layer {l}: skipped (no calibration data)", flush=True)
            continue

        spatial_entropy  = compute_entropy(spatial_avg[l])
        temporal_entropy = compute_entropy(temporal_avg[l])

        if random_control:
            num_heads = spatial_entropy.shape[0]
            semantic_head = random.randint(0, num_heads - 1)
            temporal_head = random.randint(0, num_heads - 1)
            print(f"  Layer {l}: RANDOM semantic_head={semantic_head}  temporal_head={temporal_head}", flush=True)
        else:
            semantic_head = spatial_entropy.argmax().item()   # highest entropy
            temporal_head = temporal_entropy.argmin().item()  # lowest entropy
            print(f"  Layer {l}: semantic_head={semantic_head} "
                  f"(entropy={spatial_entropy[semantic_head]:.4f})  "
                  f"temporal_head={temporal_head} "
                  f"(entropy={temporal_entropy[temporal_head]:.4f})", flush=True)

        head_map[l] = (semantic_head, temporal_head)

    return head_map


# ─────────────────────────────────────────────
# Step 3: Copy Q/K/V weights
# ─────────────────────────────────────────────

def reallocate_heads(model, head_map):
    print("\nReallocating heads:", flush=True)

    for l, (semantic_head, temporal_head) in head_map.items():
        spatial_qkv  = model.timesformer.encoder.layer[l].attention.attention.qkv
        temporal_qkv = model.timesformer.encoder.layer[l].temporal_attention.attention.qkv

        hidden_size = spatial_qkv.weight.shape[1]
        num_heads   = 12
        head_dim    = hidden_size // num_heads

        with torch.no_grad():
            sq = slice(semantic_head * head_dim, (semantic_head + 1) * head_dim)
            sk = slice(hidden_size + semantic_head * head_dim,
                       hidden_size + (semantic_head + 1) * head_dim)
            sv = slice(2 * hidden_size + semantic_head * head_dim,
                       2 * hidden_size + (semantic_head + 1) * head_dim)

            tq = slice(temporal_head * head_dim, (temporal_head + 1) * head_dim)
            tk = slice(hidden_size + temporal_head * head_dim,
                       hidden_size + (temporal_head + 1) * head_dim)
            tv = slice(2 * hidden_size + temporal_head * head_dim,
                       2 * hidden_size + (temporal_head + 1) * head_dim)

            temporal_qkv.weight[tq].copy_(spatial_qkv.weight[sq])
            temporal_qkv.weight[tk].copy_(spatial_qkv.weight[sk])
            temporal_qkv.weight[tv].copy_(spatial_qkv.weight[sv])

            if temporal_qkv.bias is not None and spatial_qkv.bias is not None:
                temporal_qkv.bias[tq].copy_(spatial_qkv.bias[sq])
                temporal_qkv.bias[tk].copy_(spatial_qkv.bias[sk])
                temporal_qkv.bias[tv].copy_(spatial_qkv.bias[sv])

        print(f"  Layer {l}: copied spatial head {semantic_head} → temporal head {temporal_head}", flush=True)

    print("Reallocation complete.", flush=True)


# ─────────────────────────────────────────────
# Concat Head
# ─────────────────────────────────────────────

class ConcatHead(nn.Module):
    def __init__(self, hidden_size=768, num_layers=4, num_classes=174):
        super().__init__()
        self.classifier = nn.Linear(hidden_size * num_layers, num_classes)

    def forward(self, hidden_states):
        cls_tokens = [h[:, 0, :] for h in hidden_states]
        combined   = torch.cat(cls_tokens, dim=-1)
        return self.classifier(combined)


# ─────────────────────────────────────────────
# Full Model Wrapper
# ─────────────────────────────────────────────

class TimeSformerRealloc(nn.Module):
    def __init__(self, base_model, num_classes=174):
        super().__init__()
        self.backbone      = base_model
        self.target_layers = [8, 9, 10, 11]
        self.head          = ConcatHead(hidden_size=768, num_layers=4, num_classes=num_classes)
        self._freeze_layers()

    def _freeze_layers(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = True

        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nParameter summary:")
        print(f"  Total:     {total:,}")
        print(f"  Trainable: {trainable:,}  ({100*trainable/total:.1f}%) — concat head only")
        print(f"  Frozen:    {total-trainable:,}  ({100*(total-trainable)/total:.1f}%)", flush=True)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values, output_hidden_states=True)
        target_hidden = [outputs.hidden_states[i + 1] for i in self.target_layers]
        return self.head(target_hidden)


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class SSv2TrainDataset(Dataset):
    def __init__(self, frames_dir, train_csv, num_frames=8, crop_size=224):
        self.frames_dir = frames_dir
        self.num_frames = num_frames
        self.crop_size  = crop_size
        self.samples    = []

        with open(train_csv, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    self.samples.append((parts[0], int(parts[1]), int(parts[2])))

        print(f"Loaded {len(self.samples)} training samples", flush=True)
        self.normalize = lambda t: (
            t - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        ) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_id, num_video_frames, label = self.samples[idx]
        frames  = _load_frames(self.frames_dir, video_id, num_video_frames, self.num_frames)
        tensors = torch.stack([_process_frame(f, self.normalize, self.crop_size) for f in frames])
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

        logits = model(pixel_values=videos)
        loss   = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds    = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        if (step + 1) % log_every == 0:
            print(f"  Epoch {epoch} Step {step+1}/{num_batches}: "
                  f"loss={total_loss/(step+1):.4f}  "
                  f"top-1={correct/total*100:.2f}%", flush=True)

    scheduler.step()
    return total_loss / len(dataloader), correct / total * 100


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Attention budget reallocation')
    parser.add_argument('--model_dir',      type=str, required=True)
    parser.add_argument('--frames_dir',     type=str, required=True)
    parser.add_argument('--train_csv',      type=str, required=True)
    parser.add_argument('--output_dir',     type=str, required=True)
    parser.add_argument('--batch_size',     type=int,   default=8)
    parser.add_argument('--epochs',         type=int,   default=5)
    parser.add_argument('--lr',             type=float, default=1e-4)
    parser.add_argument('--num_workers',    type=int,   default=0)
    parser.add_argument('--num_frames',     type=int,   default=8)
    parser.add_argument('--calib_videos',   type=int,   default=50)
    parser.add_argument('--random_control', action='store_true')
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("TimeSformer — Attention Budget Reallocation", flush=True)
    print("MODE: RANDOM CONTROL" if args.random_control else "MODE: SEMANTIC REALLOCATION (entropy-guided)", flush=True)
    print("=" * 60, flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    print("\nLoading base model...", flush=True)
    base_model = TimesformerForVideoClassification.from_pretrained(
        args.model_dir, local_files_only=True
    )
    base_model = base_model.to(device)

    # Step 1: Calibration
    target_layers = [0, 1, 2, 3]
    spatial_avg, temporal_avg = collect_attention_weights(
        base_model, args.frames_dir, args.train_csv,
        device, num_videos=args.calib_videos,
        num_frames=args.num_frames, target_layers=target_layers
    )

    # Step 2: Identify heads
    head_map = identify_heads(spatial_avg, temporal_avg, target_layers,
                               random_control=args.random_control)

    # Step 3: Reallocate
    reallocate_heads(base_model, head_map)

    head_map_serializable = {str(k): list(v) for k, v in head_map.items()}
    with open(os.path.join(args.output_dir, 'head_map.json'), 'w') as f:
        json.dump(head_map_serializable, f, indent=2)
    print(f"\nHead map saved to {args.output_dir}/head_map.json", flush=True)

    # Wrap with concat head
    model = TimeSformerRealloc(base_model, num_classes=174).to(device)

    dataset    = SSv2TrainDataset(args.frames_dir, args.train_csv, args.num_frames)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers,
                            pin_memory=True, drop_last=True)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.lr, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Resume ─────────────────────────────────────────────────────────
    start_epoch = 1
    best_loss   = float('inf')

    if os.path.exists(args.output_dir):
        ckpts = sorted([f for f in os.listdir(args.output_dir)
                        if f.startswith('epoch_') and f.endswith('.pt')])
        if ckpts:
            latest = os.path.join(args.output_dir, ckpts[-1])
            print(f"\nResuming from {latest}...", flush=True)
            ckpt = torch.load(latest, map_location=device)
            model.load_state_dict(ckpt['state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_loss   = ckpt['loss']
            print(f"Resuming from epoch {start_epoch}, best_loss: {best_loss:.4f}", flush=True)
        else:
            print("\nStarting from scratch", flush=True)
    else:
        print("\nStarting from scratch", flush=True)

    history = []

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*60}", flush=True)
        print(f"Epoch {epoch}/{args.epochs}", flush=True)
        print(f"{'='*60}", flush=True)

        loss, acc = train(model, dataloader, optimizer, scheduler, device, epoch)
        print(f"\nEpoch {epoch} done — Loss: {loss:.4f} | Acc: {acc:.2f}%", flush=True)

        ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch}.pt")
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                    'loss': loss, 'acc': acc, 'experiment': 'realloc',
                    'random_control': args.random_control,
                    'head_map': head_map_serializable}, ckpt_path)
        print(f"  Saved {ckpt_path}", flush=True)

        if loss < best_loss:
            best_loss = loss
            best_path = os.path.join(args.output_dir, 'best.pt')
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'loss': loss, 'acc': acc, 'experiment': 'realloc',
                        'random_control': args.random_control,
                        'head_map': head_map_serializable}, best_path)
            print(f"  New best model saved (loss={best_loss:.4f})", flush=True)

        history.append({'epoch': epoch, 'loss': loss, 'acc': acc})

    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best loss: {best_loss:.4f}", flush=True)
    print(f"Checkpoints: {args.output_dir}", flush=True)


if __name__ == '__main__':
    main()
