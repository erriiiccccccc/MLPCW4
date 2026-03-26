"""Reallocate early-layer temporal heads and fine-tune a concat head."""

import argparse
import json
import os
import random
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


def load_frames(frames_dir, vid_id, n_total, n_frames=8):
    fdir = os.path.join(frames_dir, vid_id)
    if n_total >= n_frames:
        idxs = np.linspace(0, n_total - 1, n_frames, dtype=int)
    else:
        idxs = list(range(n_total))
        while len(idxs) < n_frames:
            idxs.append(n_total - 1)

    frames = []
    for i in idxs[:n_frames]:
        for fmt in [f"{i+1:05d}.jpg", f"{i+1:04d}.jpg", f"{i+1}.jpg"]:
            fp = os.path.join(fdir, fmt)
            if os.path.exists(fp):
                frames.append(Image.open(fp).convert('RGB'))
                break
        else:
            frames.append(frames[-1].copy() if frames else Image.new('RGB', (224, 224)))
    return frames


def process_frame(img, normalize, crop=224):
    w, h = img.size
    if w < h:
        img = img.resize((256, int(h * 256 / w)), Image.BILINEAR)
    else:
        img = img.resize((int(w * 256 / h), 256), Image.BILINEAR)
    w, h = img.size
    x1 = (w - crop) // 2
    y1 = (h - crop) // 2
    img = img.crop((x1, y1, x1 + crop, y1 + crop))
    t = torch.from_numpy(np.array(img)).float() / 255.0
    t = t.permute(2, 0, 1)
    return normalize(t)


def collect_attn_weights(model, frames_dir, train_csv, dev,
                         n_videos=50, n_frames=8, target_layers=[0,1,2,3]):
    print(f"Calibration: collecting attn weights from {n_videos} videos...")

    samples = []
    with open(train_csv, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                samples.append((parts[0], int(parts[1]), int(parts[2])))
    random.seed(42)
    samples = random.sample(samples, min(n_videos, len(samples)))

    sp_attn = {l: [] for l in target_layers}
    tp_attn = {l: [] for l in target_layers}
    hooks = []
    captured = {}

    def make_hook(li, atype):
        def hook(mod, inp, out):
            hidden = inp[0]
            B, N, _ = hidden.shape
            hsz = mod.qkv.weight.shape[1]
            nheads = 12
            hdim = hsz // nheads

            qkv = mod.qkv(hidden)
            q = qkv[:, :, :hsz]
            k = qkv[:, :, hsz:2*hsz]
            q = q.reshape(B, N, nheads, hdim).permute(0, 2, 1, 3)
            k = k.reshape(B, N, nheads, hdim).permute(0, 2, 1, 3)

            a = torch.matmul(q, k.transpose(-2, -1)) / (hdim ** 0.5)
            a = torch.softmax(a, dim=-1)
            captured[(li, atype)] = a.detach().cpu().mean(0)
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
        for vid_id, nf, _ in samples:
            try:
                frames = load_frames(frames_dir, vid_id, nf, n_frames)
                tensors = torch.stack([process_frame(f, normalize) for f in frames])
                pv = tensors.unsqueeze(0).to(dev)
                model(pixel_values=pv)
                for l in target_layers:
                    if (l, 'spatial') in captured:
                        sp_attn[l].append(captured[(l, 'spatial')])
                    if (l, 'temporal') in captured:
                        tp_attn[l].append(captured[(l, 'temporal')])
                captured.clear()
            except Exception as e:
                print(f"  Skipped {vid_id}: {e}")
                continue

    for h in hooks:
        h.remove()

    sp_avg = {l: torch.stack(sp_attn[l]).mean(0) for l in target_layers if sp_attn[l]}
    tp_avg = {l: torch.stack(tp_attn[l]).mean(0) for l in target_layers if tp_attn[l]}
    return sp_avg, tp_avg


def compute_entropy(weights):
    w = weights.clamp(min=1e-9)
    return -(w * w.log()).sum(dim=-1).mean(dim=-1)


def identify_heads(sp_avg, tp_avg, target_layers, random_control=False):
    hmap = {}
    for l in target_layers:
        if l not in sp_avg or l not in tp_avg:
            print(f"  Layer {l}: skipped (no data)")
            continue

        sp_ent = compute_entropy(sp_avg[l])
        tp_ent = compute_entropy(tp_avg[l])

        if random_control:
            nh = sp_ent.shape[0]
            sem_h = random.randint(0, nh - 1)
            tmp_h = random.randint(0, nh - 1)
            print(f"  Layer {l}: RANDOM sem={sem_h} temp={tmp_h}")
        else:
            sem_h = sp_ent.argmax().item()
            tmp_h = tp_ent.argmin().item()
            print(f"  Layer {l}: sem={sem_h} (ent={sp_ent[sem_h]:.4f}) "
                  f"temp={tmp_h} (ent={tp_ent[tmp_h]:.4f})")

        hmap[l] = (sem_h, tmp_h)
    return hmap


def reallocate_heads(model, hmap):
    print("Reallocating heads nowwww")
    for l, (sem_h, tmp_h) in hmap.items():
        sp_qkv = model.timesformer.encoder.layer[l].attention.attention.qkv
        tp_qkv = model.timesformer.encoder.layer[l].temporal_attention.attention.qkv

        hsz = sp_qkv.weight.shape[1]
        nheads = 12
        hdim = hsz // nheads

        with torch.no_grad():
            sq = slice(sem_h * hdim, (sem_h + 1) * hdim)
            sk = slice(hsz + sem_h * hdim, hsz + (sem_h + 1) * hdim)
            sv = slice(2*hsz + sem_h * hdim, 2*hsz + (sem_h + 1) * hdim)

            tq = slice(tmp_h * hdim, (tmp_h + 1) * hdim)
            tk = slice(hsz + tmp_h * hdim, hsz + (tmp_h + 1) * hdim)
            tv = slice(2*hsz + tmp_h * hdim, 2*hsz + (tmp_h + 1) * hdim)

            tp_qkv.weight[tq].copy_(sp_qkv.weight[sq])
            tp_qkv.weight[tk].copy_(sp_qkv.weight[sk])
            tp_qkv.weight[tv].copy_(sp_qkv.weight[sv])
            if tp_qkv.bias is not None and sp_qkv.bias is not None:
                tp_qkv.bias[tq].copy_(sp_qkv.bias[sq])
                tp_qkv.bias[tk].copy_(sp_qkv.bias[sk])
                tp_qkv.bias[tv].copy_(sp_qkv.bias[sv])

        print(f"Layer {l}: spatial head {sem_h} -> temporal head {tmp_h}")
    print("Done")


class ConcatHead(nn.Module):
    def __init__(self, hsz=768, nlayers=4, nclasses=174):
        super().__init__()
        self.fc = nn.Linear(hsz * nlayers, nclasses)

    def forward(self, hidden_states):
        cls_toks = [h[:, 0, :] for h in hidden_states]
        return self.fc(torch.cat(cls_toks, dim=-1))


class TimeSformerRealloc(nn.Module):
    def __init__(self, base_model, nclasses=174):
        super().__init__()
        self.backbone = base_model
        self.target_layers = [8, 9, 10, 11]
        self.head = ConcatHead(hsz=768, nlayers=4, nclasses=nclasses)
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = True
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Params: {trainable:,} trainable / {total:,} total ({100*trainable/total:.1f}%)")

    def forward(self, pixel_values):
        out = self.backbone(pixel_values=pixel_values, output_hidden_states=True)
        hs = [out.hidden_states[i + 1] for i in self.target_layers]
        return self.head(hs)


class SSv2Dataset(Dataset):
    def __init__(self, frames_dir, csv_path, n_frames=8, crop=224):
        self.frames_dir = frames_dir
        self.n_frames = n_frames
        self.crop = crop
        self.samples = []
        with open(csv_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    self.samples.append((parts[0], int(parts[1]), int(parts[2])))
        print(f"Loaded {len(self.samples)} training samples")
        self.normalize = lambda t: (t - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) \
                                    / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid_id, nf, label = self.samples[idx]
        frames = load_frames(self.frames_dir, vid_id, nf, self.n_frames)
        tensors = torch.stack([process_frame(f, self.normalize, self.crop) for f in frames])
        return tensors, label


def train_epoch(model, loader, optim, sched, dev, epoch, log_every=100):
    model.train()
    tot_loss = 0
    correct = 0
    total = 0

    for step, (vids, labels) in enumerate(loader):
        vids = vids.to(dev)
        labels = labels.to(dev)

        logits = model(pixel_values=vids)
        loss = F.cross_entropy(logits, labels)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()

        tot_loss += loss.item()
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)

        if (step + 1) % log_every == 0:
            print(f"Step {step+1}/{len(loader)}: loss={tot_loss/(step+1):.4f} acc={correct/total*100:.2f}%")

    sched.step()
    return tot_loss / len(loader), correct / total * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--frames_dir', type=str, required=True)
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument('--calib_videos', type=int, default=50)
    parser.add_argument('--random_control', action='store_true')
    args = parser.parse_args()

    mode = "RANDOM CONTROL" if args.random_control else "SEMANTIC REALLOCATION"
    print(f"Attention Budget Reallocation [{mode}]")

    os.makedirs(args.output_dir, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {dev}")

    print("Loading model...")
    base = TimesformerForVideoClassification.from_pretrained(args.model_dir, local_files_only=True)
    base = base.to(dev)

    # calibration + head id + reallocation
    tgt_layers = [0, 1, 2, 3]
    sp_avg, tp_avg = collect_attn_weights(
        base, args.frames_dir, args.train_csv, dev,
        n_videos=args.calib_videos, n_frames=args.num_frames, target_layers=tgt_layers)
    hmap = identify_heads(sp_avg, tp_avg, tgt_layers, random_control=args.random_control)
    reallocate_heads(base, hmap)

    hmap_json = {str(k): list(v) for k, v in hmap.items()}
    with open(os.path.join(args.output_dir, 'head_map.json'), 'w') as f:
        json.dump(hmap_json, f, indent=2)

    model = TimeSformerRealloc(base, nclasses=174).to(dev)
    ds = SSv2Dataset(args.frames_dir, args.train_csv, args.num_frames)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    optim = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.05)
    sched = CosineAnnealingLR(optim, T_max=args.epochs)

    # resume if checkpoint exists
    start_ep = 1
    best_loss = float('inf')
    ckpts = sorted([f for f in os.listdir(args.output_dir) if f.startswith('epoch_') and f.endswith('.pt')])
    if ckpts:
        latest = os.path.join(args.output_dir, ckpts[-1])
        ckpt = torch.load(latest, map_location=dev)
        model.load_state_dict(ckpt['state_dict'])
        start_ep = ckpt['epoch'] + 1
        best_loss = ckpt['loss']
        print(f"Resumed from epoch {start_ep}, loss={best_loss:.4f}")

    history = []
    for ep in range(start_ep, args.epochs + 1):
        print(f"\nEpoch {ep}/{args.epochs}")
        loss, acc = train_epoch(model, loader, optim, sched, dev, ep)
        print(f"  Loss: {loss:.4f} | Acc: {acc:.2f}%")

        ckpt_path = os.path.join(args.output_dir, f"epoch_{ep}.pt")
        torch.save({'epoch': ep, 'state_dict': model.state_dict(),
                    'loss': loss, 'acc': acc, 'experiment': 'realloc',
                    'random_control': args.random_control,
                    'head_map': hmap_json}, ckpt_path)

        if loss < best_loss:
            best_loss = loss
            torch.save({'epoch': ep, 'state_dict': model.state_dict(),
                        'loss': loss, 'acc': acc, 'experiment': 'realloc',
                        'random_control': args.random_control,
                        'head_map': hmap_json},
                       os.path.join(args.output_dir, 'best.pt'))
            print(f"  New best (loss={best_loss:.4f})")

        history.append({'epoch': ep, 'loss': loss, 'acc': acc})

    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nDone. Best loss: {best_loss:.4f}")


if __name__ == '__main__':
    main()
