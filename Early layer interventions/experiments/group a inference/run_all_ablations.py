#!/usr/bin/env python3
"""Run the Group A inference-time head intervention experiments."""

import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import spectral_norm
from PIL import ImageFile
from transformers import TimesformerForVideoClassification

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../setup'))
from step3_full_evaluation import SSv2Dataset, collate_fn

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

MODEL_DIR   = '/home/s2411221/MLPCW4/timesformer-model'
FRAMES_DIR  = '/disk/scratch/MLPG102/evaluation_frames/frames'
TEST_CSV    = '/disk/scratch/MLPG102/evaluation_frames/frame_lists/test.csv'
RESULTS_DIR = '/home/s2411221/temporal_experiments/results'
BATCH_SIZE  = 8
NUM_WORKERS = 0
NUM_FRAMES  = 8
NUM_CROPS   = 3
HEAD_SIZE   = 64
NUM_HEADS   = 12
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

HARMFUL_HEADS    = [(5, 2), (2, 1), (6, 7), (0, 4)]
NEGLIGIBLE_HEADS = [(0, 5), (1, 0), (4, 0), (5, 5),
                    (7, 0), (7, 7), (8, 1), (9, 4), (9, 0), (10, 6)]
HARMFUL_LAYERS   = {0, 2, 5, 6}

os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Device: {DEVICE}")
print(f"Results → {RESULTS_DIR}")


def load_fresh_model():
    """Load a fresh frozen model instance."""
    model = TimesformerForVideoClassification.from_pretrained(
        MODEL_DIR, local_files_only=True)
    model.eval()
    model.to(DEVICE)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def make_head_mask_hook(masked_heads):
    """Zero out specific head channels in TimesformerSelfAttention output."""
    def hook(module, input, output):
        x = output[0].clone()
        for h in masked_heads:
            x[..., h * HEAD_SIZE:(h + 1) * HEAD_SIZE] = 0.0
        return (x,) + output[1:]
    return hook


def register_mask_hooks(model, head_pairs):
    """Register one masking hook per layer and return the handles."""
    by_layer = {}
    for layer, head in head_pairs:
        by_layer.setdefault(layer, []).append(head)

    hs = []
    for layer_idx, heads in by_layer.items():
        mod = model.timesformer.encoder.layer[layer_idx].temporal_attention.attention
        h = mod.register_forward_hook(make_head_mask_hook(heads))
        hs.append(h)
    return hs


def remove_hooks(hs):
    for h in hs:
        h.remove()


def apply_spectral_norm(model, target_layers):
    """Apply spectral norm to Q, K, V projections in temporal attention."""
    for idx in target_layers:
        mod = model.timesformer.encoder.layer[idx].temporal_attention.attention
        mod.qkv = spectral_norm(mod.qkv)
    print(f"  Spectral norm applied to layers {sorted(target_layers)}")


def evaluate(model, dataloader, desc=""):
    model.eval()
    prob_buf = {}
    label_map = {}

    t0 = time.time()
    with torch.no_grad():
        for batch_idx, (videos, labels, sample_indices) in enumerate(dataloader):
            videos = videos.to(DEVICE)
            outputs = model(pixel_values=videos)
            probs   = F.softmax(outputs.logits, dim=-1)

            for prob, label, sidx in zip(probs, labels, sample_indices):
                sidx  = sidx.item()
                label = label.item()
                prob_buf.setdefault(sidx, []).append(prob.cpu())
                label_map[sidx] = label

            if batch_idx % 500 == 0:
                elapsed = time.time() - t0
                rate = (batch_idx + 1) / elapsed if elapsed > 0 else 0
                remaining = (len(dataloader) - batch_idx - 1) / rate if rate > 0 else 0
                print(f"  [{desc}] Batch {batch_idx}/{len(dataloader)} "
                      f"| {rate:.1f} batch/s | ETA {remaining/60:.1f} min")

    correct_top1 = correct_top5 = 0
    total = len(prob_buf)
    per_cls_ok = {}
    per_cls_n  = {}

    for sidx, prob_list in prob_buf.items():
        avg_prob = torch.stack(prob_list).mean(0)
        label    = label_map[sidx]

        per_cls_n[label]   = per_cls_n.get(label, 0) + 1
        pred = avg_prob.argmax().item()
        if pred == label:
            correct_top1 += 1
            per_cls_ok[label] = per_cls_ok.get(label, 0) + 1
        if label in avg_prob.topk(5).indices.tolist():
            correct_top5 += 1

    top1 = correct_top1 / total
    top5 = correct_top5 / total
    per_class = {c: per_cls_ok.get(c, 0) / per_cls_n[c]
                 for c in per_cls_n}

    print(f"  [{desc}] Top-1={top1*100:.2f}%  Top-5={top5*100:.2f}%  "
          f"({time.time()-t0:.0f}s)")
    return {'top1': top1, 'top5': top5, 'per_class_acc': per_class, 'n': total}


def save_result(name, result, extra=None):
    blob = {'experiment': name, **result}
    if extra:
        blob.update(extra)
    blob['per_class_acc'] = {str(k): v for k, v in blob.get('per_class_acc', {}).items()}
    path = os.path.join(RESULTS_DIR, f'{name}.json')
    with open(path, 'w') as f:
        json.dump(blob, f, indent=2)
    print(f"  Saved → {path}")


def load_result(name):
    path = os.path.join(RESULTS_DIR, f'{name}.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def main():
    print("\nLoading dataset...")
    dataset = SSv2Dataset(
        frames_dir=FRAMES_DIR, test_csv=TEST_CSV,
        num_frames=NUM_FRAMES, num_spatial_crops=NUM_CROPS)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=False)
    print(f"  {len(dataset.samples)} videos × {NUM_CROPS} crops = {len(dataset)} samples")

    res_all = {}

    old = load_result('baseline')
    if old:
        print("\n[Baseline] Skipping — already done")
        base_top1 = old['top1']
        res_all['baseline'] = old
    else:
        print("\n[Baseline] No hooks, unmodified model")
        model = load_fresh_model()
        res = evaluate(model, dataloader, "Baseline")
        save_result('baseline', res)
        res_all['baseline'] = res
        base_top1 = res['top1']

    old = load_result('exp1_harmful_masking')
    if old:
        print("\n[Exp 1] Skipping — already done")
        res_all['exp1'] = old
    else:
        print("\n[Exp 1] Harmful head masking: L5-H2, L2-H1, L6-H7, L0-H4")
        if 'model' not in dir():
            model = load_fresh_model()
        hs = register_mask_hooks(model, HARMFUL_HEADS)
        res = evaluate(model, dataloader, "Exp1")
        remove_hooks(hs)
        save_result('exp1_harmful_masking', res, {
            'masked_heads': HARMFUL_HEADS,
            'delta_top1': res['top1'] - base_top1})
        res_all['exp1'] = res

    old = load_result('exp2_negligible_pruning')
    if old:
        print("\n[Exp 2] Skipping — already done")
        res_all['exp2'] = old
    else:
        print(f"\n[Exp 2] Negligible head pruning: {NEGLIGIBLE_HEADS}")
        if 'model' not in dir():
            model = load_fresh_model()
        hs = register_mask_hooks(model, NEGLIGIBLE_HEADS)
        res = evaluate(model, dataloader, "Exp2")
        remove_hooks(hs)
        save_result('exp2_negligible_pruning', res, {
            'masked_heads': NEGLIGIBLE_HEADS,
            'delta_top1': res['top1'] - base_top1})
        res_all['exp2'] = res

    old = load_result('exp1p2_combined_masking')
    if old:
        print("\n[Exp 1+2] Skipping — already done")
        res_all['exp1p2'] = old
    else:
        print("\n[Exp 1+2] Combined masking: 14 heads total")
        if 'model' not in dir():
            model = load_fresh_model()
        heads_now = HARMFUL_HEADS + NEGLIGIBLE_HEADS
        hs = register_mask_hooks(model, heads_now)
        res = evaluate(model, dataloader, "Exp1+2")
        remove_hooks(hs)
        save_result('exp1p2_combined_masking', res, {
            'masked_heads': heads_now,
            'delta_top1': res['top1'] - base_top1})
        res_all['exp1p2'] = res

    old = load_result('exp4_spectral_norm')
    if old:
        print("\n[Exp 4] Skipping — already done")
        res_all['exp4'] = old
    else:
        print("\n[Exp 4] Spectral norm on QKV for harmful layers {0,2,5,6}")
        sn_model = load_fresh_model()
        apply_spectral_norm(sn_model, HARMFUL_LAYERS)
        res = evaluate(sn_model, dataloader, "Exp4")
        save_result('exp4_spectral_norm', res, {
            'spectral_norm_layers': sorted(HARMFUL_LAYERS),
            'delta_top1': res['top1'] - base_top1})
        res_all['exp4'] = res
        del sn_model

    old = load_result('exp4p1_spectralnorm_masking')
    if old:
        print("\n[Exp 4+1] Skipping — already done")
        res_all['exp4p1'] = old
    else:
        print("\n[Exp 4+1] Spectral norm + harmful head masking")
        sn_mask_model = load_fresh_model()
        apply_spectral_norm(sn_mask_model, HARMFUL_LAYERS)
        hs = register_mask_hooks(sn_mask_model, HARMFUL_HEADS)
        res = evaluate(sn_mask_model, dataloader, "Exp4+1")
        remove_hooks(hs)
        save_result('exp4p1_spectralnorm_masking', res, {
            'spectral_norm_layers': sorted(HARMFUL_LAYERS),
            'masked_heads': HARMFUL_HEADS,
            'delta_top1': res['top1'] - base_top1})
        res_all['exp4p1'] = res
        del sn_mask_model

    old = load_result('control_a_random4')
    if old:
        print("\n[Control A] Skipping — already done")
        res_all['control_a'] = old
    else:
        print("\n[Control A] Random 4 heads masked (3 seeds)")
        if 'model' not in dir():
            model = load_fresh_model()
        ctrl_a_runs = []
        head_pool = [(l, h) for l in range(12) for h in range(NUM_HEADS)]
        for seed in [0, 1, 2]:
            rng = random.Random(seed)
            picked = rng.sample(head_pool, 4)
            print(f"  Seed {seed}: {picked}")
            hs = register_mask_hooks(model, picked)
            res = evaluate(model, dataloader, f"CtrlA-s{seed}")
            remove_hooks(hs)
            ctrl_a_runs.append({'seed': seed, 'heads': picked, **res,
                                'delta_top1': res['top1'] - base_top1})
        ctrl_a_mean = np.mean([r['top1'] for r in ctrl_a_runs])
        ctrl_a_std  = np.std([r['top1']  for r in ctrl_a_runs])
        save_result('control_a_random4', {
            'top1': ctrl_a_mean, 'top5': np.mean([r['top5'] for r in ctrl_a_runs]),
            'top1_std': ctrl_a_std, 'per_class_acc': {},
            'n': ctrl_a_runs[0]['n']}, {
            'seeds': ctrl_a_runs,
            'delta_top1_mean': ctrl_a_mean - base_top1})
        res_all['control_a'] = {'top1': ctrl_a_mean, 'top1_std': ctrl_a_std}

    old = load_result('control_b_random10')
    if old:
        print("\n[Control B] Skipping — already done")
        res_all['control_b'] = old
    else:
        print("\n[Control B] Random 10 heads masked (3 seeds)")
        if 'model' not in dir():
            model = load_fresh_model()
        ctrl_b_runs = []
        head_pool = [(l, h) for l in range(12) for h in range(NUM_HEADS)]
        for seed in [0, 1, 2]:
            rng = random.Random(seed + 10)
            picked = rng.sample(head_pool, 10)
            print(f"  Seed {seed}: {picked}")
            hs = register_mask_hooks(model, picked)
            res = evaluate(model, dataloader, f"CtrlB-s{seed}")
            remove_hooks(hs)
            ctrl_b_runs.append({'seed': seed, 'heads': picked, **res,
                                'delta_top1': res['top1'] - base_top1})
        ctrl_b_mean = np.mean([r['top1'] for r in ctrl_b_runs])
        ctrl_b_std  = np.std([r['top1']  for r in ctrl_b_runs])
        save_result('control_b_random10', {
            'top1': ctrl_b_mean, 'top5': np.mean([r['top5'] for r in ctrl_b_runs]),
            'top1_std': ctrl_b_std, 'per_class_acc': {},
            'n': ctrl_b_runs[0]['n']}, {
            'seeds': ctrl_b_runs,
            'delta_top1_mean': ctrl_b_mean - base_top1})
        res_all['control_b'] = {'top1': ctrl_b_mean, 'top1_std': ctrl_b_std}

    print("\n" + "=" * 65)
    print("  GROUP A SUMMARY")
    print("=" * 65)
    print(f"  {'Experiment':<30} {'Top-1':>8}  {'Δ vs Baseline':>14}")
    print("-" * 65)
    names = [
        ('Baseline',        'baseline',   None),
        ('Exp 1 Harmful',   'exp1',       base_top1),
        ('Exp 2 Negligible','exp2',       base_top1),
        ('Exp 1+2 Combined','exp1p2',     base_top1),
        ('Exp 4 SpectralN', 'exp4',       base_top1),
        ('Exp 4+1 SN+Mask', 'exp4p1',    base_top1),
    ]
    for label, key, base in names:
        r = res_all[key]
        delta = f"{(r['top1']-base)*100:+.2f}%" if base else "—"
        print(f"  {label:<30} {r['top1']*100:>7.2f}%  {delta:>14}")
    ca = res_all['control_a']
    cb = res_all['control_b']
    print(f"  {'Ctrl A Rnd4 (mean±std)':<30} {ca['top1']*100:>7.2f}% ±{ca['top1_std']*100:.2f}%")
    print(f"  {'Ctrl B Rnd10 (mean±std)':<30} {cb['top1']*100:>7.2f}% ±{cb['top1_std']*100:.2f}%")
    print("=" * 65)

    summary_path = os.path.join(RESULTS_DIR, 'group_a_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({'baseline_top1': base_top1, 'experiments': {
            k: {'top1': v['top1']} for k, v in res_all.items()}}, f, indent=2)
    print(f"\nSummary → {summary_path}")


if __name__ == '__main__':
    main()
