#!/usr/bin/env python3
"""
Group A: Inference-only head manipulation experiments.

Runs sequentially, sharing one dataset load:
  - Baseline              : unmodified model
  - Exp 1                 : harmful head masking  {(5,2),(2,1),(6,7),(0,4)}
  - Exp 2                 : negligible head pruning (10 heads)
  - Exp 1+2               : combined (14 heads)
  - Exp 4                 : spectral norm on QKV of harmful layers {0,2,5,6}
  - Exp 4+1               : spectral norm + harmful masking
  - Control A             : random 4 heads masked (3 seeds)
  - Control B             : random 10 heads masked (3 seeds)

Results saved to ~/temporal_experiments/results/<exp>.json
"""

import os, sys, json, time, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import spectral_norm
from PIL import Image, ImageFile
from transformers import TimesformerForVideoClassification

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../setup'))
from step3_full_evaluation import SSv2Dataset, collate_fn

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_DIR   = '/home/s2411221/MLPCW4/timesformer-model'
FRAMES_DIR  = '/disk/scratch/MLPG102/evaluation_frames/frames'
TEST_CSV    = '/disk/scratch/MLPG102/evaluation_frames/frame_lists/test.csv'
RESULTS_DIR = '/home/s2411221/temporal_experiments/results'
BATCH_SIZE  = 8
NUM_WORKERS = 0
NUM_FRAMES  = 8
NUM_CROPS   = 3         # paper protocol: 3 spatial crops
HEAD_SIZE   = 64        # hidden_size(768) // num_heads(12)
NUM_HEADS   = 12
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Shapley-identified head sets
HARMFUL_HEADS    = [(5, 2), (2, 1), (6, 7), (0, 4)]
NEGLIGIBLE_HEADS = [(0, 5), (1, 0), (4, 0), (5, 5),
                    (7, 0), (7, 7), (8, 1), (9, 4), (9, 0), (10, 6)]
HARMFUL_LAYERS   = {0, 2, 5, 6}   # layers containing harmful heads (for spectral norm)

os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Device: {DEVICE}")
print(f"Results → {RESULTS_DIR}")


# ── MODEL LOADING ─────────────────────────────────────────────────────────────
def load_fresh_model():
    """Load model fresh (needed before spectral_norm to avoid double-wrapping)."""
    model = TimesformerForVideoClassification.from_pretrained(
        MODEL_DIR, local_files_only=True)
    model.eval()
    model.to(DEVICE)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ── HOOKS ─────────────────────────────────────────────────────────────────────
def make_head_mask_hook(masked_heads):
    """Zero out specific head channels in TimesformerSelfAttention output."""
    def hook(module, input, output):
        ctx = output[0].clone()
        for h in masked_heads:
            ctx[..., h * HEAD_SIZE:(h + 1) * HEAD_SIZE] = 0.0
        return (ctx,) + output[1:]
    return hook


def register_mask_hooks(model, head_pairs):
    """
    Register masking hooks for (layer, head) pairs.
    Groups heads by layer to register one hook per layer.
    Returns list of hook handles.
    """
    layer_to_heads = {}
    for (layer, head) in head_pairs:
        layer_to_heads.setdefault(layer, []).append(head)

    handles = []
    for layer_idx, heads in layer_to_heads.items():
        module = model.timesformer.encoder.layer[layer_idx].temporal_attention.attention
        handle = module.register_forward_hook(make_head_mask_hook(heads))
        handles.append(handle)
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


# ── SPECTRAL NORM ─────────────────────────────────────────────────────────────
def apply_spectral_norm(model, target_layers):
    """Apply spectral norm to Q, K, V projections in temporal attention."""
    for idx in target_layers:
        attn = model.timesformer.encoder.layer[idx].temporal_attention.attention
        attn.qkv = spectral_norm(attn.qkv)
    print(f"  Spectral norm applied to layers {sorted(target_layers)}")


# ── EVALUATION ────────────────────────────────────────────────────────────────
def evaluate(model, dataloader, desc=""):
    model.eval()
    all_probs  = {}   # sample_idx -> list[prob tensor]
    all_labels = {}

    t0 = time.time()
    with torch.no_grad():
        for batch_idx, (videos, labels, sample_indices) in enumerate(dataloader):
            videos = videos.to(DEVICE)
            outputs = model(pixel_values=videos)
            probs   = F.softmax(outputs.logits, dim=-1)

            for prob, label, sidx in zip(probs, labels, sample_indices):
                sidx  = sidx.item()
                label = label.item()
                all_probs.setdefault(sidx, []).append(prob.cpu())
                all_labels[sidx] = label

            if batch_idx % 500 == 0:
                elapsed = time.time() - t0
                rate = (batch_idx + 1) / elapsed if elapsed > 0 else 0
                remaining = (len(dataloader) - batch_idx - 1) / rate if rate > 0 else 0
                print(f"  [{desc}] Batch {batch_idx}/{len(dataloader)} "
                      f"| {rate:.1f} batch/s | ETA {remaining/60:.1f} min")

    correct_top1 = correct_top5 = 0
    total = len(all_probs)
    per_class_correct = {}
    per_class_total   = {}

    for sidx, probs_list in all_probs.items():
        avg_prob = torch.stack(probs_list).mean(0)
        label    = all_labels[sidx]

        per_class_total[label]   = per_class_total.get(label, 0) + 1
        pred = avg_prob.argmax().item()
        if pred == label:
            correct_top1 += 1
            per_class_correct[label] = per_class_correct.get(label, 0) + 1
        if label in avg_prob.topk(5).indices.tolist():
            correct_top5 += 1

    top1 = correct_top1 / total
    top5 = correct_top5 / total
    per_class = {c: per_class_correct.get(c, 0) / per_class_total[c]
                 for c in per_class_total}

    print(f"  [{desc}] Top-1={top1*100:.2f}%  Top-5={top5*100:.2f}%  "
          f"({time.time()-t0:.0f}s)")
    return {'top1': top1, 'top5': top5, 'per_class_acc': per_class, 'n': total}


def save_result(name, result, extra=None):
    record = {'experiment': name, **result}
    if extra:
        record.update(extra)
    record['per_class_acc'] = {str(k): v for k, v in record.get('per_class_acc', {}).items()}
    path = os.path.join(RESULTS_DIR, f'{name}.json')
    with open(path, 'w') as f:
        json.dump(record, f, indent=2)
    print(f"  Saved → {path}")


def load_result(name):
    path = os.path.join(RESULTS_DIR, f'{name}.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    # ── Dataset (shared across all experiments) ────────────────────────────
    print("\nLoading dataset...")
    dataset = SSv2Dataset(
        frames_dir=FRAMES_DIR, test_csv=TEST_CSV,
        num_frames=NUM_FRAMES, num_spatial_crops=NUM_CROPS)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=False)
    print(f"  {len(dataset.samples)} videos × {NUM_CROPS} crops = {len(dataset)} samples")

    all_results = {}

    # ── BASELINE ──────────────────────────────────────────────────────────
    cached = load_result('baseline')
    if cached:
        print("\n[Baseline] Skipping — already done")
        baseline_top1 = cached['top1']
        all_results['baseline'] = cached
    else:
        print("\n[Baseline] No hooks, unmodified model")
        model = load_fresh_model()
        res = evaluate(model, dataloader, "Baseline")
        save_result('baseline', res)
        all_results['baseline'] = res
        baseline_top1 = res['top1']

    # ── EXP 1: Harmful head masking ────────────────────────────────────────
    cached = load_result('exp1_harmful_masking')
    if cached:
        print("\n[Exp 1] Skipping — already done")
        all_results['exp1'] = cached
    else:
        print("\n[Exp 1] Harmful head masking: L5-H2, L2-H1, L6-H7, L0-H4")
        if 'model' not in dir():
            model = load_fresh_model()
        handles = register_mask_hooks(model, HARMFUL_HEADS)
        res = evaluate(model, dataloader, "Exp1")
        remove_hooks(handles)
        save_result('exp1_harmful_masking', res, {
            'masked_heads': HARMFUL_HEADS,
            'delta_top1': res['top1'] - baseline_top1})
        all_results['exp1'] = res

    # ── EXP 2: Negligible head pruning ─────────────────────────────────────
    cached = load_result('exp2_negligible_pruning')
    if cached:
        print("\n[Exp 2] Skipping — already done")
        all_results['exp2'] = cached
    else:
        print(f"\n[Exp 2] Negligible head pruning: {NEGLIGIBLE_HEADS}")
        if 'model' not in dir():
            model = load_fresh_model()
        handles = register_mask_hooks(model, NEGLIGIBLE_HEADS)
        res = evaluate(model, dataloader, "Exp2")
        remove_hooks(handles)
        save_result('exp2_negligible_pruning', res, {
            'masked_heads': NEGLIGIBLE_HEADS,
            'delta_top1': res['top1'] - baseline_top1})
        all_results['exp2'] = res

    # ── EXP 1+2: Combined ──────────────────────────────────────────────────
    cached = load_result('exp1p2_combined_masking')
    if cached:
        print("\n[Exp 1+2] Skipping — already done")
        all_results['exp1p2'] = cached
    else:
        print("\n[Exp 1+2] Combined masking: 14 heads total")
        if 'model' not in dir():
            model = load_fresh_model()
        all_heads = HARMFUL_HEADS + NEGLIGIBLE_HEADS
        handles = register_mask_hooks(model, all_heads)
        res = evaluate(model, dataloader, "Exp1+2")
        remove_hooks(handles)
        save_result('exp1p2_combined_masking', res, {
            'masked_heads': all_heads,
            'delta_top1': res['top1'] - baseline_top1})
        all_results['exp1p2'] = res

    # ── EXP 4: Spectral norm ──────────────────────────────────────────────
    cached = load_result('exp4_spectral_norm')
    if cached:
        print("\n[Exp 4] Skipping — already done")
        all_results['exp4'] = cached
    else:
        print("\n[Exp 4] Spectral norm on QKV for harmful layers {0,2,5,6}")
        model_sn = load_fresh_model()
        apply_spectral_norm(model_sn, HARMFUL_LAYERS)
        res = evaluate(model_sn, dataloader, "Exp4")
        save_result('exp4_spectral_norm', res, {
            'spectral_norm_layers': sorted(HARMFUL_LAYERS),
            'delta_top1': res['top1'] - baseline_top1})
        all_results['exp4'] = res
        del model_sn

    # ── EXP 4+1: Spectral norm + harmful masking ──────────────────────────
    cached = load_result('exp4p1_spectralnorm_masking')
    if cached:
        print("\n[Exp 4+1] Skipping — already done")
        all_results['exp4p1'] = cached
    else:
        print("\n[Exp 4+1] Spectral norm + harmful head masking")
        model_sn1 = load_fresh_model()
        apply_spectral_norm(model_sn1, HARMFUL_LAYERS)
        handles = register_mask_hooks(model_sn1, HARMFUL_HEADS)
        res = evaluate(model_sn1, dataloader, "Exp4+1")
        remove_hooks(handles)
        save_result('exp4p1_spectralnorm_masking', res, {
            'spectral_norm_layers': sorted(HARMFUL_LAYERS),
            'masked_heads': HARMFUL_HEADS,
            'delta_top1': res['top1'] - baseline_top1})
        all_results['exp4p1'] = res
        del model_sn1

    # ── CONTROL A: Random 4 heads masked (3 seeds) ────────────────────────
    cached = load_result('control_a_random4')
    if cached:
        print("\n[Control A] Skipping — already done")
        all_results['control_a'] = cached
    else:
        print("\n[Control A] Random 4 heads masked (3 seeds)")
        if 'model' not in dir():
            model = load_fresh_model()
        ctrl_a_results = []
        all_layer_head_pairs = [(l, h) for l in range(12) for h in range(NUM_HEADS)]
        for seed in [0, 1, 2]:
            rng = random.Random(seed)
            rand_heads = rng.sample(all_layer_head_pairs, 4)
            print(f"  Seed {seed}: {rand_heads}")
            handles = register_mask_hooks(model, rand_heads)
            res = evaluate(model, dataloader, f"CtrlA-s{seed}")
            remove_hooks(handles)
            ctrl_a_results.append({'seed': seed, 'heads': rand_heads, **res,
                                    'delta_top1': res['top1'] - baseline_top1})
        ctrl_a_mean = np.mean([r['top1'] for r in ctrl_a_results])
        ctrl_a_std  = np.std([r['top1']  for r in ctrl_a_results])
        save_result('control_a_random4', {
            'top1': ctrl_a_mean, 'top5': np.mean([r['top5'] for r in ctrl_a_results]),
            'top1_std': ctrl_a_std, 'per_class_acc': {},
            'n': ctrl_a_results[0]['n']}, {
            'seeds': ctrl_a_results,
            'delta_top1_mean': ctrl_a_mean - baseline_top1})
        all_results['control_a'] = {'top1': ctrl_a_mean, 'top1_std': ctrl_a_std}

    # ── CONTROL B: Random 10 heads masked (3 seeds) ───────────────────────
    cached = load_result('control_b_random10')
    if cached:
        print("\n[Control B] Skipping — already done")
        all_results['control_b'] = cached
    else:
        print("\n[Control B] Random 10 heads masked (3 seeds)")
        if 'model' not in dir():
            model = load_fresh_model()
        ctrl_b_results = []
        all_layer_head_pairs = [(l, h) for l in range(12) for h in range(NUM_HEADS)]
        for seed in [0, 1, 2]:
            rng = random.Random(seed + 10)
            rand_heads = rng.sample(all_layer_head_pairs, 10)
            print(f"  Seed {seed}: {rand_heads}")
            handles = register_mask_hooks(model, rand_heads)
            res = evaluate(model, dataloader, f"CtrlB-s{seed}")
            remove_hooks(handles)
            ctrl_b_results.append({'seed': seed, 'heads': rand_heads, **res,
                                    'delta_top1': res['top1'] - baseline_top1})
        ctrl_b_mean = np.mean([r['top1'] for r in ctrl_b_results])
        ctrl_b_std  = np.std([r['top1']  for r in ctrl_b_results])
        save_result('control_b_random10', {
            'top1': ctrl_b_mean, 'top5': np.mean([r['top5'] for r in ctrl_b_results]),
            'top1_std': ctrl_b_std, 'per_class_acc': {},
            'n': ctrl_b_results[0]['n']}, {
            'seeds': ctrl_b_results,
            'delta_top1_mean': ctrl_b_mean - baseline_top1})
        all_results['control_b'] = {'top1': ctrl_b_mean, 'top1_std': ctrl_b_std}

    # ── SUMMARY ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  GROUP A SUMMARY")
    print("=" * 65)
    print(f"  {'Experiment':<30} {'Top-1':>8}  {'Δ vs Baseline':>14}")
    print("-" * 65)
    names = [
        ('Baseline',        'baseline',   None),
        ('Exp 1 Harmful',   'exp1',       baseline_top1),
        ('Exp 2 Negligible','exp2',       baseline_top1),
        ('Exp 1+2 Combined','exp1p2',     baseline_top1),
        ('Exp 4 SpectralN', 'exp4',       baseline_top1),
        ('Exp 4+1 SN+Mask', 'exp4p1',    baseline_top1),
    ]
    for label, key, base in names:
        r = all_results[key]
        delta = f"{(r['top1']-base)*100:+.2f}%" if base else "—"
        print(f"  {label:<30} {r['top1']*100:>7.2f}%  {delta:>14}")
    ca = all_results['control_a']
    cb = all_results['control_b']
    print(f"  {'Ctrl A Rnd4 (mean±std)':<30} {ca['top1']*100:>7.2f}% ±{ca['top1_std']*100:.2f}%")
    print(f"  {'Ctrl B Rnd10 (mean±std)':<30} {cb['top1']*100:>7.2f}% ±{cb['top1_std']*100:.2f}%")
    print("=" * 65)

    summary_path = os.path.join(RESULTS_DIR, 'group_a_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({'baseline_top1': baseline_top1, 'experiments': {
            k: {'top1': v['top1']} for k, v in all_results.items()}}, f, indent=2)
    print(f"\nSummary → {summary_path}")


if __name__ == '__main__':
    main()
