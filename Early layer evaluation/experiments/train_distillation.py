#!/usr/bin/env python3
"""
Exp 5: Knowledge distillation — teacher layers {L10, L11} → student layers {L1, L2, L5, L7}.
Exp 5-ctrl: Same setup, CE-only (no distillation) — control.

Loss = CrossEntropyLoss + λ * temporal_distillation_loss
λ = 0.1, temperature = 2.0
"""

import sys, os, json, time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from shared import (
    load_model, freeze_all, unfreeze_temporal, make_train_loader,
    evaluate, save_result, BASE_LR, BASE_WD, EPOCHS,
    RESULTS_DIR, DEVICE
)

TEACHER_LAYERS  = {10, 11}
STUDENT_LAYERS  = {1, 2, 5, 7}
TEMPERATURE     = 2.0
DISTIL_WEIGHT   = 0.1


# ── ATTENTION CAPTURE ─────────────────────────────────────────────────────────
def register_attention_capture(model, layers):
    """
    Register hooks on temporal_attention.attention (TimesformerSelfAttention)
    to capture attention_probs (output[1]) from specified layers.
    Returns captured dict (live, updated each forward pass) and hook handles.
    """
    captured = {}
    handles  = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            # output = (context_layer,) or (context_layer, attn_probs)
            # We need attn_probs — need output_attentions to be True.
            # Workaround: call module's parent forward with output_attentions
            # Instead, recompute probs from QKV inside hook.
            # output[0] = context_layer (B, seq, hidden)
            # We store the raw context for distillation instead.
            captured[layer_idx] = output[0]   # (B, seq, hidden) — use as proxy
        return hook

    for layer_idx in layers:
        module = model.timesformer.encoder.layer[layer_idx].temporal_attention.attention
        handle = module.register_forward_hook(make_hook(layer_idx))
        handles.append(handle)

    return captured, handles


def temporal_distillation_loss(captured, teacher_layers, student_layers, temperature):
    """
    KL divergence between softmax-normalised teacher and student representations.
    Uses captured context_layer (post-attention weighted sum of V) as proxy for
    attention distribution — avoids needing output_attentions=True.
    """
    teacher_vecs = [captured[l] for l in teacher_layers if l in captured]
    student_vecs = [captured[l] for l in student_layers if l in captured]

    if not teacher_vecs or not student_vecs:
        return torch.tensor(0.0, device=DEVICE)

    # Average teacher representations (detached — no gradient back to teacher)
    teacher_avg  = torch.stack(teacher_vecs).mean(0).detach()
    teacher_soft = F.softmax(teacher_avg / temperature, dim=-1)

    loss = 0.0
    for sv in student_vecs:
        student_log = F.log_softmax(sv / temperature, dim=-1)
        loss += F.kl_div(student_log, teacher_soft, reduction='batchmean')

    return loss * (temperature ** 2) / len(student_vecs)


def train(model, train_loader, use_distillation, exp_name):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=BASE_LR, weight_decay=BASE_WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    all_layers  = TEACHER_LAYERS | STUDENT_LAYERS
    captured, capture_handles = register_attention_capture(model, all_layers)

    model.eval()
    history      = []
    distil_losses = []

    for epoch in range(EPOCHS):
        epoch_loss = epoch_distil = correct = total = 0
        t0 = time.time()

        for batch_idx, (pixel_values, labels) in enumerate(train_loader):
            pixel_values = pixel_values.to(DEVICE)
            labels       = labels.to(DEVICE)
            optimizer.zero_grad()

            logits = model(pixel_values=pixel_values).logits
            ce_loss = criterion(logits, labels)

            if use_distillation:
                d_loss = temporal_distillation_loss(
                    captured, TEACHER_LAYERS, STUDENT_LAYERS, TEMPERATURE)
                loss = ce_loss + DISTIL_WEIGHT * d_loss
                epoch_distil += d_loss.item()
            else:
                loss = ce_loss

            loss.backward()
            optimizer.step()

            epoch_loss += ce_loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)

            if batch_idx % 100 == 0:
                distil_str = (f" | DistilLoss={d_loss.item():.4f}"
                              if use_distillation else "")
                print(f"  [{exp_name}] Epoch {epoch+1}/{EPOCHS} | "
                      f"Batch {batch_idx}/{len(train_loader)} | "
                      f"CE={ce_loss.item():.4f}{distil_str}")

        scheduler.step()
        avg_ce      = epoch_loss / len(train_loader)
        avg_distil  = epoch_distil / len(train_loader) if use_distillation else 0
        acc         = correct / total
        elapsed     = time.time() - t0
        history.append({'epoch': epoch+1, 'ce_loss': avg_ce,
                         'distil_loss': avg_distil, 'train_acc': acc})
        distil_losses.append(avg_distil)
        print(f"  [{exp_name}] Epoch {epoch+1} | CE={avg_ce:.4f} | "
              f"Distil={avg_distil:.4f} | TrainAcc={acc:.4f} | {elapsed:.0f}s")

    for h in capture_handles:
        h.remove()
    return history


def main():
    train_loader = make_train_loader()

    # ── Exp 5: Distillation ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("Exp 5: Knowledge distillation (teacher L10/L11 → student L1/L2/L5/L7)")
    print("="*60)
    model5 = load_model()
    freeze_all(model5)
    unfreeze_temporal(model5)
    history5 = train(model5, train_loader, use_distillation=True, exp_name="Exp5")
    print("\n  Evaluating Exp 5...")
    res5 = evaluate(model5, desc="Exp5")
    save_result('exp5_distillation', res5, {
        'history': history5,
        'config': {'teacher_layers': sorted(TEACHER_LAYERS),
                   'student_layers': sorted(STUDENT_LAYERS),
                   'temperature': TEMPERATURE, 'distil_weight': DISTIL_WEIGHT,
                   'base_lr': BASE_LR, 'epochs': EPOCHS, 'train_subset': 0.15}
    })
    del model5

    # ── Exp 5-ctrl: CE only ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("Exp 5-ctrl: CE-only fine-tuning (no distillation)")
    print("="*60)
    model_ctrl = load_model()
    freeze_all(model_ctrl)
    unfreeze_temporal(model_ctrl)
    history_ctrl = train(model_ctrl, train_loader, use_distillation=False, exp_name="Exp5-ctrl")
    print("\n  Evaluating Exp 5-ctrl...")
    res_ctrl = evaluate(model_ctrl, desc="Exp5-ctrl")
    save_result('exp5_ctrl_ce_only', res_ctrl, {
        'history': history_ctrl,
        'config': {'use_distillation': False,
                   'base_lr': BASE_LR, 'epochs': EPOCHS, 'train_subset': 0.15}
    })

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("EXP 5 RESULTS")
    print("="*60)
    print(f"  Exp 5 (distillation): Top-1={res5['top1']*100:.2f}%  Top-5={res5['top5']*100:.2f}%")
    print(f"  Exp 5-ctrl (CE only): Top-1={res_ctrl['top1']*100:.2f}%  Top-5={res_ctrl['top5']*100:.2f}%")
    print(f"  Delta: {(res5['top1']-res_ctrl['top1'])*100:+.2f}%")


if __name__ == '__main__':
    main()
