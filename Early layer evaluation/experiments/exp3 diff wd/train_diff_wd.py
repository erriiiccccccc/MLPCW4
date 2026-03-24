#!/usr/bin/env python3
"""
Exp 3: Differential weight decay (Shapley-scaled per layer).
Exp 3-ctrl: Same setup, uniform WD=5e-4 (control).

Both run sequentially in one script.
Unfreeze: temporal_attention + temporal_dense + temporal_layernorm (all 12 layers).
Everything else frozen. 5 epochs, AdamW, lr=1e-4.
"""

import sys, os, json, time
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from shared import (
    load_model, freeze_all, unfreeze_temporal, make_train_loader,
    evaluate, save_result, SHAPLEY_SUMS, BASE_LR, BASE_WD, EPOCHS,
    RESULTS_DIR, DEVICE
)


def build_param_groups(model, use_shapley_wd):
    """
    One param group per layer for temporal params.
    If use_shapley_wd: WD scaled by Shapley tier.
    If not: uniform WD=BASE_WD for all.
    """
    param_groups = []
    for idx, block in enumerate(model.timesformer.encoder.layer):
        if use_shapley_wd:
            s = SHAPLEY_SUMS[idx]
            wd = BASE_WD * 3.0 if s < 0.06 else (BASE_WD * 0.3 if s > 0.15 else BASE_WD)
        else:
            wd = BASE_WD

        temporal_params = (
            list(block.temporal_attention.parameters()) +
            list(block.temporal_dense.parameters()) +
            list(block.temporal_layernorm.parameters())
        )
        param_groups.append({
            'params': temporal_params, 'weight_decay': wd, 'lr': BASE_LR,
            'name': f'layer_{idx}'
        })
    return param_groups


def train(model, train_loader, use_shapley_wd, exp_name):
    param_groups = build_param_groups(model, use_shapley_wd)
    optimizer    = torch.optim.AdamW(param_groups)
    scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion    = nn.CrossEntropyLoss()

    if use_shapley_wd:
        wd_per_layer = {idx: (BASE_WD*3 if SHAPLEY_SUMS[idx]<0.06
                              else BASE_WD*0.3 if SHAPLEY_SUMS[idx]>0.15
                              else BASE_WD)
                        for idx in range(12)}
        print(f"  Shapley WD tiers: {wd_per_layer}")
    else:
        print(f"  Uniform WD: {BASE_WD}")

    model.eval()   # keep BN/dropout frozen
    history = []

    for epoch in range(EPOCHS):
        epoch_loss = correct = total = 0
        t0 = time.time()

        for batch_idx, (pixel_values, labels) in enumerate(train_loader):
            pixel_values = pixel_values.to(DEVICE)
            labels       = labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(pixel_values=pixel_values).logits
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)

            if batch_idx % 100 == 0:
                print(f"  [{exp_name}] Epoch {epoch+1}/{EPOCHS} | "
                      f"Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss={loss.item():.4f}")

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        acc      = correct / total
        elapsed  = time.time() - t0
        history.append({'epoch': epoch+1, 'loss': avg_loss, 'train_acc': acc})
        print(f"  [{exp_name}] Epoch {epoch+1} done | "
              f"Loss={avg_loss:.4f} | TrainAcc={acc:.4f} | {elapsed:.0f}s")

    return history


def main():
    train_loader = make_train_loader()

    # ── Exp 3: Shapley-scaled WD ──────────────────────────────────────────
    print("\n" + "="*60)
    print("Exp 3: Differential (Shapley-scaled) weight decay")
    print("="*60)
    model = load_model()
    freeze_all(model)
    unfreeze_temporal(model)
    history3 = train(model, train_loader, use_shapley_wd=True, exp_name="Exp3")
    print("\n  Evaluating Exp 3...")
    res3 = evaluate(model, desc="Exp3")
    save_result('exp3_diff_wd', res3, {
        'history': history3,
        'config': {'use_shapley_wd': True, 'base_wd': BASE_WD,
                   'base_lr': BASE_LR, 'epochs': EPOCHS, 'train_subset': 0.15}
    })
    del model

    # ── Exp 3-ctrl: Uniform WD ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("Exp 3-ctrl: Uniform weight decay (control)")
    print("="*60)
    model_ctrl = load_model()
    freeze_all(model_ctrl)
    unfreeze_temporal(model_ctrl)
    history_ctrl = train(model_ctrl, train_loader, use_shapley_wd=False, exp_name="Exp3-ctrl")
    print("\n  Evaluating Exp 3-ctrl...")
    res_ctrl = evaluate(model_ctrl, desc="Exp3-ctrl")
    save_result('exp3_ctrl_uniform_wd', res_ctrl, {
        'history': history_ctrl,
        'config': {'use_shapley_wd': False, 'base_wd': BASE_WD,
                   'base_lr': BASE_LR, 'epochs': EPOCHS, 'train_subset': 0.15}
    })

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("EXP 3 RESULTS")
    print("="*60)
    print(f"  Exp 3 (Shapley WD): Top-1={res3['top1']*100:.2f}%  Top-5={res3['top5']*100:.2f}%")
    print(f"  Exp 3 ctrl (Unif): Top-1={res_ctrl['top1']*100:.2f}%  Top-5={res_ctrl['top5']*100:.2f}%")
    print(f"  Delta: {(res3['top1']-res_ctrl['top1'])*100:+.2f}%")


if __name__ == '__main__':
    main()
