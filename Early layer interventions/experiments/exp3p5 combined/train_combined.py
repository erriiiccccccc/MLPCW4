#!/usr/bin/env python3

import os
import sys
import time
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from shared import (
    load_model, freeze_all, unfreeze_temporal, make_train_loader,
    evaluate, save_result, SHAPLEY_SUMS, BASE_LR, BASE_WD, EPOCHS,
    DEVICE
)
from train_distillation import (
    setup_hooks, distil_loss,
    TEACHER_LAYERS, STUDENT_LAYERS, TEMP, DISTIL_W
)

WEAK_THRESHOLD   = 0.06
STRONG_THRESHOLD = 0.15


def build_shapley_param_groups(model):
    groups = []
    for idx, block in enumerate(model.timesformer.encoder.layer):
        s  = SHAPLEY_SUMS[idx]
        wd = BASE_WD * 3.0 if s < WEAK_THRESHOLD else (
             BASE_WD * 0.3 if s > STRONG_THRESHOLD else BASE_WD)
        params = (list(block.temporal_attention.parameters()) +
                  list(block.temporal_dense.parameters()) +
                  list(block.temporal_layernorm.parameters()))
        groups.append({'params': params, 'weight_decay': wd, 'lr': BASE_LR})
    return groups


def main():
    print("Exp 3+5: Shapley WD + Knowledge Distillation (combined)")

    train_loader = make_train_loader()

    model = load_model()
    freeze_all(model)
    unfreeze_temporal(model)

    param_groups = build_shapley_param_groups(model)
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    captured, capture_handles = setup_hooks(
        model, TEACHER_LAYERS | STUDENT_LAYERS)

    model.eval()
    history = []

    for epoch in range(EPOCHS):
        epoch_ce = epoch_distil = correct = total = 0
        t0 = time.time()

        for batch_idx, (pixel_values, labels) in enumerate(train_loader):
            pixel_values = pixel_values.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()

            logits = model(pixel_values=pixel_values).logits
            ce_loss = criterion(logits, labels)
            d_loss = distil_loss(
                captured, TEACHER_LAYERS, STUDENT_LAYERS, TEMP)
            loss = ce_loss + DISTIL_W * d_loss
            loss.backward()
            optimizer.step()

            epoch_ce += ce_loss.item()
            epoch_distil += d_loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

            if batch_idx % 100 == 0:
                print(f"  [Exp3+5] Epoch {epoch+1}/{EPOCHS} | "
                      f"Batch {batch_idx}/{len(train_loader)} | "
                      f"CE={ce_loss.item():.4f} | Distil={d_loss.item():.4f}")

        scheduler.step()
        avg_ce = epoch_ce / len(train_loader)
        avg_distil = epoch_distil / len(train_loader)
        acc = correct / total
        elapsed = time.time() - t0
        history.append({'epoch': epoch+1, 'ce_loss': avg_ce,
                         'distil_loss': avg_distil, 'train_acc': acc})
        print(f"  [Exp3+5] Epoch {epoch+1} | CE={avg_ce:.4f} | "
              f"Distil={avg_distil:.4f} | TrainAcc={acc:.4f} | {elapsed:.0f}s")

    for h in capture_handles:
        h.remove()

    res = evaluate(model, desc="Exp3+5")
    save_result('exp3p5_combined', res, {
        'history': history,
        'config': {
            'use_shapley_wd': True, 'base_wd': BASE_WD,
            'teacher_layers': sorted(TEACHER_LAYERS),
            'student_layers': sorted(STUDENT_LAYERS),
            'temperature': TEMPERATURE, 'distil_weight': DISTIL_WEIGHT,
            'base_lr': BASE_LR, 'epochs': EPOCHS, 'train_subset': 0.15
        }
    })

    print(f"\nExp 3+5: Top-1={res['top1']*100:.2f}%  Top-5={res['top5']*100:.2f}%", flush=True)


if __name__ == '__main__':
    main()
