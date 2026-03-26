import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from shared import (
    load_model, freeze_all, unfreeze_temporal, make_train_loader,
    evaluate, save_result, BASE_LR, BASE_WD, EPOCHS,
    DEVICE
)

TEACHER_LAYERS = {10, 11}
STUDENT_LAYERS = {1, 2, 5, 7}
TEMPERATURE = 2.0
DISTIL_WEIGHT = 0.1


def register_attention_capture(model, layers):
    captured = {}
    handles = []

    def make_hook(li):
        def hook(mod, inp, out):
            captured[li] = out[0]
        return hook

    for li in layers:
        mod = model.timesformer.encoder.layer[li].temporal_attention.attention
        handles.append(mod.register_forward_hook(make_hook(li)))
    return captured, handles


def temporal_distillation_loss(captured, t_layers, s_layers, temp):
    t_vecs = [captured[l] for l in t_layers if l in captured]
    s_vecs = [captured[l] for l in s_layers if l in captured]
    if not t_vecs or not s_vecs:
        return torch.tensor(0.0, device=DEVICE)

    t_avg = torch.stack(t_vecs).mean(0).detach()
    t_soft = F.softmax(t_avg / temp, dim=-1)

    loss = 0.0
    for sv in s_vecs:
        s_log = F.log_softmax(sv / temp, dim=-1)
        loss += F.kl_div(s_log, t_soft, reduction='batchmean')
    return loss * (temp ** 2) / len(s_vecs)


def train(model, loader, use_distil, name):
    optim = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=BASE_LR, weight_decay=BASE_WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)
    ce_fn = nn.CrossEntropyLoss()

    captured, hooks = register_attention_capture(model, TEACHER_LAYERS | STUDENT_LAYERS)

    model.eval()
    history = []

    for ep in range(EPOCHS):
        ep_loss = ep_distil = correct = total = 0
        t0 = time.time()

        for bi, (pv, labels) in enumerate(loader):
            pv = pv.to(DEVICE)
            labels = labels.to(DEVICE)
            optim.zero_grad()

            logits = model(pixel_values=pv).logits
            ce = ce_fn(logits, labels)

            if use_distil:
                dl = temporal_distillation_loss(captured, TEACHER_LAYERS, STUDENT_LAYERS, TEMPERATURE)
                loss = ce + DISTIL_WEIGHT * dl
                ep_distil += dl.item()
            else:
                loss = ce

            loss.backward()
            optim.step()

            ep_loss += ce.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

            if bi % 100 == 0:
                ds = f" distil={dl.item():.4f}" if use_distil else ""
                print(f"  [{name}] ep {ep+1}/{EPOCHS} batch {bi}/{len(loader)} ce={ce.item():.4f}{ds}")

        sched.step()
        avg_ce = ep_loss / len(loader)
        avg_dl = ep_distil / len(loader) if use_distil else 0
        acc = correct / total
        dt = time.time() - t0
        history.append({'epoch': ep+1, 'ce_loss': avg_ce, 'distil_loss': avg_dl, 'train_acc': acc})
        print(f"  [{name}] ep {ep+1} ce={avg_ce:.4f} distil={avg_dl:.4f} acc={acc:.4f} {dt:.0f}s")

    for h in hooks:
        h.remove()
    return history


def main():
    loader = make_train_loader()

    print("\nExp 5: distillation (teacher L10/L11 -> student L1/L2/L5/L7)")
    m5 = load_model()
    freeze_all(m5)
    unfreeze_temporal(m5)
    hist5 = train(m5, loader, use_distil=True, name="Exp5")
    res5 = evaluate(m5, desc="Exp5")
    save_result('exp5_distillation', res5, {
        'history': hist5,
        'config': {'teacher_layers': sorted(TEACHER_LAYERS),
                   'student_layers': sorted(STUDENT_LAYERS),
                   'temperature': TEMPERATURE, 'distil_weight': DISTIL_WEIGHT,
                   'base_lr': BASE_LR, 'epochs': EPOCHS, 'train_subset': 0.15}
    })
    del m5

    print("\nExp 5-ctrl: CE-only (no distillation)")
    mc = load_model()
    freeze_all(mc)
    unfreeze_temporal(mc)
    hist_c = train(mc, loader, use_distil=False, name="Exp5-ctrl")
    res_c = evaluate(mc, desc="Exp5-ctrl")
    save_result('exp5_ctrl_ce_only', res_c, {
        'history': hist_c,
        'config': {'use_distillation': False,
                   'base_lr': BASE_LR, 'epochs': EPOCHS, 'train_subset': 0.15}
    })

    print(f"\nExp5 distil: top1={res5['top1']*100:.2f}% top5={res5['top5']*100:.2f}%")
    print(f"Exp5 ctrl:   top1={res_c['top1']*100:.2f}% top5={res_c['top5']*100:.2f}%")
    print(f"Delta: {(res5['top1']-res_c['top1'])*100:+.2f}%", flush=True)


if __name__ == '__main__':
    main()
