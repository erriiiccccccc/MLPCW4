#!/usr/bin/env python3
import os
import sys
import json
import argparse
import time
import numpy as np
import pandas as pd
from math import factorial
from itertools import combinations
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def inspect_model(model_dir):
    net = load_model(model_dir)

    print("\n--- All named modules ---")
    for n, mod in net.named_modules():
        print(f"  {n:60s} | {type(mod).__name__}")

    print("\n--- Looking for transformer blocks ---")
    blks = find_transformer_blocks(net)
    if blks is not None:
        print(f"  Found {len(blks)} transformer blocks")
        print(f"  First block type: {type(blks[0]).__name__}")
    else:
        print("  WARNING: Could not auto-detect blocks.")

    return net


def load_model(model_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_files = [f for f in os.listdir(model_dir)
                  if f.endswith(('.pth', '.pt', '.pyth', '.pkl'))]

    try:
        from timesformer.models.vit import TimeSformer
        config_file = os.path.join(model_dir, 'config.yaml')
        if os.path.exists(config_file):
            import yaml
            with open(config_file) as f:
                yaml.safe_load(f)

        model = TimeSformer(
            img_size=224, num_classes=174,
            num_frames=8, attention_type='divided_space_time',
        )
        if ckpt_files:
            ckpt_path = os.path.join(model_dir, ckpt_files[0])
            print(f"Loading checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            if isinstance(ckpt, dict) and 'model_state' in ckpt:
                model.load_state_dict(ckpt['model_state'])
            elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
                model.load_state_dict(ckpt['state_dict'])
            else:
                model.load_state_dict(ckpt)
        model = model.to(device)
        model.eval()
        return model
    except ImportError:
        print("timesformer package not found, trying others")
    except Exception as e:
        print(f"timesformer package load failed: {e}")

    try:
        if ckpt_files:
            ckpt_path = os.path.join(model_dir, ckpt_files[0])
            print(f"Trying direct torch.load: {ckpt_path}")
            model = torch.load(ckpt_path, map_location=device)
            if isinstance(model, dict):
                raise ValueError("Got a state_dict, not a model object")
            model = model.to(device)
            model.eval()
            return model
    except Exception as e:
        print(f"Direct load failed: {e}")

    try:
        from transformers import TimesformerForVideoClassification
        model = TimesformerForVideoClassification.from_pretrained(
            model_dir, local_files_only=True)
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"HuggingFace load failed: {e}")

    raise RuntimeError(
        f"Could not load model from {model_dir}. "
    )


def find_transformer_blocks(model):
    paths = [
        'model.blocks', 'blocks',
        'timesformer.encoder.layer', 'encoder.layer',
        'encoder.layers', 'transformer.layers',
        'model.encoder.layer',
    ]
    for path in paths:
        cur = model
        try:
            for attr in path.split('.'):
                cur = getattr(cur, attr)
            if hasattr(cur, '__len__') and len(cur) > 0:
                return cur
        except AttributeError:
            continue
    return None


def find_attention_modules(block):
    for n, mod in block.named_modules():
        if 'attn' in n.lower() and hasattr(mod, 'forward'):
            return mod
    return None


class FrameVideoDataset(Dataset):
    def __init__(self, csv_path, frames_dir, num_frames=8):
        self.frames_dir = frames_dir
        self.num_frames = num_frames
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.samples = []
        with open(csv_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    self.samples.append((parts[0], int(parts[2])))

        self.video_list = self.samples
        print(f"Dataset: {len(self.samples)} videos from {csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_id, label = self.video_list[idx]
        frame_dir = os.path.join(self.frames_dir, video_id)
        frames = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])

        if len(frames) >= self.num_frames:
            indices = np.linspace(0, len(frames) - 1,
                                  self.num_frames, dtype=int)
        else:
            indices = list(range(len(frames)))
            while len(indices) < self.num_frames:
                indices.append(len(frames) - 1)

        imgs = []
        for i in indices:
            path = os.path.join(frame_dir, frames[i])
            img = Image.open(path).convert('RGB')
            imgs.append(self.transform(img))

        return torch.stack(imgs), label


class LayerFeatureExtractor:
    def __init__(self, model, num_layers=12, capture_attention=True):
        self.model = model
        self.num_layers = num_layers
        self.capture_attention = capture_attention
        self.device = next(model.parameters()).device

        self.blocks = find_transformer_blocks(model)
        if self.blocks is None:
            raise RuntimeError(
                "Cannot find transformer blocks. "
            )
        actual_num = len(self.blocks)
        if actual_num != num_layers:
            print(f"WARNING: Expected {num_layers} layers but found {actual_num}")
            self.num_layers = actual_num

        self.feat_buf = {i: [] for i in range(self.num_layers)}
        self.attn_buf = {i: [] for i in range(self.num_layers)}
        self.labels = []
        self.hooks = []

    def _make_feature_hook(self, layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            if hidden.dim() == 3:
                cls = hidden[:, 0, :].detach().cpu()
            elif hidden.dim() == 2:
                cls = hidden.detach().cpu()
            else:
                cls = hidden.reshape(hidden.shape[0], -1).detach().cpu()

            self.feat_buf[layer_idx].append(cls)
        return hook_fn

    def _make_attention_hook(self, layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]
                if attn_weights is not None:
                    self.attn_buf[layer_idx].append(attn_weights.detach().cpu())
        return hook_fn

    def register_hooks(self):
        for i, block in enumerate(self.blocks):
            if i >= self.num_layers:
                break
            self.hooks.append(block.register_forward_hook(self._make_feature_hook(i)))

            if self.capture_attention:
                attn_module = find_attention_modules(block)
                if attn_module is not None:
                    self.hooks.append(
                        attn_module.register_forward_hook(self._make_attention_hook(i))
                    )

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def extract(self, dataloader):
        self.register_hooks()
        self.model.eval()

        print(f"\nExtracting features from {self.num_layers} layers...")
        with torch.no_grad():
            for batch_idx, (videos, labels) in enumerate(dataloader):
                videos = videos.to(self.device)
                _ = self.model(pixel_values=videos)
                self.labels.append(labels)

                if batch_idx % 20 == 0:
                    print(f"  Batch {batch_idx + 1}/{len(dataloader)}")

        self.remove_hooks()

        out = {}
        for i in range(self.num_layers):
            if self.feat_buf[i]:
                out[i] = {
                    'embeddings': torch.cat(self.feat_buf[i], dim=0).numpy()
                }
                if self.attn_buf[i]:
                    out[i]['attention'] = torch.cat(self.attn_buf[i], dim=0).numpy()
            else:
                print(f"  WARNING: No features captured for layer {i}")

        ys = torch.cat(self.labels, dim=0).numpy()

        for i in sorted(out.keys()):
            emb_shape = out[i]['embeddings'].shape
            attn_shape = out[i]['attention'].shape if 'attention' in out[i] else 'None'
            print(f"  Layer {i:2d}: embeddings {emb_shape}, attention {attn_shape}")

        return out, ys


def train_linear_probe(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', verbose=0)
    clf.fit(X_train_s, y_train)

    y_train_pred = clf.predict(X_train_s)
    y_test_pred = clf.predict(X_test_s)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    cm = confusion_matrix(y_test, y_test_pred)

    per_cls = {}
    for cls_id in np.unique(y_test):
        mask = y_test == cls_id
        if mask.sum() > 0:
            per_cls[int(cls_id)] = float(accuracy_score(
                y_test[mask], y_test_pred[mask]
            ))

    return {
        'train_acc': float(train_acc),
        'test_acc': float(test_acc),
        'per_class_acc': per_cls,
        'confusion_matrix': cm,
        'probe_weights': clf.coef_,
        'probe_bias': clf.intercept_,
        'scaler_mean': scaler.mean_,
        'scaler_std': scaler.scale_,
    }


def run_all_linear_probes(train_layer_results, train_labels,
                          test_layer_results, test_labels, output_dir):
    rows = []

    for layer_idx in sorted(train_layer_results.keys()):
        layer_dir = os.path.join(output_dir, f'layer_{layer_idx:02d}')
        os.makedirs(layer_dir, exist_ok=True)

        train_emb = train_layer_results[layer_idx]['embeddings']
        test_emb = test_layer_results[layer_idx]['embeddings']
        print(f"\nLayer {layer_idx}: train {train_emb.shape}, test {test_emb.shape}")

        np.save(os.path.join(layer_dir, 'embeddings.npy'), train_emb)
        np.save(os.path.join(layer_dir, 'labels.npy'), train_labels)
        np.save(os.path.join(layer_dir, 'test_embeddings.npy'), test_emb)
        np.save(os.path.join(layer_dir, 'test_labels.npy'), test_labels)

        if 'attention' in train_layer_results[layer_idx]:
            attn = train_layer_results[layer_idx]['attention']
            np.save(os.path.join(layer_dir, 'attention_maps.npy'), attn)
            print(f"  Saved attention maps: {attn.shape}")

        print(f"  Training linear probe...")
        probe_result = train_linear_probe(train_emb, train_labels,
                                          test_emb, test_labels)
        print(f"  Train acc: {probe_result['train_acc']:.4f}")
        print(f"  Test acc:  {probe_result['test_acc']:.4f}")

        meta = {
            'layer': layer_idx,
            'train_acc': probe_result['train_acc'],
            'test_acc': probe_result['test_acc'],
            'embedding_dim': int(train_emb.shape[1]),
            'num_train_samples': int(train_emb.shape[0]),
            'num_test_samples': int(test_emb.shape[0]),
            'per_class_acc': probe_result['per_class_acc'],
        }
        with open(os.path.join(layer_dir, 'probe_accuracy.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        np.save(os.path.join(layer_dir, 'probe_weights.npy'),
                probe_result['probe_weights'])
        np.save(os.path.join(layer_dir, 'confusion_matrix.npy'),
                probe_result['confusion_matrix'])

        rows.append({
            'layer': layer_idx,
            'train_acc': probe_result['train_acc'],
            'test_acc': probe_result['test_acc'],
            'embedding_dim': int(train_emb.shape[1]),
        })

    return rows


def coalition_value(layer_results, labels, picked,
                    test_size=0.3, random_state=42):
    if len(picked) == 0:
        _, counts = np.unique(labels, return_counts=True)
        return float(counts.max()) / len(labels)

    x = np.concatenate(
        [layer_results[i]['embeddings'] for i in picked], axis=1
    )

    X_train, X_test, y_train, y_test = train_test_split(
        x, labels, test_size=test_size,
        random_state=random_state, stratify=labels
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
    clf.fit(X_train_s, y_train)
    return float(clf.score(X_test_s, y_test))


def exact_shapley(layer_results, labels, num_layers):
    players = list(range(num_layers))
    n = len(players)
    total_sets = 2 ** n

    print(f"\nEXACT SHAPLEY VALUES")
    print(f"Players: {n}, Coalitions: {total_sets}")

    seen = {}
    done = 0
    t0 = time.time()

    for size in range(n + 1):
        for combo in combinations(players, size):
            combo_key = frozenset(combo)
            seen[combo_key] = coalition_value(layer_results, labels, list(combo))
            done += 1

            if done % 100 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                remaining = (total_sets - done) / rate
                print(f"  Evaluated {done}/{total_sets} "
                      f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    elapsed = time.time() - t0
    print(f"\n  All {total_sets} coalitions evaluated in {elapsed:.1f}s")

    svs = {}
    for i in players:
        sv = 0.0
        others = [j for j in players if j != i]
        for size in range(n):
            for combo in combinations(others, size):
                combo_key = frozenset(combo)
                with_i = combo_key | {i}
                marginal = seen[with_i] - seen[combo_key]
                weight = (factorial(size) * factorial(n - size - 1)
                          / factorial(n))
                sv += weight * marginal
        svs[i] = sv
        print(f"  Layer {i:2d}: Shapley value = {sv:+.6f}")

    return svs, seen


def plot_results(rows, svs, output_dir):
    out_dir = os.path.join(output_dir, 'summary')
    os.makedirs(out_dir, exist_ok=True)

    layers = [row['layer'] for row in rows]
    test_accs = [row['test_acc'] for row in rows]
    train_accs = [row['train_acc'] for row in rows]
    shapley_vals = [svs.get(layer, 0) for layer in layers]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(layers, train_accs, 'b-o', label='Train', markersize=6)
    axes[0].plot(layers, test_accs, 'r-o', label='Test', markersize=6)
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Linear Probe Accuracy per Layer')
    axes[0].legend()
    axes[0].set_xticks(layers)
    axes[0].grid(True, alpha=0.3)

    colors = ['green' if v >= 0 else 'red' for v in shapley_vals]
    axes[1].bar(layers, shapley_vals, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Shapley Value')
    axes[1].set_title('Shapley Value per Layer')
    axes[1].set_xticks(layers)
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].grid(True, alpha=0.3)

    gains = [test_accs[0]] + [test_accs[i] - test_accs[i-1]
                               for i in range(1, len(test_accs))]
    colors_g = ['green' if g >= 0 else 'red' for g in gains]
    axes[2].bar(layers, gains, color=colors_g, alpha=0.7, edgecolor='black')
    axes[2].set_xlabel('Layer')
    axes[2].set_ylabel('Accuracy Gain')
    axes[2].set_title('Marginal Accuracy Gain (vs Previous Layer)')
    axes[2].set_xticks(layers)
    axes[2].axhline(y=0, color='black', linewidth=0.5)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'layer_comparison_plot.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved plot to {out_dir}/layer_comparison_plot.png")


def save_summary(rows, svs, coal_cache, output_dir):
    out_dir = os.path.join(output_dir, 'summary')
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'all_layer_accuracies.csv'), index=False)

    sv_dump = {
        'shapley_values': {str(k): v for k, v in svs.items()},
        'ranking': [str(k) for k, v in sorted(
            svs.items(), key=lambda x: x[1], reverse=True
        )],
        'description': (
            'Shapley value = average marginal contribution of each layer '
            'across all possible coalitions of layers. Higher = more important.'
        ),
    }
    with open(os.path.join(out_dir, 'shapley_values.json'), 'w') as f:
        json.dump(sv_dump, f, indent=2)

    coal_dump = {}
    for combo_key, value in coal_cache.items():
        key = ','.join(str(x) for x in sorted(combo_key)) or 'empty'
        coal_dump[key] = value
    with open(os.path.join(out_dir, 'coalition_values.json'), 'w') as f:
        json.dump(coal_dump, f, indent=2)

    print(f"\nFINAL LAYER RANKING BY SHAPLEY VALUE")
    for rank, (layer, sv) in enumerate(sorted(
        svs.items(), key=lambda x: x[1], reverse=True
    )):
        acc = next(row['test_acc'] for row in rows if row['layer'] == layer)
        print(f"  #{rank+1:2d}  Layer {layer:2d}  |  "
              f"Shapley: {sv:+.6f}  |  Probe acc: {acc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--frames_dir', type=str, required=True)
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--test_csv', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--inspect_only', action='store_true')
    parser.add_argument('--skip_shapley', action='store_true')
    parser.add_argument('--multi_gpu', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n[1/5] Loading model...")
    model = load_model(args.model_dir)

    if args.inspect_only:
        inspect_model(args.model_dir)
        sys.exit(0)

    if args.multi_gpu and torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    print("\n[2/5] Loading datasets...")
    base_model = model.module if isinstance(model, nn.DataParallel) else model

    train_dataset = FrameVideoDataset(
        csv_path=args.train_csv,
        frames_dir=args.frames_dir,
        num_frames=args.num_frames
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    test_dataset = FrameVideoDataset(
        csv_path=args.test_csv,
        frames_dir=args.frames_dir,
        num_frames=args.num_frames
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    print("\n[3/5] Extracting layer features from train set...")
    train_extractor = LayerFeatureExtractor(
        base_model, num_layers=args.num_layers, capture_attention=True
    )
    train_layer_results, train_labels = train_extractor.extract(train_dataloader)

    print("\n    Extracting layer features from test set...")
    test_extractor = LayerFeatureExtractor(
        base_model, num_layers=args.num_layers, capture_attention=False
    )
    test_layer_results, test_labels = test_extractor.extract(test_dataloader)

    print("\n[4/5] Training linear probes...")
    rows = run_all_linear_probes(
        train_layer_results, train_labels,
        test_layer_results, test_labels,
        args.output_dir
    )

    if not args.skip_shapley:
        print("\n[5/5] Computing exact Shapley values...")
        svs, coal_cache = exact_shapley(
            train_layer_results, train_labels, train_extractor.num_layers
        )
    else:
        print("\n[5/5] Skipping Shapley (--skip_shapley)")
        svs = {row['layer']: 0.0 for row in rows}
        coal_cache = {}

    save_summary(rows, svs, coal_cache, args.output_dir)
    plot_results(rows, svs, args.output_dir)

    print("\nAll results saved to:", args.output_dir)


if __name__ == '__main__':
    main()
