"""gradient-based head importance scoring (Michel et al. 2019).

computes importance as expected gradient magnitude w.r.t learnable head masks.
way faster than zero-out ablation since you only need one backward pass per batch.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List
from config import AblationConfig


class HeadMask:
    """manages learnable mask variables for all attention heads (temporal + spatial)."""

    def __init__(self, config: AblationConfig):
        self.config = config
        # mask tensors shape: (num_layers, num_heads) for each attn type
        self.temporal_masks = torch.ones(
            config.num_layers, config.num_heads,
            device=config.device, requires_grad=True
        )
        self.spatial_masks = torch.ones(
            config.num_layers, config.num_heads,
            device=config.device, requires_grad=True
        )
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

    def _make_mask_hook(self, layer_idx: int, attn_type: str):
        """hook that scales each head's output by its mask variable."""
        masks = self.temporal_masks if attn_type == "temporal" else self.spatial_masks
        num_heads = self.config.num_heads
        head_dim = self.config.head_dim

        def hook_fn(module, input, output):
            context = output[0]  # (B, N, C)
            B, N, C = context.shape
            context = context.view(B, N, num_heads, head_dim)

            # scale each head by its mask - gradient through this gives importance
            mask = masks[layer_idx]  # (num_heads,)
            context = context * mask[None, None, :, None]
            context = context.view(B, N, C)

            if len(output) > 1:
                return (context,) + output[1:]
            return (context,)

        return hook_fn

    def register_hooks(self, model) -> None:
        """Register masking hooks on all attention modules."""
        self.remove_hooks()
        encoder = model.timesformer.encoder

        for i, layer in enumerate(encoder.layer):
            h1 = layer.temporal_attention.attention.register_forward_hook(
                self._make_mask_hook(i, "temporal")
            )
            h2 = layer.attention.attention.register_forward_hook(
                self._make_mask_hook(i, "spatial")
            )
            self._hooks.extend([h1, h2])

    def remove_hooks(self) -> None:
        """Remove all hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get_importance_scores(self) -> Dict[str, torch.Tensor]:
        """Return absolute gradient of masks (importance scores)."""
        scores = {}
        if self.temporal_masks.grad is not None:
            scores["temporal"] = self.temporal_masks.grad.abs().detach().cpu()
        if self.spatial_masks.grad is not None:
            scores["spatial"] = self.spatial_masks.grad.abs().detach().cpu()
        return scores

    def zero_grad(self) -> None:
        """Zero out accumulated gradients."""
        if self.temporal_masks.grad is not None:
            self.temporal_masks.grad.zero_()
        if self.spatial_masks.grad is not None:
            self.spatial_masks.grad.zero_()


def compute_gradient_importance(
    model,
    dataloader,
    config: AblationConfig,
    num_batches: int = None,
) -> pd.DataFrame:
    """Compute gradient-based importance scores for all heads.

    Args:
        model: TimeSformer model (will not be modified)
        dataloader: DataLoader for computing gradients
        config: Configuration
        num_batches: Number of batches to average over

    Returns:
        DataFrame with importance scores for all heads
    """
    num_batches = num_batches or config.grad_num_batches

    head_mask = HeadMask(config)
    head_mask.register_hooks(model)

    # Accumulate importance scores
    temporal_importance = torch.zeros(config.num_layers, config.num_heads)
    spatial_importance = torch.zeros(config.num_layers, config.num_heads)

    model.eval()  # keep eval mode but enable grad for masks
    criterion = nn.CrossEntropyLoss()

    print(f"\nComputing gradient-based importance over {num_batches} batches...")

    batch_count = 0
    for pixel_values, labels in tqdm(dataloader, desc="Grad importance",
                                     total=num_batches):
        if batch_count >= num_batches:
            break

        pixel_values = pixel_values.to(config.device)
        labels = labels.to(config.device)

        head_mask.zero_grad()

        # Forward pass (model params frozen, only mask grads flow)
        outputs = model(pixel_values=pixel_values)
        loss = criterion(outputs.logits, labels)

        # Backward pass to get gradients on mask variables
        loss.backward()

        scores = head_mask.get_importance_scores()
        if "temporal" in scores:
            temporal_importance += scores["temporal"]
        if "spatial" in scores:
            spatial_importance += scores["spatial"]

        batch_count += 1

    head_mask.remove_hooks()

    # Average over batches
    temporal_importance /= batch_count
    spatial_importance /= batch_count

    # Build results DataFrame
    results = []
    for layer_idx in range(config.num_layers):
        for head_idx in range(config.num_heads):
            results.append({
                "layer": layer_idx,
                "attn_type": "temporal",
                "head": head_idx,
                "importance": temporal_importance[layer_idx, head_idx].item(),
            })
            results.append({
                "layer": layer_idx,
                "attn_type": "spatial",
                "head": head_idx,
                "importance": spatial_importance[layer_idx, head_idx].item(),
            })

    df = pd.DataFrame(results)

    # Save
    output_path = config.output_dir / "gradient_importance.csv"
    df.to_csv(output_path, index=False)
    print(f"Gradient importance scores saved to {output_path}")

    return df
