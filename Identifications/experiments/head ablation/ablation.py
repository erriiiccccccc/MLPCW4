"""Hooks for zeroing individual attention heads."""


def make_ablation_hook(head_idx: int, num_heads: int = 12, head_dim: int = 64):
    """Create a forward hook that zeros one attention head."""

    def hook_fn(module, input, output):
        context = output[0]
        B, N, C = context.shape

        context = context.view(B, N, num_heads, head_dim)
        context[:, :, head_idx, :] = 0.0
        context = context.view(B, N, C)

        if len(output) > 1:
            return (context,) + output[1:]
        return (context,)

    return hook_fn
