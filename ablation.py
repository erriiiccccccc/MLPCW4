"""hook for zeroing out individual temporal attention heads in TimeSformer.
basically intercepts the attention output and sets a specific head to 0.
"""


def make_ablation_hook(head_idx: int, num_heads: int = 12, head_dim: int = 64):
    """creates a forward hook that zeros out one attention head.

    hooks into TimesformerSelfAttention output and sets teh target head to 0.
    head_idx is 0-indexed so valid range is 0-11 for a 12-head model.
    """

    def hook_fn(module, input, output):
        # output is a tuple (context, attn_probs) - we only care about context
        context = output[0]
        B, N, C = context.shape

        # reshape so we can index individual heads
        context = context.view(B, N, num_heads, head_dim)
        context[:, :, head_idx, :] = 0.0
        context = context.view(B, N, C)

        # return with the rest of the tuple if theres anything else
        if len(output) > 1:
            return (context,) + output[1:]
        return (context,)

    return hook_fn
