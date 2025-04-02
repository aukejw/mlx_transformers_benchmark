from typing import Optional

import mlx
import mlx.core as mx
import mlx.nn
import torch.nn


def create_torch_attention_mask(
    mask_type: Optional[str],
    attention_layer: torch.nn.MultiheadAttention,
    num_tokens: int,
    compile: bool = False,
) -> torch.Tensor:
    """Create an attention mask for the torch function."""
    if mask_type == "causal":
        param = next(attention_layer.parameters())

        mask = (
            torch.triu(
                torch.ones(num_tokens, num_tokens),
                diagonal=1,
            )
            .bool()
            .to(device=param.device)
        )

    elif mask_type is None:
        mask = None

    else:
        raise ValueError(f"Unknown mask type: {mask_type}")

    return mask


def create_mlx_attention_mask(
    mask_type: Optional[str],
    attention_layer: mlx.nn.MultiHeadAttention,
    num_tokens: int,
    compile: bool = False,
) -> mx.array:
    """Create an attention mask for the MLX layer."""
    dtype = attention_layer.query_proj.weight.dtype

    if mask_type == "causal":
        mask = attention_layer.create_additive_causal_mask(
            N=num_tokens,
            dtype=dtype,
        )

    elif mask_type is None:
        if compile:
            mask = mx.zeros((num_tokens, num_tokens), dtype=dtype)
        else:
            mask = None

    else:
        raise ValueError(f"Unknown mask type: {mask_type}")

    return mask
