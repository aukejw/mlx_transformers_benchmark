from typing import Optional

import mlx.core as mx
import mlx.nn
import torch


def create_torch_attention_mask(
    mask_type: Optional[str],
    dtype: torch.dtype,
    device: torch.device,
    num_tokens: int,
    compile: bool = False,
) -> torch.Tensor:
    """Create an attention mask for the torch function."""
    if mask_type == "causal":
        max_float = torch.finfo(dtype).max

        lower_triangular = torch.ones(
            num_tokens,
            num_tokens,
            dtype=torch.bool,
        ).tril(diagonal=0)

        mask = (
            torch.zeros(num_tokens, num_tokens)
            .to(dtype)
            .masked_fill_(lower_triangular.logical_not(), -max_float)
            .to(dtype=dtype, device=device)
        )

    elif mask_type is None:
        mask = None

    else:
        raise ValueError(f"Unknown mask type: {mask_type}")

    return mask


def create_mlx_attention_mask(
    mask_type: Optional[str],
    dtype: mx.Dtype,
    device: mx.Device,
    num_tokens: int,
    compile: bool = False,
) -> mx.array:
    """Create an attention mask for the MLX layer."""
    if mask_type == "causal":
        mask = mlx.nn.MultiHeadAttention.create_additive_causal_mask(
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
