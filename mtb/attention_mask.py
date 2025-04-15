from typing import Optional

import mlx.core as mx
import mlx.nn
import torch


def validate_attention_kwargs(
    feature_dim: int,
    num_heads: int,
    mask_type: Optional[str],
):
    """Check that arguments are correct."""

    if feature_dim % num_heads != 0:
        raise ValueError(
            f"feature_dim={feature_dim} must be divisible by num_heads={num_heads}"
        )

    if mask_type not in (None, "causal"):
        raise ValueError(
            f"mask_type must be None or 'causal', but received '{mask_type}'"
        )

    return


def create_attention_mask(
    framework: str,
    **kwargs,
) -> Optional[torch.Tensor]:
    """Create an attention mask for the given framework."""
    if framework == "torch":
        return create_torch_attention_mask(**kwargs)
    elif framework == "mlx":
        return create_mlx_attention_mask(**kwargs)
    else:
        raise NotImplementedError(
            f"Unknown framework {framework}. " "Only 'torch' and 'mlx' are supported."
        )


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
            # compilation with None inputs is not supported
            mask = mx.zeros((num_tokens, num_tokens), dtype=dtype)
        else:
            mask = None

    else:
        raise ValueError(f"Unknown mask type: {mask_type}")

    return mask
