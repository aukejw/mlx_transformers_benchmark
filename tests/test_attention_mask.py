import mlx.core as mx
import numpy as np
import pytest
import torch

from mtb.attention_mask import (
    create_mlx_attention_mask,
    create_torch_attention_mask,
    validate_attention_kwargs,
)


@pytest.fixture
def torch_dtype():
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float16)
    return torch.float16


@pytest.fixture
def mlx_dtype():
    mx.set_default_device(mx.DeviceType.cpu)
    return mx.float16


def test_validate_attention_kwargs():
    validate_attention_kwargs(feature_dim=16, num_heads=4, mask_type=None)
    validate_attention_kwargs(feature_dim=32, num_heads=8, mask_type="causal")

    with pytest.raises(ValueError):
        validate_attention_kwargs(feature_dim=16, num_heads=5, mask_type=None)

    with pytest.raises(ValueError):
        validate_attention_kwargs(feature_dim=16, num_heads=4, mask_type="nonexisting")


def test_create_torch_attention_mask_causal(torch_dtype):
    num_tokens = 5
    mask = create_torch_attention_mask(
        mask_type="causal",
        num_tokens=num_tokens,
        device=torch.device("cpu"),
        dtype=torch.float16,
    )
    assert mask is not None
    assert mask.shape == (num_tokens, num_tokens)
    assert mask.dtype == torch.float16


def test_create_torch_attention_mask_none(torch_dtype):
    num_tokens = 5
    mask = create_torch_attention_mask(
        mask_type=None,
        num_tokens=num_tokens,
        device=torch.device("cpu"),
        dtype=torch.float16,
    )
    assert mask is None


def test_create_torch_attention_mask_invalid(torch_dtype):
    num_tokens = 5
    with pytest.raises(ValueError, match="Unknown mask type: invalid"):
        create_torch_attention_mask(
            mask_type="invalid",
            num_tokens=num_tokens,
            device=torch.device("cpu"),
            dtype=torch.float16,
        )


def test_create_mlx_attention_mask_causal(mlx_dtype):
    num_tokens = 5
    mask = create_mlx_attention_mask(
        mask_type="causal",
        device=mx.default_device(),
        dtype=mx.float16,
        num_tokens=num_tokens,
    )
    assert mask is not None
    assert mask.shape == (num_tokens, num_tokens)
    assert mask.dtype == mx.float16


def test_create_mlx_attention_mask_none(mlx_dtype):
    num_tokens = 5
    mask = create_mlx_attention_mask(
        mask_type=None,
        num_tokens=num_tokens,
        device=mx.default_device(),
        dtype=mx.float16,
    )
    assert mask is None


def test_create_mlx_attention_mask_none_compile(mlx_dtype):
    num_tokens = 5
    mask = create_mlx_attention_mask(
        mask_type=None,
        num_tokens=num_tokens,
        device=mx.default_device(),
        dtype=mx.float16,
        compile=True,
    )
    assert mask is not None
    assert mask.shape == (num_tokens, num_tokens)
    assert mask.dtype == mx.float16


def test_create_mlx_attention_mask_invalid(mlx_dtype):
    num_tokens = 5
    with pytest.raises(ValueError, match="Unknown mask type: invalid"):
        create_mlx_attention_mask(
            mask_type="invalid",
            device=mx.default_device(),
            dtype=mx.float16,
            num_tokens=num_tokens,
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_attention_mask_equality(dtype):
    num_tokens = 5

    torch_dtype = dtype
    torch_mask: torch.Tensor = create_torch_attention_mask(
        mask_type="causal",
        num_tokens=num_tokens,
        device=torch.device("cpu"),
        dtype=torch_dtype,
    )

    mlx_dtype = {
        torch.float16: mx.float16,
        torch.float32: mx.float32,
        torch.bfloat16: mx.bfloat16,
    }[torch_dtype]
    mlx_mask: mx.array = create_mlx_attention_mask(
        mask_type="causal",
        device=mx.default_device(),
        dtype=mlx_dtype,
        num_tokens=num_tokens,
    )
    assert torch_mask is not None
    assert mlx_mask is not None
    assert torch_mask.shape == mlx_mask.shape
    assert torch_mask.dtype == torch_dtype
    assert mlx_mask.dtype == mlx_dtype

    # both masks are additive causal masks by default
    np.testing.assert_equal(mlx_mask.astype(mx.float32), torch_mask.float().numpy())
