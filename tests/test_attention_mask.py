import mlx
import mlx.core as mx
import mlx.nn
import numpy as np
import pytest
import torch

from mtb.attention_mask import create_mlx_attention_mask, create_torch_attention_mask


@pytest.fixture
def torch_attention_layer():
    try:
        torch.set_default_device("mps")
    except:
        torch.set_default_device("cpu")
    return torch.nn.MultiheadAttention(embed_dim=16, num_heads=4)


@pytest.fixture
def mlx_attention_layer():
    try:
        mx.set_default_device(mx.DeviceType.gpu)
    except ValueError:
        mx.set_default_device(mx.DeviceType.cpu)
    return mlx.nn.MultiHeadAttention(dims=16, num_heads=4)


def test_create_torch_attention_mask_causal(torch_attention_layer):
    num_tokens = 5
    mask = create_torch_attention_mask(
        mask_type="causal",
        attention_layer=torch_attention_layer,
        num_tokens=num_tokens,
    )
    assert mask is not None
    assert mask.shape == (num_tokens, num_tokens)
    assert mask.dtype == torch.bool


def test_create_torch_attention_mask_none(torch_attention_layer):
    num_tokens = 5
    mask = create_torch_attention_mask(
        mask_type=None,
        attention_layer=torch_attention_layer,
        num_tokens=num_tokens,
    )
    assert mask is None


def test_create_torch_attention_mask_invalid(torch_attention_layer):
    num_tokens = 5
    with pytest.raises(ValueError, match="Unknown mask type: invalid"):
        create_torch_attention_mask(
            mask_type="invalid",
            attention_layer=torch_attention_layer,
            num_tokens=num_tokens,
        )


def test_create_mlx_attention_mask_causal(mlx_attention_layer):
    num_tokens = 5
    mask = create_mlx_attention_mask(
        mask_type="causal",
        attention_layer=mlx_attention_layer,
        num_tokens=num_tokens,
    )
    assert mask is not None
    assert mask.shape == (num_tokens, num_tokens)


def test_create_mlx_attention_mask_none(mlx_attention_layer):
    num_tokens = 5
    mask = create_mlx_attention_mask(
        mask_type=None,
        attention_layer=mlx_attention_layer,
        num_tokens=num_tokens,
    )
    assert mask is None


def test_create_mlx_attention_mask_none_compile(mlx_attention_layer):
    num_tokens = 5
    mask = create_mlx_attention_mask(
        mask_type=None,
        attention_layer=mlx_attention_layer,
        num_tokens=num_tokens,
        compile=True,
    )
    assert mask is not None
    assert mask.shape == (num_tokens, num_tokens)


def test_create_mlx_attention_mask_invalid(mlx_attention_layer):
    num_tokens = 5
    with pytest.raises(ValueError, match="Unknown mask type: invalid"):
        create_mlx_attention_mask(
            mask_type="invalid",
            attention_layer=mlx_attention_layer,
            num_tokens=num_tokens,
        )


def test_attention_mask_equality(torch_attention_layer, mlx_attention_layer):
    num_tokens = 5
    torch_mask: torch.Tensor = create_torch_attention_mask(
        mask_type="causal",
        attention_layer=torch_attention_layer,
        num_tokens=num_tokens,
    )
    mlx_mask: mx.array = create_mlx_attention_mask(
        mask_type="causal",
        attention_layer=mlx_attention_layer,
        num_tokens=num_tokens,
    )
    assert torch_mask is not None
    assert mlx_mask is not None
    assert torch_mask.shape == mlx_mask.shape

    # the mlx mask is an additive causal mask with -infty for masked positions.
    mlx_mask_numpy = (mlx_mask < 0).astype(mx.float32)

    # the torch mask is - since pytorch 1.9.0 - a boolean mask: faster and memory-efficient
    torch_mask_numpy = torch_mask.cpu().numpy().astype(np.float32)

    np.testing.assert_equal(mlx_mask_numpy, torch_mask_numpy)
