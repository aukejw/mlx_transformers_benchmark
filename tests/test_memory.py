import mlx.core as mx
import numpy as np
import pytest
import torch

from mtb import FLAG_ON_MAC
from mtb.memory import (
    bytes_to_gb,
    get_mlx_memory_gb,
    get_process_memory_gb,
    get_torch_memory_gb,
)


def test_get_process_memory():
    initial_memory = get_process_memory_gb()

    array = np.ones((1_000, 1_000), dtype=np.float32)
    array_gb = bytes_to_gb(array.nbytes)

    # allow for some overhead
    difference_gb = get_process_memory_gb() - initial_memory
    assert difference_gb > array_gb


@pytest.mark.skipif(not torch.mps.is_available(), reason="MPS is not available")
def test_get_torch_memory_gb_mps():
    initial_process_memory = get_process_memory_gb()
    initial_torch_memory = get_torch_memory_gb()

    # create a tensor on cpu -> only uses RAM
    tensor = torch.ones((1_000, 1_000), dtype=torch.float32, device="cpu")
    tensor_gb = bytes_to_gb(tensor.element_size() * tensor.numel())

    # torch memory should stay the same here
    difference_gb = get_torch_memory_gb() - initial_torch_memory
    assert difference_gb == 0

    # we should use at least as much memor than the tensor
    difference_gb = get_process_memory_gb() - initial_process_memory
    assert difference_gb > tensor_gb

    # but if we allocate a tensor on GPU -> uses mps memory buffers
    tensor = tensor.to("mps")
    difference_gb = get_torch_memory_gb() - initial_torch_memory
    assert difference_gb == tensor_gb


@pytest.mark.skipif(not FLAG_ON_MAC, reason="Must run on Mac")
def test_get_mlx_memory_gb():
    initial_process_memory = get_process_memory_gb()
    initial_mlx_memory = get_mlx_memory_gb()

    # create a tensor in mlx is lazy -> not actually allocated yet
    tensor = mx.ones((1_000, 1_000), dtype=mx.float32)
    tensor_gb = bytes_to_gb(tensor.nbytes)

    # quite some RAM should be allocated, but much less than the tensor
    difference_gb = get_process_memory_gb() - initial_process_memory
    assert difference_gb < 1e-3
    assert difference_gb < tensor_gb

    # mlx should have only allocated a 4-byte placeholder
    difference_gb = get_mlx_memory_gb() - initial_mlx_memory
    assert difference_gb == bytes_to_gb(4)

    # but once we evaluate, the memory is allocated (plus some overhead)
    mx.eval(tensor)
    difference_gb = get_mlx_memory_gb() - initial_mlx_memory
    assert difference_gb > tensor_gb
