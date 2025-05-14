import mlx.core as mx
import numpy as np
import pytest
import torch

from mtb import FLAG_ON_MAC
from mtb.system.memory import (
    bytes_to_gib,
    estimate_model_size,
    get_available_ram_gib,
    get_mlx_memory_gib,
    get_process_memory_gib,
    get_torch_memory_gib,
    get_used_ram_gib,
)


def test_get_process_memory():
    initial_memory = get_process_memory_gib()

    array = np.ones((10_000, 10_000), dtype=np.float32)
    array_gib = bytes_to_gib(array.nbytes)

    # we should have allocated process memory, but allow for overhead, deduplication, etc.
    difference_gib = get_process_memory_gib() - initial_memory
    assert difference_gib > 0.1 * array_gib


def test_get_ram():
    memory = get_available_ram_gib()
    assert memory > 0
    memory = get_used_ram_gib()
    assert memory > 0


@pytest.mark.skipif(not torch.mps.is_available(), reason="MPS is not available")
def test_get_torch_memory_gib_mps():
    initial_process_memory = get_process_memory_gib()
    initial_torch_memory = get_torch_memory_gib()

    # create a tensor on cpu -> only uses RAM
    tensor = torch.ones((10_000, 10_000), dtype=torch.float32, device="cpu")
    tensor_gib = bytes_to_gib(tensor.element_size() * tensor.numel())

    # torch memory should stay the same for now
    difference_gib = get_torch_memory_gib() - initial_torch_memory
    assert difference_gib == 0

    # we should use some RAM for this tensor
    difference_gib = get_process_memory_gib() - initial_process_memory
    assert difference_gib > 0

    # but if we allocate a tensor on GPU -> uses mps memory buffers -> exact match
    tensor = tensor.to("mps")
    difference_gib = get_torch_memory_gib() - initial_torch_memory
    assert difference_gib == tensor_gib


@pytest.mark.skipif(not FLAG_ON_MAC, reason="Must run on Mac")
def test_get_mlx_memory_gib():
    initial_process_memory = get_process_memory_gib()
    initial_mlx_memory = get_mlx_memory_gib()

    # create a tensor in mlx is lazy -> not actually allocated yet
    tensor = mx.ones((10_000, 10_000), dtype=mx.float32)
    tensor_gib = bytes_to_gib(tensor.nbytes)

    # quite some RAM should be allocated, but much less than the tensor
    difference_gib = get_process_memory_gib() - initial_process_memory
    assert difference_gib < 1e-3
    assert difference_gib < tensor_gib

    # mlx should have only allocated a 4-byte placeholder
    difference_gib = get_mlx_memory_gib() - initial_mlx_memory
    assert difference_gib == bytes_to_gib(4)

    # but once we evaluate, the memory is allocated (plus some overhead)
    mx.eval(tensor)
    difference_gib = get_mlx_memory_gib() - initial_mlx_memory
    assert difference_gib > tensor_gib


def test_estimate_model_size():
    for num_params, dtype, bits in [
        (1_000_000, "float32", 32),
        (500_000, "float16", 16),
        (200_000, "bfloat16", 16),
        (10_000, "int8", 8),
        (2_000, "int6", 6),
        (2_000, "int4", 4),
        (2_000, "int3", 3),
    ]:
        estimated_model_size = estimate_model_size(num_params, dtype)
        size = bytes_to_gib(num_params * bits / 8)
        assert estimated_model_size == size
