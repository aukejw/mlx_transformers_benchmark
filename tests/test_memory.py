import mlx.core as mx
import numpy as np
import pytest
import torch

from mtb import FLAG_ON_MAC
from mtb.memory import (
    MemoryTracker,
    bytes_to_gb,
    get_mlx_memory_gb,
    get_process_memory_gb,
    get_torch_memory_gb,
)


def test_get_process_memory():
    initial_memory = get_process_memory_gb()

    array = np.ones((10_000, 10_000), dtype=np.float32)
    array_gb = bytes_to_gb(array.nbytes)

    # we should have allocated process memory, but allow for overhead, deduplication, etc.
    difference_gb = get_process_memory_gb() - initial_memory
    assert difference_gb > 0.1 * array_gb


@pytest.mark.skipif(not torch.mps.is_available(), reason="MPS is not available")
def test_get_torch_memory_gb_mps():
    initial_process_memory = get_process_memory_gb()
    initial_torch_memory = get_torch_memory_gb()

    # create a tensor on cpu -> only uses RAM
    tensor = torch.ones((10_000, 10_000), dtype=torch.float32, device="cpu")
    tensor_gb = bytes_to_gb(tensor.element_size() * tensor.numel())

    # torch memory should stay the same for now
    difference_gb = get_torch_memory_gb() - initial_torch_memory
    assert difference_gb == 0

    # we should use some RAM for this tensor
    difference_gb = get_process_memory_gb() - initial_process_memory
    assert difference_gb > 0

    # but if we allocate a tensor on GPU -> uses mps memory buffers -> exact match
    tensor = tensor.to("mps")
    difference_gb = get_torch_memory_gb() - initial_torch_memory
    assert difference_gb == tensor_gb


@pytest.mark.skipif(not FLAG_ON_MAC, reason="Must run on Mac")
def test_get_mlx_memory_gb():
    initial_process_memory = get_process_memory_gb()
    initial_mlx_memory = get_mlx_memory_gb()

    # create a tensor in mlx is lazy -> not actually allocated yet
    tensor = mx.ones((10_000, 10_000), dtype=mx.float32)
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


def test_memory_tracker():
    # print(get_process_memory_gb())
    tracker = MemoryTracker(framework="torch", backend="mps")
    assert tracker.get_used_memory() == 0

    array = np.ones((10_000, 10_000), dtype=np.float32)
    array_gb = bytes_to_gb(array.nbytes)

    # we should have allocated process memory, but allow for overhead, deduplication, etc.
    used_memory = tracker.get_used_memory()
    assert used_memory > 0.5 * array_gb

    del array

    new_used_memory = tracker.get_used_memory()
    assert new_used_memory < used_memory
