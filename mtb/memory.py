import os

import mlx.core as mx
import psutil
import torch

process = None


def bytes_to_gb(bytes: int) -> float:
    """Convert bytes to gigabytes."""
    return bytes / (1024**3)


def get_process_memory_gb() -> float:
    """Return the current process' allocated memory in GB."""
    global process
    if process is None:
        pid = os.getpid()
        process = psutil.Process(pid)
    memory_info = process.memory_info()
    return bytes_to_gb(memory_info.rss)


def get_torch_memory_gb() -> float:
    """Return the memory allocated by torch tensors in GB.

    Tensors on GPU typically live in a separate memory space, despite the
    unified memory model. This function returns the allocated GPU memory.

    """
    if torch.mps.is_available():
        mem = torch.mps.current_allocated_memory()
    elif torch.cuda.is_available():
        mem = torch.cuda.memory_allocated()
    else:
        # all data lives in RAM?
        mem = 0

    return bytes_to_gb(mem)


def get_mlx_memory_gb() -> float:
    """Return the memory allocated by mlx tensors in bytes.

    Note that mlx is lazy, which means that tensors are only allocated
    once evaluated! For example, creating a tensor with `mx.ones` only
    allocates `tensor.size * 4` bytes when evaluated:

        >>> import mlx
        >>> import mlx.core as mx
        >>> tensor = mx.ones((1_000, 1_000), dtype=mx.float32)
        >>> assert mx.get_active_memory() == 12
        >>> mx.eval(tensor)
        >>> assert mx.get_active_memory() > tensor.size * 4

    """
    mem = mx.get_active_memory()
    return bytes_to_gb(mem)
