import os
from typing import Optional

import mlx.core as mx
import psutil
import torch


def bytes_to_gib(bytes: int) -> float:
    """Convert bytes to gibibytes."""
    return bytes / (1024**3)


def get_process_memory_gib() -> float:
    """Return the current process' allocated memory in GiB."""
    pid = os.getpid()
    process = psutil.Process(pid)
    memory_info = process.memory_info()
    return bytes_to_gib(memory_info.rss)


def get_available_ram_gib() -> float:
    """Return the available RAM in GiB."""
    ram = psutil.virtual_memory().available
    return bytes_to_gib(ram)


def get_used_ram_gib() -> float:
    """Return the used RAM in GiB."""
    ram = psutil.virtual_memory().used
    return bytes_to_gib(ram)


def get_torch_memory_gib(backend: Optional[str] = None) -> float:
    """Return the memory allocated by torch tensors in GiB.

    Tensors on GPU typically live in a separate memory space, despite the
    unified memory model. This function returns the allocated GPU memory.

    """
    if backend is None:
        if torch.mps.is_available():
            backend = "mps"
        elif torch.cuda.is_available():
            backend = "cuda"
        else:
            backend = "cpu"

    if backend == "mps":
        mem = torch.mps.current_allocated_memory()
    elif backend == "cuda":
        mem = torch.cuda.memory_allocated()
    else:
        # all data lives in RAM -- crude approximation
        mem = get_process_memory_gib()

    return bytes_to_gib(mem)


def get_mlx_memory_gib() -> float:
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
    return bytes_to_gib(mem)


def estimate_model_size(
    num_params: int,
    dtype: str,
) -> float:
    """Estimate the model size in GiB."""
    dtype_to_bits = {
        "float32": 32,
        "bfloat16": 16,
        "float16": 16,
        "int8": 8,
        "int6": 6,  # not always true in practice
        "int4": 4,
        "int3": 3,  # not always true in practice
    }
    return bytes_to_gib(num_params * dtype_to_bits[dtype] / 8)


def get_lmstudio_memory():
    """Return the memory allocated to LmStudio processes in GiB.

    This includes all helper processes as well.

    Returns:
        A dictionary mapping process names to their memory usage in GiB.
        A key 'total' is included as well, summing all other processes.

    """
    memory = dict()
    for proc in psutil.process_iter():
        try:
            if proc.name().lower().replace(" ", "").startswith("lmstudio"):
                memory[proc.name] = bytes_to_gib(proc.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    memory["total"] = sum(memory.values())
    return memory
