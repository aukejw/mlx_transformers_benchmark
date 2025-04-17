import os

import mlx.core as mx
import psutil
import torch

from mtb import FLAG_ON_LINUX, FLAG_ON_MAC

process = None


class MemoryTracker:
    """Track memory in a separate process."""

    def __init__(self, framework: str, backend: str, sleep: float = 0.001):
        self.framework = framework
        self.backend = backend
        self.sleep = sleep

        # make sure torch is initialized, if possible, before doing any measurements
        get_torch_memory_gb()

        self.initial_memory = self.measure_fn()

    def measure_fn(self):
        if FLAG_ON_MAC:
            # Unified memory: we just care about free vs used RAM. The problem
            # is that tensors may live in GPU memory via MPS or Metal, but auxiliary
            # info (e.g. KV cache) does not -- we then need to know RAM and GPU mem.
            # Instead, we use psutil as an approximation of the total used memory.
            ram = get_process_memory_gb()

            if self.framework == "torch" and self.backend == "mps":
                vram = get_torch_memory_gb()
            elif self.framework == "mlx" and self.backend == "metal":
                vram = get_mlx_memory_gb()
            else:
                vram = 0.0

            return ram + vram

        elif FLAG_ON_LINUX:
            if self.framework == "torch" and self.backend == "cuda":
                # only measure (dedicated) GPU VRAM - this is what's important
                return get_torch_memory_gb
            else:
                # fall back on RAM
                return get_process_memory_gb()

    def get_used_memory(self):
        return self.measure_fn() - self.initial_memory


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


def get_available_ram_gb() -> float:
    """Return the available RAM in GB."""
    ram = psutil.virtual_memory().available
    return bytes_to_gb(ram)


def get_used_ram_gb() -> float:
    """Return the used RAM in GB."""
    ram = psutil.virtual_memory().used
    return bytes_to_gb(ram)


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
