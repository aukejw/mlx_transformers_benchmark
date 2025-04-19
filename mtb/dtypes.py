import mlx.core as mx
import torch


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    dtype = dict(
        float32=torch.float32,
        bfloat16=torch.bfloat16,
        float16=torch.float16,
        int8=torch.int8,
        int6="torch.int6",  # doesn't exist
        int4="torch.int4",  # doesn't exist
    )[dtype_str]
    return dtype


def get_mlx_dtype(dtype_str) -> mx.Dtype:
    dtype = dict(
        float32=mx.float32,
        bfloat16=mx.bfloat16,
        float16=mx.float16,
        int8=mx.int8,
        int6="mx.int6",  # doesn't exist
        int4="mx.int4",  # doesn't exist
    )[dtype_str]
    return dtype
