import mlx.core as mx
import mlx.nn
import numpy as np
import pytest
import torch
from lmstudio import LLM
from mlx_lm.tokenizer_utils import TokenizerWrapper
from transformers import GemmaTokenizerFast
from transformers.models.gemma3.modeling_gemma3 import Gemma3PreTrainedModel

from mtb import FLAG_ON_MAC
from mtb.llm_benchmarks.models.gemma import Gemma3_1B_it
from mtb.llm_benchmarks.run_llm_benchmark import create_benchmark
from mtb.lmstudio_utils import check_lms_server_running


@pytest.fixture(scope="session")
def benchmark_torch():
    benchmark = create_benchmark(
        model_spec=Gemma3_1B_it,
        framework="torch",
        backend="mps",
        dtype="bfloat16",
        max_num_tokens=10,
    )
    benchmark.setup()
    yield benchmark
    benchmark.teardown()


@pytest.fixture(scope="session")
def benchmark_mlx():
    benchmark = create_benchmark(
        model_spec=Gemma3_1B_it,
        framework="mlx",
        backend="metal",
        dtype="bfloat16",
        max_num_tokens=10,
    )
    benchmark.setup()
    yield benchmark
    benchmark.teardown()


@pytest.fixture(scope="session")
def benchmark_lms():
    benchmark = create_benchmark(
        model_spec=Gemma3_1B_it,
        framework="lmstudio",
        backend="metal+llama.cpp",
        dtype="int4",
        max_num_tokens=10,
    )
    benchmark.setup()
    yield benchmark
    benchmark.teardown()


expected_bf16_response = "Okay, here’s a story about Albert Einstein"
expected_int4_response = "Okay, here's a story about Albert"


@pytest.mark.skipif(not FLAG_ON_MAC, reason="Must run on Mac")
@pytest.mark.skipif(not torch.mps.is_available(), reason="Must run on MPS backend")
def test_gemma_torch(benchmark_torch):
    torch.manual_seed(0)
    assert isinstance(benchmark_torch.model, Gemma3PreTrainedModel)
    assert isinstance(benchmark_torch.tokenizer, GemmaTokenizerFast)

    prompt_tokens = benchmark_torch.format_prompt("OK")
    assert isinstance(prompt_tokens, torch.Tensor)

    timing = benchmark_torch.run_once(prompt="Write a story about Einstein")
    assert timing.prompt_tps > 0
    assert timing.prompt_time_sec > 0
    assert timing.generation_tps > 0

    response = timing.response
    assert response == expected_bf16_response


@pytest.mark.skipif(not FLAG_ON_MAC, reason="Must run on Mac")
def test_gemma_mlx(benchmark_mlx):
    mx.random.seed(0)
    assert isinstance(benchmark_mlx.model, mlx.nn.Module)
    assert isinstance(benchmark_mlx.tokenizer, TokenizerWrapper)

    prompt_tokens = benchmark_mlx.format_prompt("OK")
    assert isinstance(prompt_tokens, mx.array)

    timing = benchmark_mlx.run_once(prompt="Write a story about Einstein")
    assert timing.prompt_tps > 0
    assert timing.prompt_time_sec > 0
    assert timing.generation_tps > 0

    response = timing.response
    assert response == expected_bf16_response


@pytest.mark.skipif(not FLAG_ON_MAC, reason="Must run on Mac")
@pytest.mark.skipif(not check_lms_server_running(), reason="Must run on LLM Studio")
def test_gemma_lmstudio(benchmark_lms):
    assert isinstance(benchmark_lms.model, LLM)

    prompt_tokens = benchmark_lms.format_prompt("OK")
    assert isinstance(prompt_tokens, np.ndarray)

    timing = benchmark_lms.run_once(prompt="Write a story about Einstein")
    assert timing.prompt_tps > 0
    assert timing.prompt_time_sec > 0
    assert timing.generation_tps > 0

    response = timing.response
    assert response == expected_int4_response
