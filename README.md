# Benchmarking transformer operators on Apple silicon

[![tests](https://github.com/aukejw/mlx_transformers_benchmark/actions/workflows/tests.yaml/badge.svg)](https://github.com/aukejw/mlx_transformers_benchmark/actions/workflows/tests.yaml)

Let's say you're interested in performing LLM inference on Apple hardware. You care about speed, but don't know which model or framwork to pick.

Do you:
- use [PyTorch with the Metal Performance Shaders backend](https://pytorch.org/docs/stable/notes/mps.html)
- use Apple's [MLX, built directly for Metal](https://github.com/ml-explore/mlx)?
- use `LMStudio` with the `Metal+llama.cpp` backend?

We aim to help make this choice by benchmarking inference for a few common models and operators. Results can be found at 
[https://aukejw.github.io/mlx_transformers_benchmark/](https://aukejw.github.io/mlx_transformers_benchmark/).


### Dependencies

Before you start, you will need:
 - [`pyenv`](https://github.com/pyenv/pyenv) to manage the python version
 - [`poetry`](https://python-poetry.org/) for dependency management

 To (optionally) benchmark `Metal+llama.cpp` models, you will also need:
 - [`lmstudio`](https://lmstudio.ai/) 

### Quickstart

1. Clone the repo:
   ```
   git clone git@github.com:aukejw/mlx_transformers_benchmark.git
   cd mlx_transformers_benchmark
   ```

2. Set up a python3.11 virtual environment using 
   [`pyenv`](https://github.com/pyenv/pyenv) and 
   [`poetry`](https://python-poetry.org/), and activate it:
   ```
   make create-venv
   source .venv/bin/activate
   ```

3. For good measure, run the tests. This also tells you whether we can use the GPU.
   ```
   make test
   ```

3. Run benchmarking, here for the 0.5B parameter `Qwen2.5` model:
   ```
   python scripts/run_llm_benchmarks.py \
      --run_only_benchmarks qwen-2.5-0.5b-it \
      --dtypes \["int4","int8"\] \
      --num_iterations 3 \
   ```
   This creates a new result in the `measurements` folder.

   Optionally, to run a full benchmark for `bfloat16`, `int8`, `int4` datatypes, you can use:
   ``` 
   make run-llm-benchmarks
   ```
   This will take a longer time however, so make sure you aren't busy!

4. To create a HTML report of all measurements and open the index page:
   ```
   make show-llm-benchmarks
   ```
   This should open a page similar to 
   [https://aukejw.github.io/mlx_transformers_benchmark/](https://aukejw.github.io/mlx_transformers_benchmark/).


### Contributing

If you have an Apple device, additional measurements are always welcome! The easiest way to contribute is to [fork the repo]( https://github.com/aukejw/mlx_transformers_benchmark/fork), and run benchmarks for common LLMs and/or operators. 

See [CONTRIBUTING.md](https://github.com/aukejw/mlx_transformers_benchmark/blob/main/contributing.md) for the instructions.


### On reproducibility

As Apple machines share memory with other background processes, these benchmarks are not exact, certainly not for Macbooks. Still, the numbers should give a decent idea of the performance to expect. 

Although the default parameters do not result in thermal throttling for a Macbook M4 Pro, older
machines may have trouble with the heavier models and operators. We do try to skip large models,
but you may still have too little RAM and fall back on swap space. If you see huge memory pressure 
or outlier measurements, do take a closer look!

> [!NOTE] 
> For a large number of iterations, the GPU will certainly heat up. If needed, you can 
increase the cooldown period using the `cooldown_time_fraction` argument. Monitoring GPU 
temperature programatically requires admin privileges, but you can use third-party apps like 
[stats](https://github.com/exelban/stats), also available as [homebrew](https://formulae.brew.sh/cask/stats).


### Notes

Apple silicon is fairly cost-effective for LLM inference due to its unified memory architecture.
As LLM inference is mostly memory-bound for low batch sizes, devices with high memory bandwidth 
typically obtain 
[high tokens/sec in inference benchmarks](https://github.com/ggml-org/llama.cpp/discussions/4167).

This benchmark focuses on the inference time of easy-to-run LLMs and unquantized transformer ops, primarily 
useful when finetuning custom models for (or on!) Apple devices. While speed is one factor,
do consider ecosystem and cross-platform support too - here, Nvidia+CUDA remain hard to beat!  

You may also be interested in:

- Tristan Bilot's comprehensive benchmark for fundamental operators for `mlx`, 
  `torch+mps`, and `torch+cuda` ([link](https://github.com/TristanBilot/mlx-benchmark)). Placing both `mlx` 
  and `torch` functions in a single benchmark class makes it easy to see the differences between the 
  two, and we adopt the same strategy here.

- [The work of Feng et al.](https://arxiv.org/pdf/2501.14925) comparing training on Nvidia cards vs Apple Silicon. 
