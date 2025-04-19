# Benchmarking transformer operators on Apple silicon

Let's say you're interested in performing inference for small (unquantized) transformers on 
Apple hardware. You do care about speed, but aren't ready to quantize just yet, and you do 
not want to fry your machine.

This means you need to make an important choice: 

- use [PyTorch with the Metal Performance Shaders backend](https://pytorch.org/docs/stable/notes/mps.html),
- or move to Apple's [MLX, built directly for Metal](https://github.com/ml-explore/mlx)?

We aim to help make this choice by benchmarking inference for a few common operations and layers. 
Results can be found at 
[https://aukejw.github.io/mlx_transformers_benchmark/](https://aukejw.github.io/mlx_transformers_benchmark/).


### Dependencies

Before you start, you will need:
 - [`pyenv`](https://github.com/pyenv/pyenv) to manage the python version
 - [`poetry`](https://python-poetry.org/) for dependency management

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

3. Run benchmarking. By default, we will test multiple batch sizes and sequence lengths for each specified operator. 
   ```
   python scripts/run_llm_benchmarks.py \
      --num_warmup_iterations 1 \
      --num_iterations 3 \
      --dtype bfloat16 
   ```
   This creates a new result in the `measurements` folder.

   Optionally, to run a full benchmark on GPU for `float32`, `float16`, `bfloat16` datatypes, you can also use:
   ``` 
   make run
   ```

4. To create a HTML report of all measurements and open the index page:
   ```
   make show
   ```

   This should open a page similar to 
   [https://aukejw.github.io/mlx_transformers_benchmark/](https://aukejw.github.io/mlx_transformers_benchmark/).


### On reproducibility

Apple's virtualization framework does not provide a GPU API for virtualized envionments, 
and using Metal from a Docker container is not supported yet. This makes exact reproducibility 
challenging, but the numbers should give a decent idea nevertheless. 

Although the default parameters do not result in thermal throttling for a Macbook M4 Pro, older
machines may have trouble with the heavier operators, or may have too little RAM and fall back on 
swap space. If you see huge outliers, do take a closer look!

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

This benchmark focuses on the runtime of unquantized transformer ops, primarily useful 
when training small, custom models for (or on!) Apple devices. While speed is one factor,
do consider ecosystem and cross-platform support too - here, Nvidia+CUDA remain hard to beat!  

You may also be interested in:
- Tristan Bilot's comprehensive benchmark for fundamental operators for `mlx`, 
  `torch+mps`, and `torch+cuda` ([link](https://github.com/TristanBilot/mlx-benchmark)). Placing both `mlx` 
  and `torch` functions in a single benchmark class makes it easy to see the differences between the 
  two, and we adopt the same strategy here.

- [The work of Feng et al](https://arxiv.org/pdf/2501.14925) comparing training on Nvidia cards vs Apple Silicon. 


### Contributing

If you have an Apple device, additional measurements are always welcome! 

The easiest way to contribute is to set up the repo as described, and run benchmarks for three dtypes:
```
make run
```
Running benchmarks for all operators should take around 20 minutes per dtype.

This creates a new measurement folder for each dtype, and stores the platform info as well as the  
`mlx` and `torch` versions in a settings file. Pull requests welcome!
