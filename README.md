# Benchmarking transformer operators on Apple silicon

Let's say you're interested in performing inference for small (unquantized) transformers on Apple hardware. 
You care about speed, but don't want to fry your machine.

This means you need to make an important choice: use 
[PyTorch with the Metal Performance Shaders backend](https://pytorch.org/docs/stable/notes/mps.html),
or move to Apple's
[MLX, built directly for Metal](https://github.com/ml-explore/mlx)?

We aim to answer this question here by benchmarking inference for a few common operations and layers.

> [!NOTE] 
> Although the default parameters do not result in thermal throttling for a Macbook M4 Pro, older
machines may have trouble with the heavier operators, or may have too little RAM and fall back on 
swap space. For a large number of iterations, the GPU will certainly heat up. If needed, you can 
increase the cooldown period using the `cooldown_time_fraction` argument.  


### Dependencies

You will need:
 - [`pyenv`](https://github.com/pyenv/pyenv) to manage python versions
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

3. Run benchmarking. By default, we will test multiple batch sizes and sequence lengths for each operator. 
   ```
   python scripts/run_benchmarks.py \
      --num_warmup_iterations 10 \
      --num_iterations 30 \
      --dtype float16 \
      --run_torch_mps=True \
      --run_mlx_metal=True \
      --run_mlx_metal_compiled=True  \
      --cooldown_time_fraction=0.1 
   ```

   To run a full benchmark on GPU for `float32`, `float16`, `bfloat16` datatypes, you can also use:
   ``` 
   make run
   ```

4. Visualize measurements and open the index page:
   ```
   make show
   ```

### Notes

This benchmark focuses on runtime. Monitoring GPU temperature is interesting too, but typically 
requires admin privileges. For manual monitoring, you can use third-party apps like 
[stats](https://github.com/exelban/stats), also available as [homebrew](https://formulae.brew.sh/cask/stats).

You may also be interested in Tristan Bilot's comprehensive benchmark for fundamental operators for `mlx`, 
`torch+mps`, and `torch+cuda` ([link](https://github.com/TristanBilot/mlx-benchmark)). Placing both `mlx` 
and `torch` functions in a single benchmark class makes it easy to see the differences between the 
two, and we adopt the same strategy here.
  

### Contributing

PRs welcome! Feel free to add measurement files or new benchmark tasks.