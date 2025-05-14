## Targets for virtual environments

# Sets up a virtual environment and activates it
setup:
	uv python install 3.11.11
	uv sync --group=dev

# Activate the virtual environment
activate-venv:
	@echo "Run to activate the virtual environment: "
	@echo "source .venv/bin/activate"
	@echo "You can also use `uv run` to run scripts in the virtual environment without activating it."

## Targets for running benchmarks

run-llm-benchmarks:
	uv run python scripts/run_llm_benchmarks.py --num_iterations 3

run-layer-benchmarks:
	uv run python scripts/run_layer_benchmarks.py --num_iterations 30

show-llm-benchmarks:
	uv run python scripts/visualize_llm_benchmarks.py --show_all_measurements
	open visualizations/index.html 

show-layer-benchmarks:
	uv run python scripts/visualize_layer_benchmarks.py --show_all_measurements
	open visualizations/index.html 

test:
	uv run pytest --cov --cov-report=term-missing --cov-report=html --disable-warnings -v
