## Targets for virtual environments

# Create a virtual environment using uv
setup:
	uv sync --group=dev

# Activate the virtual environment
activate-venv:
	@echo "We recommend using `uv run python <your-script>` to use the venv without activating it."
	@echo "If you insist, to activate the virtual environment, run: "
	@echo "source .venv/bin/activate"

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
