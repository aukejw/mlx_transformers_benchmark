## Targets for virtual environments

# Sets up a virtual environment and activates it
create-venv:
	pyenv local 3.11.11
	POETRY_VIRTUALENVS_IN_PROJECT=1 poetry env use $(shell pyenv which python3)
	poetry lock
	poetry install --with dev

# Activate the virtual environment
activate-venv:
	@echo "Run to activate the virtual environment: "
	@echo "source $(shell poetry env info --path)/bin/activate"

## Targets for running benchmarks

run:
	poetry run python scripts/run_benchmarks.py --num_iterations 20 --num_warmup_iterations 10 --dtype float32
	poetry run python scripts/run_benchmarks.py --num_iterations 20 --num_warmup_iterations 10 --dtype float16
	poetry run python scripts/run_benchmarks.py --num_iterations 20 --num_warmup_iterations 10 --dtype bfloat16

show-layer-benchmarks:
	poetry run python scripts/visualize_layer_benchmarks.py --show_all_measurements
	open visualizations/layer_benchmarks/index.html 

show-llm-benchmarks:
	poetry run python scripts/visualize_llm_benchmarks.py --show_all_measurements
	open visualizations/llm_benchmarks/index.html 

test:
	poetry run pytest --cov --cov-report=term-missing --cov-report=html --disable-warnings -v
