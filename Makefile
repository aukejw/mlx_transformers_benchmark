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

show:
	poetry run python scripts/visualize_measurements.py --show_all_measurements
	open benchmark_visualizations/index.html 

test:
	poetry run pytest --cov --cov-report=term-missing --cov-report=html --disable-warnings -v
