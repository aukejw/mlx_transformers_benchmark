## Targets for virtual environments

# Sets up a virtual environment and activates it
create-venv:
	pyenv local 3.11.11
	POETRY_VIRTUALENVS_IN_PROJECT=1 poetry env use $(shell pyenv which python3)
	poetry lock
	poetry install

# Activate the virtual environment
activate-venv:
	@echo "Run to activate the virtual environment: "
	@echo "source $(shell poetry env info --path)/bin/activate"

## Targets for running benchmarks

run:
	python scripts/run_benchmarks.py --num_iterations 30 --num_warmup_iterations 10 --dtype float32
	python scripts/run_benchmarks.py --num_iterations 30 --num_warmup_iterations 10 --dtype float16
	python scripts/run_benchmarks.py --num_iterations 30 --num_warmup_iterations 10 --dtype bfloat16

show:
	python scripts/visualize_measurements.py
	open benchmark_visualizations/index.html 

test:
	pytest --cov --cov-report=term-missing --cov-report=html --disable-warnings -v tests
