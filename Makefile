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
	poetry run python scripts/run_llm_benchmarks.py --num_iterations 3 --dtype bfloat16

show-layer-benchmarks:
	poetry run python scripts/visualize_layer_benchmarks.py --show_all_measurements
	open visualizations/index.html 

show-llm-benchmarks:
	poetry run python scripts/visualize_llm_benchmarks.py --show_all_measurements
	open visualizations/index.html 

test:
	poetry run pytest --cov --cov-report=term-missing --cov-report=html --disable-warnings -v
