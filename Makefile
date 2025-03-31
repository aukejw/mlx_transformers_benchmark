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
	python scripts/run_benchmarks.py
