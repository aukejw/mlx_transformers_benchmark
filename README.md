# Benchmarking ML on Apple silicon

## Dependencies

First, you will need:
 - [`pyenv`](https://github.com/pyenv/pyenv) to manage python versions
 - [`poetry`](https://python-poetry.org/) for dependency management

These are available in homebrew:
```
brew install pyenv poetry
```

## Quickstart

1. Clone the repo:
   ```
   git clone ..
   ```

2. Set up a python3.11 virtual environment using 
   [`pyenv`](https://github.com/pyenv/pyenv) and 
   [`poetry`](https://python-poetry.org/), and activate it:

   ```
   make create-venv
   make activate-venv
   ```

3. Run benchmarking:
   ``` 
   make run
   ```
