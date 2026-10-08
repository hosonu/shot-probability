LOCAL_BIN   := ./bin
VENV        := ./.venv
UV          := $(LOCAL_BIN)/uv
UV_PY       := ./uv_python
UV_CACHE    := ./uv_cache

# Virtual environment
export UV_PYTHON_INSTALL_DIR := $(UV_PY)
export UV_CACHE_DIR := $(UV_CACHE)
export UV_PYTHON_DOWNLOADS := auto
export UV_PROJECT_ENVIRONMENT := $(VENV)

.PHONY: help install format lint check data train clean fclean

help:
	@echo "install  create the local uv, Python, and virtual environment"
	@echo "format   format Python files with ruff"
	@echo "lint     run ruff lint and format checks"
	@echo "check    run mypy and pytest"
	@echo "data     build data/shots.parquet from StatsBomb open data"
	@echo "train    fit the logistic baseline from data/shots.parquet"
	@echo "clean    remove Python caches and outputs/"
	@echo "fclean   clean, then remove .venv, bin, uv_python, uv_cache"

$(UV):
	mkdir -p $(LOCAL_BIN)
	curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$(LOCAL_BIN) UV_NO_MODIFY_PATH=1 sh

install: $(UV)
	$(UV) sync --locked

format: install
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

lint: install
	$(UV) run ruff check .
	$(UV) run ruff format --check .

check: install
	$(UV) run mypy src tests
	$(UV) run pytest

data: install
	$(UV) run python -m shot_probability.build_shots

train: install
	$(UV) run python -m shot_probability.train

clean:
	find . -type d -name __pycache__ -not -path "$(VENV)/*" -not -path "$(UV_PY)/*" -not -path "$(UV_CACHE)/*" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache outputs

fclean: clean
	rm -rf $(VENV) $(LOCAL_BIN) $(UV_PY) $(UV_CACHE)
