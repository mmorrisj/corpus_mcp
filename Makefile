VENV ?= .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.DEFAULT_GOAL := help

$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	$(PIP) install --quiet --upgrade pip
	$(PIP) install --quiet -e ".[dev]"

.PHONY: install
install: $(VENV)/bin/activate ## Install the server and dev tools

.PHONY: demo
demo: install ## Run a query against the bundled example corpus
	$(PY) -m corpus_mcp --root examples/corpus search "why does my coffee taste sour"

.PHONY: serve
serve: install ## Run the MCP server over stdio against the example corpus
	$(PY) -m corpus_mcp --root examples/corpus serve

.PHONY: test
test: install ## Run the test suite
	$(VENV)/bin/pytest -q

.PHONY: smoke
smoke: install ## Launch the installed server as a subprocess and exercise it
	$(PY) scripts/stdio_smoke.py examples/corpus

.PHONY: lint
lint: install ## Lint and format-check
	$(VENV)/bin/ruff check src tests scripts
	$(VENV)/bin/ruff format --check src tests scripts

.PHONY: clean
clean: ## Remove caches and build artifacts
	rm -rf $(VENV) .pytest_cache .ruff_cache build dist src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-10s\033[0m %s\n", $$1, $$2}'
