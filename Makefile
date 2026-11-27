.PHONY: help lint test format rfix
.DEFAULT_GOAL := help

UV := uv run
path ?= .

help:
	@echo Available commands:
	@echo "  make lint      - Check code style with ruff"
	@echo "  make test      - Run unit tests with pytest"

lint:
	$(UV) ruff check .

test:
	$(UV) pytest

format:
	$(UV) ruff format $(path)

rfix:
	$(UV) ruff check --fix
