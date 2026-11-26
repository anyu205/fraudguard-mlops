.PHONY: help lint test
.DEFAULT_GOAL := help

UV := uv run

help:
	@echo Available commands:
	@echo "  make lint      - Check code style with ruff"
	@echo "  make test      - Run unit tests with pytest"

lint:
	$(UV) ruff check .

test:
	$(UV) pytest
