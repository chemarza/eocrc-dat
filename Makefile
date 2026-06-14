.PHONY: help install dev lint type test smoke docker clean

help:
	@echo "targets: install dev lint type test smoke docker clean"

install:
	pip install -e .

dev:
	pip install -e ".[dev,tuning]"

lint:
	ruff check .
	isort --check-only .
	black --check .

type:
	mypy eocrc_dat

test:
	pytest -q

smoke:
	eocrc-dat train --experiment _smoke

docker:
	docker build -t eocrc-dat .

clean:
	rm -rf build dist *.egg-info .mypy_cache .ruff_cache .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
