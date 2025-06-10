.PHONY: help install install-dev test test-cov lint format clean build docs

# Default target
help:
	@echo "Ear Segmentation AI - Development Commands"
	@echo ""
	@echo "Installation:"
	@echo "  make install       Install package in production mode"
	@echo "  make install-dev   Install package in development mode"
	@echo ""
	@echo "Testing:"
	@echo "  make test          Run all tests"
	@echo "  make test-cov      Run tests with coverage report"
	@echo "  make test-unit     Run unit tests only"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint          Run linting (ruff, mypy, bandit)"
	@echo "  make format        Format code (black, isort)"
	@echo "  make pre-commit    Run pre-commit hooks"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs          Build documentation"
	@echo "  make docs-serve    Serve documentation locally"
	@echo ""
	@echo "Build & Release:"
	@echo "  make build         Build package"
	@echo "  make clean         Clean build artifacts"
	@echo ""
	@echo "Development:"
	@echo "  make benchmark     Run performance benchmark"
	@echo "  make download-model Download the model"

# Installation
install:
	poetry install --only main

install-dev:
	poetry install
	pre-commit install

# Testing
test:
	poetry run pytest tests/

test-cov:
	poetry run pytest tests/ --cov=src/earsegmentationai --cov-report=html --cov-report=term

test-unit:
	poetry run pytest tests/unit/

test-integration:
	poetry run pytest tests/integration/

# Code Quality
lint:
	poetry run ruff check src/ tests/
	poetry run mypy src/
	poetry run bandit -r src/ -c pyproject.toml

format:
	poetry run black src/ tests/
	poetry run isort src/ tests/
	poetry run ruff check src/ tests/ --fix

pre-commit:
	poetry run pre-commit run --all-files

# Documentation
docs:
	@echo "Building documentation..."
	# Add your documentation build command here
	# e.g., poetry run mkdocs build

docs-serve:
	@echo "Serving documentation..."
	# Add your documentation serve command here
	# e.g., poetry run mkdocs serve

# Build & Release
build: clean
	poetry build

clean:
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Development
benchmark:
	@echo "Running benchmark on example image..."
	poetry run earsegmentationai benchmark examples/0210.png --iterations 50

download-model:
	poetry run earsegmentationai download-model

# Quick development cycle
dev: format lint test

# CI/CD simulation
ci: install-dev lint test-cov