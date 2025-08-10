# Multi-stage Dockerfile for Ear Segmentation AI

# Base image
FROM python:3.11-slim AS base

# Builder stage: install dependencies and package
FROM base AS builder

# Install build tools
RUN apt-get update && apt-get install --no-install-recommends -y build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install poetry to manage dependencies
COPY pyproject.toml poetry.lock ./
RUN pip install --no-cache-dir poetry && \
    poetry export -f requirements.txt --output requirements.txt --without-hashes && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# Install the package itself
COPY . .
RUN pip install --no-cache-dir --prefix=/install .

# Final runtime image
FROM base AS runtime
WORKDIR /app

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy example code
COPY examples ./examples

# Default command runs the basic example script
CMD ["python", "examples/basic/basic_usage.py"]
