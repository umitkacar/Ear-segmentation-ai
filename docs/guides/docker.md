# Docker Usage Guide

This guide explains how to build and run the Ear Segmentation AI project using Docker.

## Build the image

```bash
docker build -t ear-segmentation-ai .
```

This command uses the multi-stage `Dockerfile` in the repository root to install all dependencies and prepare the example application.

## Run the container

Run the example application directly with Docker:

```bash
docker run --rm ear-segmentation-ai
```

The container executes the `examples/basic/basic_usage.py` script by default.

## Using docker-compose

For local development you can use **docker-compose**:

```bash
docker compose up
```

This uses `docker-compose.yml` which mounts the project directory and exposes port `8000` (adjust as needed).

Stop the services with `Ctrl+C` and remove containers with:

```bash
docker compose down
```
