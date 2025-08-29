# Docker Usage for MedAISure

This guide shows how to build and run MedAISure in Docker for both CPU and GPU environments.

## Images

- CPU: `Dockerfile` (Python 3.10 slim)
- GPU: `Dockerfile.gpu` (NVIDIA CUDA 11.8 runtime + cuDNN)

Both images install Python deps from `requirements.txt` and use the CLI entrypoint `medaisure-benchmark` defined in `setup.py`.

## Build

```bash
# CPU image
docker build -t medaisure/cpu:latest -f Dockerfile .

# GPU image (requires NVIDIA Docker runtime)
docker build -t medaisure/gpu:latest -f Dockerfile.gpu .
```

## Run (Docker)

```bash
# List tasks (CPU)
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/results:/app/results" medaisure/cpu:latest list --json

# List tasks (GPU)
docker run --rm --gpus all -v "$PWD/data:/app/data" -v "$PWD/results:/app/results" medaisure/gpu:latest list --json

# Show a task (replace <id>)
docker run --rm medaisure/cpu:latest show <id>
```

## Run (Docker Compose)

```bash
# Build and run CPU service
docker compose up --build medaisure-cpu

# Build and run GPU service (requires NVIDIA runtime)
docker compose up --build medaisure-gpu
```

Volumes:
- `./data` is mounted to `/app/data`
- `./results` is mounted to `/app/results`

## Environment Variables

Create a local `.env` for API keys or configs and mount it if needed:

```bash
docker run --rm --env-file .env medaisure/cpu:latest list --json
```

Alternatively, pass specific vars:

```bash
docker run --rm -e HF_TOKEN=... medaisure/cpu:latest list --json
```

## Notes

- The entrypoint uses the console script `medaisure-benchmark` (from `bench/cli.py`). If you prefer a Python module entrypoint, you can run:
  ```bash
  docker run --rm medaisure/cpu:latest python -m bench.cli list --json
  ```
- GPU runs require a host with NVIDIA drivers and the NVIDIA Container Toolkit installed. Test with `docker run --rm --gpus all nvidia/cuda:11.8.0-base nvidia-smi`.
