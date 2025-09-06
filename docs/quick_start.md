# Quick Start

This guide helps you get up and running with MedAISure quickly using either local Python or Docker (CPU/GPU).

## Prerequisites

- Python 3.8+
- Docker (optional but recommended for reproducibility)
- NVIDIA drivers + NVIDIA Container Toolkit (optional, for GPU)

## 1) Local Installation

```bash
# Clone
git clone https://github.com/junaidi-ai/MedAISure.git
cd MedAISure

# Create and activate venv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install package + dev extras
pip install -e .[dev]

# (Optional) set env vars
cp .env.example .env
# edit .env as needed (API keys, etc.)
```

List available tasks via CLI:

```bash
medaisure-benchmark list --json
```

For curated dataset registry entries (e.g., medaisure-core and planned sets) and how to list/inspect them via CLI/Python, see the [Datasets](datasets/overview.md) page.

## 2) Docker (CPU)

```bash
# Build CPU image
docker build -t medaisure/cpu:latest -f Dockerfile .

# List tasks
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/results:/app/results" \
  medaisure/cpu:latest list --json
```

## 3) Docker (GPU)

```bash
# Build GPU image (requires NVIDIA Container Toolkit)
docker build -t medaisure/gpu:latest -f Dockerfile.gpu .

# Quick GPU smoke test (sanity check)
docker run --rm --gpus all medaisure/gpu:latest \
  python3 scripts/gpu_smoke.py

# List tasks with GPU image
docker run --rm --gpus all -v "$PWD/data:/app/data" -v "$PWD/results:/app/results" \
  medaisure/gpu:latest list --json
```

## 4) Docker Compose

```bash
# CPU service
docker compose up --build medaisure-cpu

# GPU service (requires NVIDIA runtime)
docker compose up --build medaisure-gpu

# GPU smoke-only (profile)
docker compose --profile smoke up --build medaisure-gpu-smoke
```

## CPU vs GPU Differences

- GPU image installs CUDA 11.8-enabled PyTorch wheels. Use `--gpus all` when running.
- CPU image uses generic wheels; it will run on systems without NVIDIA GPUs.
- Performance: GPU path accelerates tensor ops; see `scripts/gpu_smoke.py` for a quick timing check.

## Volumes & Data Best Practices

- Mount `./data` to `/app/data` for inputs and `./results` to `/app/results` for outputs.
- Keep large datasets outside of the image; prefer volume mounts.
- For reproducibility, keep a copy of configs under version control.

## Environment Variables

- Copy `.env.example` to `.env` and set any required keys.
- With Docker, pass `--env-file .env` or `-e KEY=value` for specific variables.

## Troubleshooting

- See [docs/troubleshooting.md](./troubleshooting.md)
- Verify Docker GPU support: `docker run --rm --gpus all nvidia/cuda:11.8.0-base nvidia-smi`
- If GPU is not available, use the CPU image/commands.
