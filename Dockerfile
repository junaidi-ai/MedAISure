# Base CPU Dockerfile for MedAISure
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy source
COPY . .

# Install package to expose console_scripts (medaisure-benchmark)
RUN pip install .

# Default entrypoint uses the console script from setup.py
ENTRYPOINT ["medaisure-benchmark"]
# Default command prints tasks list; override with `docker run ... <cmd>` as needed
CMD ["list", "--json"]
