# Stage 1: The Builder (High-speed dependency resolution)
FROM ghcr.io/astral-sh/uv:latest AS builder

# Stage 2: Final Runtime (Blackwell sm_100 Optimized)
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# Avoid stuck prompts
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# 1. Install System dependencies (OpenCV, DVC, Git)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev git ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Get uv
COPY --from=builder /uv /uvx /bin/

# 3. Configure uv (Global settings)
ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# 4. Install Dependencies
# We copy ONLY the lockfiles first to cache this layer
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev

# Add the virtual environment to the System PATH
# This ensures 'dvc', 'python', and 'yolo' work globally
# -----------------------------------------------------------
ENV PATH="/app/.venv/bin:$PATH"

# 5. Copy DVC Configuration (CRITICAL STEP)
# We must copy the .dvc folder so DVC knows what 'storage' is
COPY .dvc/ ./.dvc/
COPY dvc.yaml params.yaml README.md ./

# 6. DVC Authentication (Baked-in Credentials)
# Copy the credentials file to a safe location
COPY .dvc/tmp/gdrive-user-credentials.json /app/.dvc/tmp/gdrive-user-credentials.json

# Tell DVC to use this specific file
# Note: This modifies .dvc/config.local inside the image
RUN dvc remote modify storage --local gdrive_user_credentials_file /app/.dvc/tmp/gdrive-user-credentials.json

# 7. Copy Source Code
COPY configs/ ./configs/
COPY src/ ./src/
COPY deployment/ ./deployment/

# 8. Create data folders
RUN mkdir -p data/raw data/processed

# Default to shell for flexibility
CMD ["/bin/bash"]