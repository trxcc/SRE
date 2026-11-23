# Build in an environment with network access to apt/PyPI/PyTorch wheels.
ARG BASE_IMAGE=nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-venv git build-essential ninja-build \
    cmake curl ca-certificates pkg-config libnuma-dev libaio-dev libsndfile1 \
    ffmpeg poppler-utils zstd ripgrep pandoc texlive-latex-base \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/deal
COPY . .
RUN bash scripts/install.sh

ENTRYPOINT ["bash", "scripts/launch.sh"]
CMD ["--config", "configs/local.env"]
