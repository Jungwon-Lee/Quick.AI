FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    clang-format-14 \
    cmake \
    flatbuffers-compiler \
    git \
    libflatbuffers-dev \
    libiniparser-dev \
    libomp-dev \
    libopenblas-dev \
    pkg-config \
    python3 \
    python3-pip \
    python3-venv \
    wget \
  && python3 -m pip install --break-system-packages --no-cache-dir meson ninja \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

ENV OMP_NUM_THREADS=4 \
    OMP_WAIT_POLICY=active \
    OMP_PROC_BIND=true \
    OMP_PLACES=cores

WORKDIR /workspace/Quick.AI

CMD ["/bin/bash"]
