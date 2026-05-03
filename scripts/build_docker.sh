#!/usr/bin/env bash
set -euo pipefail

docker build --platform linux/amd64 \
  -t quick-ai-x86:ubuntu24.04 \
  -f docker/x86/Dockerfile .
