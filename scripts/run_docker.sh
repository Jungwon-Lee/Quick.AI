#!/usr/bin/env bash
set -euo pipefail

docker run --rm -it --platform linux/amd64 \
  -v "$PWD:/workspace/Quick.AI" \
  quick-ai-x86-test:ubuntu24.04
