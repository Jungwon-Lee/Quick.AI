#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

QUANT_BIN="build/quick_dot_ai_quantize_stream"

build_quantizer() {
  if [[ ! -f build/build.ninja ]]; then
    echo "Build directory is not configured. Run: meson setup build" >&2
    exit 1
  fi

  ninja -C build quick_dot_ai_quantize_stream
}

if [[ $# -lt 2 ]]; then
  if [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
    build_quantizer
    "${QUANT_BIN}" --help
    exit 0
  fi
  echo "Usage: $0 <model_dir> <output_dir> [output_bin] [options]" >&2
  echo "Options are passed to quantize_stream, e.g. --dtype Q6_K" >&2
  exit 1
fi

build_quantizer
"${QUANT_BIN}" "$@"
