#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV_DIR="${ROOT_DIR}/.venv"
OUT_DIR="${ROOT_DIR}/out/benchmark"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

START=$(date +%s)
python -m vidforge.cli render \
  --plan "${ROOT_DIR}/examples/shots_example.json" \
  --out "${OUT_DIR}" \
  --model cogvideox \
  --fps 16 \
  --width 640 \
  --height 360 \
  --seconds-per-shot 4 \
  --rife \
  --esrgan \
  --seed 123
END=$(date +%s)
ELAPSED=$((END - START))

printf "Benchmark completed in %s seconds\n" "${ELAPSED}"
