#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV_DIR="${ROOT_DIR}/.venv"
OUT_DIR="${ROOT_DIR}/out/desert_patrol"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m vidforge.cli \
  --plan "${ROOT_DIR}/examples/shots_example.json" \
  --out "${OUT_DIR}" \
  --model cogvideox \
  --seed 7 \
  --fp16 \
  --rife \
  --esrgan \
  --vap-style "${ROOT_DIR}/assets/refs/style.jpg" || {
    echo "Render finished with warnings. Check logs for details." >&2
}
