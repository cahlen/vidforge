#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN=${PYTHON_BIN:-python3.11}
USER_SET_CUDA_TAG=0
if [[ -n "${CUDA_VERSION_TAG+x}" ]]; then
  USER_SET_CUDA_TAG=1
fi
CUDA_VERSION_TAG=${CUDA_VERSION_TAG:-cu124}
TORCH_VERSION=${TORCH_VERSION:-2.5.1}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.20.1}
TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION:-2.5.1}
TORCH_CHANNEL=${TORCH_CHANNEL:-stable}

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

DETECTED_CC=""
if command -v nvidia-smi >/dev/null 2>&1; then
  DETECTED_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '[:space:]') || true
fi

if [[ -n "${DETECTED_CC}" ]]; then
  printf 'Detected GPU compute capability: %s\n' "${DETECTED_CC}"
  SM_TOKEN=${DETECTED_CC//./}
  case "${SM_TOKEN}" in
    50|52|53|60|61|62|70|72|75|80|86|87|89|90)
      : # supported by stable wheels
      ;;
    *)
      if [[ "${TORCH_CHANNEL}" == "stable" ]]; then
        echo "Compute capability ${DETECTED_CC} is newer than standard stable wheels. Switching to PyTorch nightly build." >&2
        TORCH_CHANNEL=nightly
        if [[ ${USER_SET_CUDA_TAG} -eq 0 ]]; then
          CUDA_VERSION_TAG=cu128
        fi
      fi
      ;;
  esac
fi

TORCH_INDEX_BASE="https://download.pytorch.org/whl"
if [[ "${TORCH_CHANNEL}" == "nightly" ]]; then
  TORCH_INDEX="${TORCH_INDEX_BASE}/nightly/${CUDA_VERSION_TAG}"
  echo "Installing PyTorch nightly from ${TORCH_INDEX}." >&2
  pip install --upgrade --force-reinstall --pre torch torchvision torchaudio --index-url "${TORCH_INDEX}"
  if pip show xformers >/dev/null 2>&1; then
    echo "Removing preinstalled xformers to avoid ABI mismatches; reinstall a matching build manually if needed." >&2
    pip uninstall -y xformers >/dev/null 2>&1 || true
  fi
else
  TORCH_INDEX="${TORCH_INDEX_BASE}/${CUDA_VERSION_TAG}"
  pip install \
    --upgrade \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url "${TORCH_INDEX}"
fi

pip install -e .[dev]

# Optional extras (install only if you have compatible ONNX builds)
# TODO: add concrete installation commands for ESRGAN and RIFE once binary wheels are confirmed.

python - <<'PYCODE'
import torch

def _gpu_probe() -> None:
    if not torch.cuda.is_available():
        print("CUDA not detected. Check your NVIDIA drivers.")
        return
    try:
        device = torch.device("cuda")
        x = torch.rand(1, 3, 4, 4, device=device)
        y = torch.nn.functional.softmax(x, dim=1)
        print("CUDA OK - tensor norm:", y.norm().item())
    except Exception as exc:  # capture mismatch between wheel and GPU arch
        print(f"Warning: CUDA probe failed: {exc}")
        print("Torch installed, but GPU kernel support may require a newer build.")

_gpu_probe()
PYCODE

echo "Environment ready in ${VENV_DIR}"
