# VidForge

VidForge is a modular toolkit for stitching text-to-video (T2V) and image-to-video (I2V) shots into cohesive long-form sequences with optional style guidance, frame continuity, and post-processing.

## Quickstart

1. **Provision dependencies**
  ```bash
  ./scripts/setup_env.sh
  ```
  The setup script autodetects your GPU. Cards newer than the PyTorch stable matrix (for example RTX 5090, compute capability 12.x) trigger an automatic fallback to the PyTorch nightly `cu128` wheels and drop conflicting extras such as `xformers`. Set `CUDA_VERSION_TAG` explicitly (e.g. `CUDA_VERSION_TAG=cu121`) if you need a different build.
2. **Render the demo plan**
   ```bash
   ./scripts/render_example.sh
   ```
3. Find outputs under `out/desert_patrol/`.

> **Note:** The repository ships with lightweight placeholder generators for both CogVideoX and Hunyuan so the pipeline and tests run end-to-end without bulky checkpoints. Replace them with real model integrations before production use (see TODO markers in `vidforge/models/`).

## Features

- JSON-driven shot orchestration with per-shot overrides for fps, resolution, and carryover.
- Automatic last-frame carryover from T2V → I2V shots, plus previews of first/last frames for QA.
- Optional video-as-prompt adapter interface for consistent style or semantics across shots.
- Post-processing hooks for RIFE smoothing, ESRGAN upscaling, crossfades, and audio muxing.
- Rich CLI with Typer, structured logging, resume support, and VRAM detection hints.

## Model Setup

1. Download CogVideoX-5B and/or HunyuanVideo checkpoints.
2. Update the `TODO` markers inside `vidforge/models/cogvideox.py` and `vidforge/models/hunyuan.py` with local checkpoint paths or loading code.
3. Install any backend-specific dependencies (Diffusers, xFormers, FlashAttention, etc.).

## VRAM Guidance (RTX 5090 32 GB)

| Resolution | FPS | Shot Length | Expected VRAM |
| ---------- | --- | ----------- | ------------- |
| 480p       | 24  | 6s          | ~12 GB        |
| 720p       | 24  | 6s          | ~22 GB        |
| 1080p      | 24  | 4s          | ~30 GB (enable chunking) |

For stability:
- Reuse seeds for iterative refinements.
- Keep shots short (4–8 s) and focus on coverage rather than single long renders.
- Enable carryover on continuity-critical shots and provide strong textual descriptors.

## Example Shot Plan

See `examples/shots_example.json` for a ready-to-run plan.

## Testing & Benchmarking

- Smoke tests:
  ```bash
  source .venv/bin/activate
  pytest
  ```
  👷 Tip: The pipeline tries `ffmpeg` first and automatically falls back to OpenCV’s `VideoWriter` when the CLI binary or Python bindings are missing, so tests succeed even on minimal environments. Install a system `ffmpeg` to match production behavior.
- Benchmark (480p baseline):
  ```bash
  ./scripts/benchmark.sh
  ```

## Post-Processing

The placeholders in `vidforge/pipeline/post.py` copy inputs by default. Integrate real RIFE/ESRGAN pipelines where marked once you have compatible builds (ONNX Runtime, TensorRT, ect.).

## License

Released under the MIT License. See `LICENSE` for the full text.
