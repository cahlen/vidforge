# VidForge

VidForge is a modular toolkit for stitching text-to-video (T2V) and image-to-video (I2V) shots into cohesive long-form sequences with optional style guidance, frame continuity, and post-processing.

## Quickstart

1. **Provision dependencies**
  ```bash
  ./scripts/setup_env.sh
  ```
  The setup script autodetects your GPU. Cards newer than the PyTorch stable matrix (for example RTX 5090, compute capability 12.x) trigger an automatic fallback to the PyTorch nightly `cu128` wheels and drop conflicting extras such as `xformers`. Set `CUDA_VERSION_TAG` explicitly (e.g. `CUDA_VERSION_TAG=cu121`) if you need a different build.
2. **Point to model checkpoints**
  ```bash
  huggingface-cli login  # optional if you use private weights
  export VIDFORGE_COGVIDEOX_PATH=THUDM/CogVideoX-5B
  export VIDFORGE_HUNYUAN_PATH=TencentARC/HunyuanVideo
  ```
  Replace the repo IDs with local filesystem paths if you have mirrored copies. The environment variables gate the heavyweight diffusion backends so you can opt-in when storage is available.
3. **Render the demo plan**
   ```bash
   ./scripts/render_example.sh
   ```
4. Find outputs under `out/desert_patrol/`.

### Talking-Head Quickstart

1. (Optional) provide a Wav2Lip checkpoint if you already have one:
  ```bash
  export VIDFORGE_WAV2LIP_CHECKPOINT=/path/to/wav2lip_gan.pth
  ```
  If unset, VidForge attempts to download the official GAN checkpoint from Hugging Face (or Google Drive via `gdown`) the first time you run the talking-head command.
2. Generate a presenter portrait, synthesize narration, and lip-sync the result:
  ```bash
  ./scripts/render_talking_head.sh
  ```
3. The talking-head output appears under `out/talking_head_demo/`. If Wav2Lip cannot be obtained, VidForge falls back to a static portrait video while keeping the audio track intact.

> **Note:** The repository ships with lightweight placeholder generators for both CogVideoX and Hunyuan so the pipeline and tests run end-to-end without bulky checkpoints. Replace them with real model integrations before production use (see TODO markers in `vidforge/models/`).

## Features

- JSON-driven shot orchestration with per-shot overrides for fps, resolution, and carryover.
- Automatic last-frame carryover from T2V → I2V shots, plus previews of first/last frames for QA.
- Optional video-as-prompt adapter interface for consistent style or semantics across shots.
- Post-processing hooks for RIFE smoothing, ESRGAN upscaling, crossfades, and audio muxing.
- Rich CLI with Typer, structured logging, resume support, and VRAM detection hints.

## Model Setup

1. Download CogVideoX-5B and/or HunyuanVideo checkpoints from Hugging Face (or mirror them locally).
2. Export `VIDFORGE_COGVIDEOX_PATH` / `VIDFORGE_HUNYUAN_PATH` to point at the repo IDs or local directories and, if required, run `huggingface-cli login` so diffusers can authenticate.
3. Optional: install backend-specific accelerators (FlashAttention, xFormers) once compatible wheels ship for your CUDA version.

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

For a narrated conversation, see `examples/conversation_plan.json` which pairs with the `scripts/render_conversation.sh` helper. That script synthesizes a spoken track with Coqui TTS (downloaded automatically on first use) and renders a presenter-style sequence in `out/studio_chat/`.

## Talking-Head Workflow

Use the dedicated CLI when you want a single-speaker avatar that matches a textual description and speaks an audio track you provide:

```bash
source .venv/bin/activate
vidforge talking-head \
  --prompt "professional news anchor, cinematic lighting" \
  --audio-file speech.wav \
  --out out/anchor_demo
```

Key points:

- Supply `--audio-text` instead of `--audio-file` to let VidForge synthesize the narration with Coqui TTS on the fly.
- Provide `--wav2lip-checkpoint` if you have already downloaded `wav2lip_gan.pth`. Otherwise, VidForge will clone the upstream Wav2Lip repository into `~/.cache/vidforge/wav2lip/` and attempt to fetch weights automatically. This requires Git, ffmpeg, and either Hugging Face credentials or network access to Google Drive.
- Set `VIDFORGE_DISABLE_WAV2LIP=1` to deliberately skip lip-syncing (useful for tests). The command still produces a video by looping the generated portrait and muxing the narration audio.
- Portraits now work best with Black Forest Labs Flux. Set `VIDFORGE_PORTRAIT_MODEL_ID=black-forest-labs/FLUX.1-dev` (or another Flux variant) for the highest-fidelity faces; VidForge automatically offloads the pipeline to fit in GPU memory. The Stable Diffusion fallback (`runwayml/stable-diffusion-v1-5`) remains available if Flux weights are absent.
- Fine-tune lip-sync with knobs like `--wav2lip-static/--no-wav2lip-static`, `--wav2lip-pads 0 12 0 0`, `--wav2lip-resize-factor 0.9`, or `--wav2lip-face-det-scale 1.2`; every run loudness-normalises narration to -16 LUFS to maintain consistent audio levels.

## Talking-Head Datasets

Scale beyond single clips by describing a roster of presenters inside a JSON config and letting VidForge batch the renders:

```bash
source .venv/bin/activate
vidforge talking-head-dataset examples/talking_head_dataset.json --out out/datasets/demo
```

- Each entry defines an `id`, portrait `prompt`, and either narration `text` (synthesised via Coqui TTS) or an `audio_file` you prepared in advance.
- Use the config’s `base_prompt` to lock framing (e.g. first-person interview) and override per-entry look with `appearance` and environment with `background`. The pipeline automatically appends a strict negative prompt to avoid multi-subject scenes.
- Advanced lip-sync settings (`wav2lip_pads`, `wav2lip_resize_factor`, `wav2lip_face_det_scale`, `wav2lip_face_det_batch_size`, `wav2lip_static`) can be declared once at the top level or overridden per entry.
- Outputs land in `<out>/<id>/` subfolders containing `portrait.png`, normalised `speech.wav`, and the rendered `talking_head.mp4`.
- A consolidated `dataset_manifest.jsonl` plus `dataset_summary.json` capture relative paths and metadata for downstream training.
- Provide `--wav2lip-checkpoint` or set `VIDFORGE_WAV2LIP_CHECKPOINT` to ensure clips animate with full lip-sync; otherwise the generator falls back to static portraits.

## Testing & Benchmarking

- Smoke tests:
  ```bash
  source .venv/bin/activate
  pytest
  ```
  👷 Tip: Encoding now cascades from `ffmpeg-python` → `imageio` → OpenCV, so smoke tests succeed even without a working `ffmpeg` binary. Install system ffmpeg to mirror production and avoid the reduced-dimension warning emitted by `imageio`.
- Benchmark (480p baseline):
  ```bash
  ./scripts/benchmark.sh
  ```

## Post-Processing

The placeholders in `vidforge/pipeline/post.py` copy inputs by default. Integrate real RIFE/ESRGAN pipelines where marked once you have compatible builds (ONNX Runtime, TensorRT, ect.).

Audio muxing gracefully skips missing or empty tracks and falls back to video-only output if ffmpeg cannot combine the streams.

## Narration & Audio

Pass `--generate-audio-text "..."` to the CLI (or use `scripts/render_conversation.sh`) to synthesize narration with Coqui TTS. The first invocation downloads the configured TTS checkpoint and saves the spoken WAV beside the render output.

## License

Released under the MIT License. See `LICENSE` for the full text.
