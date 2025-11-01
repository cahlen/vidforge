"""VidForge command-line interface."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from vidforge.pipeline.post import combine_shots, mux_audio_if_present
from vidforge.pipeline.render import RenderOptions, render_plan
from vidforge.pipeline.shots import load_plan
from vidforge.utils.video import detect_vram

app = typer.Typer(help="Render long-form videos from structured shot plans.")
console = Console()
logging.basicConfig(level=logging.INFO)


@app.command()
def render(
    plan: Path = typer.Option(..., exists=True, readable=True, help="Path to the shot list JSON."),
    out: Path = typer.Option(Path("out/project"), help="Output directory for renders."),
    model: str = typer.Option("cogvideox", help="Model backend to use.", case_sensitive=False),
    seed: Optional[int] = typer.Option(None, help="Global seed for reproducible output."),
    fps: Optional[int] = typer.Option(None, help="Override plan FPS."),
    width: Optional[int] = typer.Option(None, help="Override output width."),
    height: Optional[int] = typer.Option(None, help="Override output height."),
    seconds_per_shot: Optional[float] = typer.Option(None, help="Default seconds when missing per shot."),
    rife: bool = typer.Option(True, help="Enable RIFE smoothing."),
    esrgan: bool = typer.Option(True, help="Enable ESRGAN upscaling."),
    vap_style: Optional[Path] = typer.Option(None, exists=False, help="Optional style reference image."),
    vap_video: Optional[Path] = typer.Option(None, exists=False, help="Optional reference video."),
    resume: bool = typer.Option(False, help="Skip shots with existing outputs."),
    keep_intermediates: bool = typer.Option(True, help="Retain intermediate frames."),
    fp16: bool = typer.Option(True, help="Prefer fp16 precision."),
    bf16: bool = typer.Option(False, help="Prefer bf16 precision."),
    guidance_scale: float = typer.Option(7.0, help="Guidance scale."),
    num_steps: int = typer.Option(30, help="Inference steps."),
    scheduler: str = typer.Option("ddim", help="Scheduler key."),
    crossfade: bool = typer.Option(False, help="Use crossfades between shots."),
    crossfade_seconds: float = typer.Option(0.6, help="Crossfade duration in seconds."),
) -> None:
    model = model.lower()
    if model not in {"cogvideox", "hunyuan"}:
        raise typer.BadParameter("Model must be 'cogvideox' or 'hunyuan'.")

    plan_data = load_plan(plan)
    if fps:
        plan_data.fps = fps
    if width:
        plan_data.width = width
    if height:
        plan_data.height = height

    options = RenderOptions(
        out_dir=out,
        model_name=model,
        seed=seed,
        default_seconds=seconds_per_shot,
        fp16=fp16,
        bf16=bf16,
        guidance_scale=guidance_scale,
        num_steps=num_steps,
        scheduler=scheduler,
        vap_style=str(vap_style) if vap_style else None,
        vap_video=str(vap_video) if vap_video else None,
        resume=resume,
        keep_intermediates=keep_intermediates,
    )

    vram = detect_vram()
    if vram is not None:
        console.log(f"[cyan]Detected GPU memory:[/] {vram} MB")

    shot_videos = render_plan(plan_data, options)
    final = combine_shots(
        shot_videos=shot_videos,
        out_dir=out,
        use_crossfade=crossfade,
        crossfade_seconds=crossfade_seconds,
        apply_rife_opt=rife,
        apply_esrgan_opt=esrgan,
    )
    final = mux_audio_if_present(final, plan_data.audio_file, out)
    console.log(f"[bold green]Final video ready:[/] {final}")


def main() -> None:  # pragma: no cover - CLI entrypoint
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
