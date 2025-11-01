"""Smoke test for the rendering pipeline."""
from __future__ import annotations

from pathlib import Path

import cv2

from vidforge.pipeline.post import combine_shots
from vidforge.pipeline.render import RenderOptions, render_plan
from vidforge.pipeline.shots import RenderPlan, ShotConfig


def test_pipeline_smoke(tmp_path: Path) -> None:
    plan = RenderPlan(
        project_title="Smoke",
        fps=6,
        width=64,
        height=64,
        audio_file=None,
        model="cogvideox",
        shots=[
            ShotConfig(id="a", type="t2v", seconds=1.0, prompt="Scene A"),
            ShotConfig(id="b", type="i2v", seconds=1.0, prompt="Scene B", carryover=True),
            ShotConfig(id="c", type="t2v", seconds=1.0, prompt="Scene C"),
        ],
    )

    options = RenderOptions(
        out_dir=tmp_path,
        model_name=plan.model,
        seed=7,
        default_seconds=None,
        fp16=False,
        bf16=False,
        guidance_scale=1.0,
        num_steps=4,
        scheduler="ddim",
        vap_style=None,
        vap_video=None,
        resume=False,
        keep_intermediates=False,
    )

    shot_videos = render_plan(plan, options)
    assert len(shot_videos) == 3
    for path in shot_videos:
        assert path.exists()

    final = combine_shots(
        shot_videos=shot_videos,
        out_dir=tmp_path,
        use_crossfade=False,
        crossfade_seconds=0.3,
        apply_rife_opt=False,
        apply_esrgan_opt=False,
    )
    assert final.exists()

    cap = cv2.VideoCapture(str(shot_videos[0]))
    assert cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    assert abs(fps - plan.fps) <= 1
    assert width == plan.width
    assert height == plan.height
