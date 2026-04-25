"""
forge/step7_cleanup.py
Step 7 — Cleanup

Removes intermediate files to free disk space.
The trained model and shared negative features are preserved.

Levels:
  light  — remove only broken/temp files
  medium — remove wav_aug (regeneratable), keep wav_base and NPYs
  full   — remove everything except final model outputs
"""
from __future__ import annotations
import shutil
from pathlib import Path

from .common import (
    WORKSPACE, log_ok, log_info, log_warn, log_section, confirm,
    fcount, get_state, mark_done,
)


def run(model_name: str, level="medium", auto=False):
    log_section("Step 7 — Cleanup")
    log_info(f"Level: {level}")

    mdir = WORKSPACE / "models" / model_name

    if level == "light":
        _light(mdir)
    elif level == "medium":
        _medium(mdir, auto)
    elif level == "full":
        _full(mdir, auto)
    else:
        log_warn(f"Unknown level '{level}' — use light / medium / full")

    mark_done("cleanup", model_name)
    log_ok("Cleanup done")


def _light(mdir: Path):
    """Remove broken/empty files and temp download dirs."""
    removed = 0
    for d in [mdir / "wav_aug", mdir / "wav_base", mdir / "features"]:
        if d.exists() and not d.is_symlink():
            for f in d.iterdir():
                if f.is_file() and f.stat().st_size < 500:
                    f.unlink(); removed += 1
    for tmp in WORKSPACE.glob("_*"):
        if tmp.is_dir():
            shutil.rmtree(str(tmp)); log_info(f"Removed tmp dir: {tmp.name}")
    log_ok(f"Light cleanup: {removed} broken files removed")


def _medium(mdir: Path, auto: bool):
    """Remove wav_aug (large), keep wav_base, NPYs, and trained model."""
    aug = mdir / "wav_aug"
    n   = fcount(aug) if aug.exists() and not aug.is_symlink() else 0
    if n == 0:
        log_info("wav_aug already empty or symlink"); return

    if not confirm(f"Remove wav_aug ({n:,} files, ~several GB)?", auto):
        log_info("Skipped"); return

    if aug.is_symlink():
        aug.unlink()
    else:
        shutil.rmtree(str(aug))
    log_ok(f"Removed wav_aug ({n:,} files)")


def _full(mdir: Path, auto: bool):
    """Keep only final model outputs (ONNX/tflite)."""
    onnx   = get_state(f"onnx_{mdir.name}")
    tflite = get_state(f"tflite_{mdir.name}")
    keep   = {Path(p) for p in [onnx, tflite] if p}

    if not confirm(f"Full cleanup of {mdir.name}/ (keeps only ONNX/tflite)?", auto):
        log_info("Skipped"); return

    for item in mdir.iterdir():
        if item in keep or item.suffix in (".onnx", ".tflite"):
            continue
        if item.is_dir():
            shutil.rmtree(str(item))
        else:
            item.unlink()

    log_ok("Full cleanup done — model outputs preserved")
