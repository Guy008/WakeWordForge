"""
forge/step2_verify.py
Step 2 — Wake Word Verification

Generates one TTS sample (Hebrew + English) so the user can hear
whether the pronunciation is correct before committing to a full training run.
Identical to the "listen first" cell in the original Colab notebook.
"""
from __future__ import annotations
import os, subprocess, sys, tempfile
from pathlib import Path

from .common import (
    WORKSPACE, EDGE_HE_VOICES, EDGE_EN_VOICES, EDGE_RATES, EDGE_PITCHES,
    log_ok, log_info, log_warn, log_err, log_step, log_section,
    py_bin, confirm, mark_done, fcount,
)


def run(model_name: str, he_text: str, en_text: str, auto=False) -> bool:
    """
    Generate one preview sample per language and play it.
    Returns True when user confirms the pronunciation is acceptable.
    """
    log_section("Step 2 — Wake Word Verification")
    log_info(f"Model    : {model_name}")
    log_info(f"Hebrew   : {he_text}")
    log_info(f"English  : {en_text}")

    tmp = WORKSPACE / "_verify"
    tmp.mkdir(parents=True, exist_ok=True)

    samples = []

    he_out = tmp / "preview_he.mp3"
    if _tts(he_text, EDGE_HE_VOICES[0], he_out):
        samples.append(("Hebrew",  he_out))
        log_ok(f"Hebrew sample: {he_out.name}")
    else:
        log_warn("Hebrew TTS failed — check edge-tts in venv")

    en_out = tmp / "preview_en.mp3"
    if _tts(en_text, EDGE_EN_VOICES[0], en_out):
        samples.append(("English", en_out))
        log_ok(f"English sample: {en_out.name}")
    else:
        log_warn("English TTS failed")

    if not samples:
        log_err("No samples generated — cannot verify pronunciation")
        return False

    for label, path in samples:
        _play(path, label)

    if auto:
        log_info("Auto mode — skipping confirmation")
        mark_done("verify", model_name)
        return True

    ok = confirm("Pronunciation sounds correct? Continue to training?")
    if ok:
        mark_done("verify", model_name)
        log_ok("Pronunciation confirmed")
    else:
        log_warn("Adjust the wake word text and re-run step 2")
    return ok


# ── TTS helper ─────────────────────────────────────────────
def _tts(text: str, voice: str, out: Path) -> bool:
    if out.exists() and out.stat().st_size > 500:
        return True
    script = f"""
import asyncio, sys
import edge_tts

async def go():
    c = edge_tts.Communicate(text={repr(text)}, voice={repr(voice)})
    await c.save({repr(str(out))})

asyncio.run(go())
"""
    r = _run_py(script)
    return r == 0 and out.exists() and out.stat().st_size > 500


def _run_py(script: str) -> int:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                     delete=False, encoding="utf-8") as fh:
        fh.write(script); tmp = fh.name
    try:
        return subprocess.run([py_bin(), tmp], capture_output=True).returncode
    except Exception:
        return 1
    finally:
        try: os.unlink(tmp)
        except Exception: pass


# ── Playback helper ────────────────────────────────────────
def _play(path: Path, label: str):
    print(f"\n  Playing {label}: {path.name}")
    if sys.platform == "linux":
        cmds = [["ffplay", "-nodisp", "-autoexit", str(path)],
                ["aplay", str(path)]]
    elif sys.platform == "darwin":
        cmds = [["afplay", str(path)]]
    elif sys.platform == "win32":
        cmds = [["powershell", "-c",
                 f"(New-Object Media.SoundPlayer '{path}').PlaySync()"]]
    else:
        cmds = []

    for cmd in cmds:
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode == 0:
            return
    log_info(f"  (Open manually: {path})")
