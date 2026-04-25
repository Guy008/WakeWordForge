"""
Step 8 — Custom Verifier Model
Trains a lightweight personal verifier on top of the base model.
Uses the user's own recordings as positive examples.
This dramatically improves accuracy for the specific speaker.
"""
import subprocess
from pathlib import Path

from .common import (
    log_section, log_step, log_ok, log_info, log_warn, log_err,
    WORKSPACE, py_bin, wav_base, mark_done, fcount,
)


def run(model_name: str, force: bool = False) -> bool:
    log_section("Step 8 — Custom Verifier Model")

    model_onnx = WORKSPACE / "models" / f"{model_name}.onnx"
    if not model_onnx.exists():
        log_err("Base model not found — run steps 1-5 first")
        return False

    # Find custom recordings (positive examples for verifier)
    custom_dir = wav_base(model_name)
    custom_clips = list(custom_dir.glob("custom_*.wav"))

    if len(custom_clips) < 3:
        log_warn(f"  Only {len(custom_clips)} custom recordings found")
        log_warn("  Need at least 3 for verifier — skipping")
        log_info("  Add more recordings to workspace/positive_custom/")
        return True

    log_info(f"  Custom recordings: {len(custom_clips)}")
    log_info(f"  Base model: {model_onnx.name}")

    verifier_out = WORKSPACE / "models" / f"{model_name}_verifier.pkl"
    if not force and verifier_out.exists():
        log_info(f"  Verifier already trained: {verifier_out.name} (skip)")
        return True

    log_step(f"  Training verifier on {len(custom_clips)} personal recordings...")

    script = f"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, {repr(str(WORKSPACE / "openwakeword"))})
import openwakeword

positive_clips = {repr([str(c) for c in custom_clips])}
output_path = {repr(str(verifier_out))}
model_name = {repr(str(model_onnx))}

try:
    openwakeword.train_custom_verifier(
        positive_reference_clips=positive_clips,
        negative_reference_clips=[],
        output_path=output_path,
        model_name=model_name,
    )
    print(f"Verifier saved: {{output_path}}")
except Exception as e:
    print(f"Verifier training failed: {{e}}")
    import traceback; traceback.print_exc()
"""
    r = subprocess.run([py_bin(), "-c", script], capture_output=False)

    if verifier_out.exists():
        log_ok(f"  Verifier saved: {verifier_out.name}")
        log_info("  Use with: oww = openwakeword.Model(")
        log_info(f"    wakeword_models=['{model_onnx}'],")
        log_info(f"    custom_verifier_models={{'{model_name}': '{verifier_out}'}}")
        log_info("  )")
        mark_done("verifier", model_name)
        return True
    else:
        log_warn("  Verifier training failed — base model still works without it")
        return True  # not fatal
