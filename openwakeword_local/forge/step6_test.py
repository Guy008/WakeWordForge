"""
forge/step6_test.py
Step 6 — Live Model Testing

Runs the trained ONNX model against the microphone in real time.
Shows a live score bar. Press Ctrl+C to stop.
"""
from __future__ import annotations
import subprocess, sys, tempfile, os
from pathlib import Path

from .common import (
    WORKSPACE, log_ok, log_info, log_warn, log_err, log_step, log_section,
    py_bin, get_state, mark_done,
)


def run(model_name: str, threshold=0.5, duration=60):
    log_section("Step 6 — Live Testing")

    model_path, framework = _find_model(model_name)
    if not model_path:
        log_err(f"No model found for '{model_name}' — run step 5 first")
        return False

    log_ok(f"Model: {model_path.name} ({framework})")
    log_info(f"Threshold : {threshold}")
    log_info(f"Duration  : {duration}s (Ctrl+C to stop)")
    log_info("Say the wake word...")
    print()

    # Prefer ONNX for live testing — tflite crashes if buffer not yet full
    onnx_candidate = WORKSPACE / "models" / f"{model_name}.onnx"
    live_path      = onnx_candidate if onnx_candidate.exists() else model_path
    live_fw        = "onnx"        if onnx_candidate.exists() else framework

    script = f"""
import sys, time, collections
sys.path.insert(0, {repr(str(WORKSPACE / "openwakeword"))})

import numpy as np
from openwakeword.model import Model

try:
    import pyaudio
    CHUNK    = 1280
    FORMAT   = pyaudio.paInt16
    CHANNELS = 1
    RATE     = 16000

    oww = Model(wakeword_models=[{repr(str(live_path))}],
                inference_framework={repr(live_fw)})
    model_key = list(oww.models.keys())[0]

    # Read required frames from model input shape
    n_frames = 16
    try:
        mdl_obj = list(oww.models.values())[0]
        if hasattr(mdl_obj, 'get_input_details'):
            n_frames = int(mdl_obj.get_input_details()[0]['shape'][1])
        else:
            n_frames = int(mdl_obj.get_inputs()[0].shape[1])
    except Exception:
        pass

    silence = np.zeros(CHUNK, dtype=np.int16)
    for _ in range(n_frames * 2 + 20):
        try:
            oww.predict(silence)
        except Exception:
            pass

    pa  = pyaudio.PyAudio()
    stream = pa.open(format=FORMAT, channels=CHANNELS,
                     rate=RATE, input=True,
                     frames_per_buffer=CHUNK)

    scores    = collections.deque(maxlen=20)
    threshold = {threshold}
    start     = time.time()
    duration  = {duration}

    print("  Listening... (Ctrl+C to stop)")
    while time.time() - start < duration:
        audio = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False),
                              dtype=np.int16)
        pred  = oww.predict(audio)
        score = pred.get(model_key, 0.0)
        scores.append(score)
        peak  = max(scores)

        bar_len = 40
        filled  = int(peak * bar_len)
        bar     = "#" * filled + "-" * (bar_len - filled)
        marker  = " <<< WAKE WORD!" if score >= threshold else ""
        print(f"\\r  [{{bar}}] {{score:.3f}}{{marker}}", end="", flush=True)

    stream.stop_stream()
    stream.close()
    pa.terminate()
    print("\\nDone.")

except ImportError:
    print("pyaudio not available — cannot run live test")
except KeyboardInterrupt:
    print("\\nStopped.")
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                     delete=False, encoding="utf-8") as fh:
        fh.write(script); tmp = fh.name
    try:
        subprocess.run([py_bin(), tmp])
    except KeyboardInterrupt:
        pass  # Ctrl+C in child process — normal exit
    finally:
        try: os.unlink(tmp)
        except Exception: pass

    mark_done("test", model_name)
    return True


def _find_model(model_name: str):
    """Find best available model file: prefer tflite, fall back to onnx."""
    # 1. tflite from state
    cached_tflite = get_state(f"tflite_{model_name}")
    if cached_tflite and Path(cached_tflite).exists():
        return Path(cached_tflite), "tflite"
    # 2. tflite in workspace/models/
    tflite = WORKSPACE / "models" / f"{model_name}.tflite"
    if tflite.exists() and tflite.stat().st_size > 1000:
        return tflite, "tflite"
    # 3. onnx from state
    cached_onnx = get_state(f"onnx_{model_name}")
    if cached_onnx and Path(cached_onnx).exists():
        return Path(cached_onnx), "onnx"
    # 4. onnx in workspace/models/
    onnx = WORKSPACE / "models" / f"{model_name}.onnx"
    if onnx.exists():
        return onnx, "onnx"
    # 5. Scan model dir
    mdir = WORKSPACE / "models" / model_name
    for ext, fw in [(".tflite","tflite"), (".onnx","onnx")]:
        hits = list(mdir.rglob(f"*{ext}")) if mdir.exists() else []
        if hits: return hits[0], fw
    return None, None
