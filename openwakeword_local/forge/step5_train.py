"""
forge/step5_train.py
Step 5 — Model Training

Calls openWakeWord's train.py in two phases:
  --augment_clips   reads positive_train/ positive_test/ negative_train/ negative_test/
                    and writes *_features_train.npy / *_features_test.npy
  --train_model     actual PyTorch training -> model.onnx
"""
from __future__ import annotations
import os, subprocess
from pathlib import Path

import yaml

from .common import (
    WORKSPACE,
    DEFAULT_N_SAMPLES, DEFAULT_N_STEPS, DEFAULT_PENALTY,
    TARGET_ACCURACY, TARGET_RECALL, PIPER_NEG_PHRASES,
    log_ok, log_info, log_warn, log_err, log_step, log_section,
    py_bin, fcount, mark_done, save_state,
)

# Files that are NEVER deleted without explicit --force
PRECIOUS_PATTERNS = [
    "*features*.npy",   # hours to build
    "*.onnx",           # trained model
    "*.tflite",         # trained model
]

def _is_precious(path) -> bool:
    from pathlib import Path
    p = Path(path)
    import fnmatch
    return any(fnmatch.fnmatch(p.name, pat) for pat in PRECIOUS_PATTERNS)

def _safe_delete(path, force: bool, reason: str = "") -> bool:
    """Delete only if force=True. Logs and skips otherwise."""
    from pathlib import Path
    p = Path(path)
    if not p.exists(): return True
    if _is_precious(p) and not force:
        log_warn(f"Keeping existing {p.name} — use --force to replace. {reason}")
        return False
    p.unlink()
    log_info(f"Deleted: {p.name}")
    return True


def run(model_name: str, hw,
        n_samples=DEFAULT_N_SAMPLES,
        n_steps=DEFAULT_N_STEPS,
        penalty=DEFAULT_PENALTY,
        force=False,
        en_text: str = "",
        he_text: str = "",
        arch: str = "open",
        target: str = "both") -> bool:
    """
    target: 'oww'  → ONNX only  (wyoming-openwakeword)
            'mww'  → TFLite only (microWakeWord)
            'both' → ONNX + TFLite (default)
    """
    log_section("Step 5 — Training")

    mdir = WORKSPACE / "models" / model_name
    mdir.mkdir(parents=True, exist_ok=True)

    # Verify required WAV dirs exist (created by step 4)
    required = [
        mdir / "positive_train",
        mdir / "positive_test",
        mdir / "negative_train",
        mdir / "negative_test",
    ]
    missing = [str(d) for d in required if not d.exists()]
    if missing:
        log_err("Missing directories — run step 4 first:")
        for m in missing: log_err(f"  {m}")
        return False

    # Delete stale OWW feature caches ONLY if --force is set.
    # Never auto-delete on normal runs — costs hours to rebuild!
    # The Recall=0 bug only happens when reusing features from a DIFFERENT model.
    # Since we name the model dir uniquely, stale cache is not an issue.
    if force:
        for f in mdir.glob("*features*.npy"):
            _safe_delete(f, force=True, reason="--force flag set")

    acav = WORKSPACE / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    val  = WORKSPACE / "validation_set_features.npy"
    if hw.val_rows <= 5_000:
        small = WORKSPACE / "validation_set_small.npy"
        if small.exists(): val = small

    yaml_path = mdir / "train_config.yaml"
    _write_yaml(model_name, mdir, acav, val,
                n_samples, n_steps, penalty, yaml_path, hw,
                en_text=en_text, he_text=he_text, arch=arch)

    oww_train = WORKSPACE / "openwakeword" / "openwakeword" / "train.py"
    if not oww_train.exists():
        log_err(f"train.py not found: {oww_train}")
        log_err("Run step 1 first to clone openWakeWord")
        return False

    py  = py_bin()
    env = _env(hw)

    pos_train_npy = mdir / "positive_features_train.npy"
    neg_train_npy = mdir / "negative_features_train.npy"
    pos_done = pos_train_npy.exists() and pos_train_npy.stat().st_size > 10_000
    neg_done = neg_train_npy.exists() and neg_train_npy.stat().st_size > 10_000

    if pos_done and neg_done and not force:
        log_info("Feature NPY files already exist — skipping augment_clips")
        log_info("  (use --force to recompute)")
    else:
        if hw.has_cuda:
            log_step("Phase 1 — augment_clips  (GPU)")
        else:
            log_step("Phase 1 — augment_clips  (CPU — slow without GPU)")
            log_warn("Install onnxruntime-gpu to speed up: pip install onnxruntime-gpu==1.19.2")

        # Run augment_clips on CPU to preserve VRAM for train_model
        # augment_clips is not GPU-bottlenecked — CPU is fine and avoids OOM
        # IMPORTANT: if positive NPY exists but negative is missing, OWW would skip
        # everything (it only checks for positive_features_train.npy existence).
        # Fix: delete the partial positive NPY so OWW regenerates all 4 files cleanly.
        if pos_done and not neg_done:
            log_info(f"  partial state: positive NPY exists but negative is missing — deleting positive to force full regeneration")
            for partial_npy in [pos_train_npy,
                                 mdir / "positive_features_test.npy"]:
                if partial_npy.exists():
                    partial_npy.unlink()
                    log_info(f"  Deleted partial: {partial_npy.name}")
        cpu_env = {**env, "CUDA_VISIBLE_DEVICES": "", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        r = subprocess.run(
            [py, str(oww_train), "--training_config", str(yaml_path), "--augment_clips"],
            env=cpu_env)
        ok = r.returncode == 0

        if not ok:
            log_err("augment_clips failed")
            for npy in mdir.glob("*features*.npy"):
                mb = npy.stat().st_size // 1_000_000
                log_info(f"  Partial: {npy.name} ({mb} MB)")
            return False
        log_ok("augment_clips done")

    # OWW sometimes skips creating test NPYs — create them from train if missing
    _fix_missing_test_npy(mdir)

    log_step("Phase 2 — train_model")
    r = subprocess.run(
        [py, str(oww_train), "--training_config", str(yaml_path), "--train_model"],
        env=env)

    # train_model may fail ONLY on tflite conversion (onnx_tf missing)
    # but the ONNX model is already saved — check for it
    onnx_out = WORKSPACE / "models" / f"{model_name}.onnx"
    tflite_out = WORKSPACE / "models" / f"{model_name}.tflite"

    if r.returncode != 0:
        if not onnx_out.exists():
            log_err("train_model failed — no ONNX output found")
            return False
        # ONNX saved but tflite conversion inside OWW failed — that's OK
        # We'll convert with onnx2tf below (it works and is already installed)
        log_info(f"OWW tflite step failed, will convert with onnx2tf instead")
    else:
        log_ok("train_model done")

    if not onnx_out.exists():
        log_err("No ONNX model found after training")
        return False

    import shutil as _sh
    mb = onnx_out.stat().st_size // 1000
    log_ok(f"ONNX saved: {onnx_out.name} ({mb} KB)")

    # Copy ONNX into model dir for easy access
    onnx_in_mdir = mdir / onnx_out.name
    if not onnx_in_mdir.exists():
        _sh.copy2(str(onnx_out), str(onnx_in_mdir))

    # Convert ONNX → TFLite (unless target is oww-only)
    output_dir   = WORKSPACE / "models" / "tflite_output" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    final_tflite = WORKSPACE / "models" / f"{model_name}.tflite"

    if target == "oww":
        log_info("Target=oww: skipping TFLite conversion")
        final_tflite = None
    elif not final_tflite.exists() or final_tflite.stat().st_size < 1000:
        log_step("Converting ONNX → TFLite (native TF rebuild)")
        # onnx2tf transposes input axes — use native TF rebuild to preserve shape.
        tflite_bytes = _convert_onnx_to_tflite_native(onnx_out, py, env)
        if tflite_bytes:
            final_tflite.write_bytes(tflite_bytes)
            mb = len(tflite_bytes) // 1000
            log_ok(f"TFLite saved: {final_tflite.name} ({mb} KB)")
        else:
            log_warn("TFLite conversion failed — ONNX still usable for HA wyoming-openwakeword")
            final_tflite = None
    else:
        mb = final_tflite.stat().st_size // 1000
        log_info(f"TFLite already exists: {final_tflite.name} ({mb} KB)")

    onnx   = onnx_out if onnx_out.exists() else _find_output(mdir, ".onnx")
    tflite = final_tflite if (final_tflite and final_tflite.exists()) else None

    save_state(**{
        f"onnx_{model_name}":   str(onnx)   if onnx   else "",
        f"tflite_{model_name}": str(tflite) if tflite else "",
    })

    if tflite: log_ok(f"TFLite: {tflite}")
    elif onnx: log_ok(f"ONNX:   {onnx}  (tflite not available)")

    # Always use ONNX for the test script — OWW natively uses ONNX and the
    # TFLite from onnx2tf has transposed axes that break OWW's inference.
    # TFLite is only for microWakeWord deployment (different inference pipeline).
    _write_test_script(model_name, onnx if onnx else tflite,
                       "onnx" if onnx else "tflite")
    _create_ha_output(model_name, onnx, tflite, n_steps=n_steps)
    mark_done("train", model_name)
    log_ok("Step 5 complete")
    return True


def _write_test_script(model_name: str, model_path, framework: str):
    from .common import WORKSPACE
    project_root = WORKSPACE.parent
    script_path  = project_root / f"test_{model_name}.sh"
    oww_dir  = str(WORKSPACE / "openwakeword")
    venv_py  = str(WORKSPACE / "venv" / "bin" / "python3")
    model_p  = str(model_path) if model_path else ""

    inner = f'''import sys
sys.path.insert(0, "{oww_dir}")
import numpy as np
import pyaudio
from openwakeword.model import Model

model = Model(
    wakeword_models=["{model_p}"],
    inference_framework="{framework}"
)
model_key = list(model.models.keys())[0]
pa     = pyaudio.PyAudio()
CHUNK  = 1280
stream = pa.open(rate=16000, channels=1,
                 format=pyaudio.paInt16,
                 input=True, frames_per_buffer=CHUNK)
print("מאזין... דבר את מילת ההשכמה!")
print("Ctrl+C לעצירה")
# Read required frames directly from tflite/onnx model input shape
n_frames = 16  # default
try:
    mdl_obj = list(model.models.values())[0]
    if hasattr(mdl_obj, 'get_input_details'):
        # tflite
        n_frames = mdl_obj.get_input_details()[0]['shape'][1]
    else:
        # onnx
        n_frames = mdl_obj.get_inputs()[0].shape[1]
except Exception:
    pass

# Warm up: feed silence until buffer is full (n_frames rounds guaranteed)
silence = np.zeros(CHUNK, dtype=np.int16)
for _ in range(n_frames * 2 + 20):
    try:
        model.predict(silence)
    except Exception:
        pass

try:
    while True:
        audio = np.frombuffer(
            stream.read(CHUNK, exception_on_overflow=False),
            dtype=np.int16)
        pred  = model.predict(audio)
        score = pred.get(model_key, 0.0)
        if score > 0.5:
            print(f"זוהה! ציון: {{score:.3f}}")
        elif score > 0.2:
            print(f"חלש: {{score:.3f}}")
except KeyboardInterrupt:
    pass
finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()
'''

    bash = f"#!/usr/bin/env bash\n# Auto-generated by WakeWordForge\n{venv_py} - <<'PYEOF'\n{inner}PYEOF\n"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(bash)
    import os; os.chmod(script_path, 0o755)
    log_ok(f"Test script: test_{model_name}.sh")


def _calc_val_samples(hw) -> int:
    """Calculate safe validation set size based on VRAM.
    
    OWW loads the FULL validation set as a GPU tensor during training.
    With RTX 2060 (5.6GB), 20k rows causes OOM. Use conservative limits.
    """
    if not hw.has_cuda:
        return 1_000   # CPU: no VRAM concern
    vram_gb = hw.vram_gb or 4.0

    # Conservative table based on real OOM failures:
    if vram_gb >= 16:  return 10_000
    if vram_gb >= 10:  return  5_000
    if vram_gb >=  8:  return  3_000
    if vram_gb >=  6:  return  2_000
    return 1_000   # <6GB (RTX 2060 etc.) — keep small



def _build_hard_negatives(en_text: str, he_text: str) -> list:
    """
    Build hard negative phrases for OWW custom_negative_phrases.
    Generates wake-word-specific confusable variants so the model learns to
    reject near-misses (partial phrases, prefix-less triggers, similar sounds).
    OWW generates Piper audio of these and trains the model to reject them.
    """
    negatives = set(PIPER_NEG_PHRASES)

    for text in [en_text, he_text]:
        text = text.strip()
        if not text:
            continue
        words = text.split()
        # Individual words (long enough to be meaningful)
        for w in words:
            if len(w) > 3:
                negatives.add(w)
        # Drop the first word (e.g. "hey X Y" → "X Y")
        if len(words) > 1:
            negatives.add(" ".join(words[1:]))
        # Drop the last word
        if len(words) > 1:
            negatives.add(" ".join(words[:-1]))
        # Just the last word
        if words:
            negatives.add(words[-1])
        # "play <wake word>" / "okay <wake word>" — common accidental triggers
        negatives.add(f"play {text}")
        negatives.add(f"okay {text}")
        # Prefix variants that sound similar — cover "hey" vs "okay" confusion
        if text.lower().startswith("hey "):
            rest = text[4:]
            negatives.add(f"okay {rest}")
            negatives.add(rest)

    clean = en_text.strip().lower()
    result = [p for p in negatives
              if p.strip() and p.lower() != clean and len(p.strip()) > 2]

    log_info(f"  Hard negatives: {len(result)} phrases")
    return sorted(result)[:30]


def _write_yaml(name, mdir, acav, val,
                n_samples, n_steps, penalty, yaml_path, hw,
                en_text: str = "", he_text: str = "", arch: str = "open"):
    # Build target_phrase list from all phonetic variants
    # OWW --generate_clips uses this list to generate Piper clips directly
    phrases = set()
    phrases.add(name.replace("_", " "))
    for txt in [en_text, he_text]:
        if txt.strip():
            phrases.add(txt.strip())
    target_phrases = sorted(phrases)
    rir_dir = WORKSPACE / "mit_rirs"
    # Background paths — include chime6/MUSAN if available
    bg_paths = [str(WORKSPACE / "audioset_16k"), str(WORKSPACE / "fma")]
    chime_dir = WORKSPACE / "chime6_clips"
    if chime_dir.exists() and fcount(chime_dir) > 10:
        bg_paths.append(str(chime_dir))

    cfg = {
        "model_name":    name,
        "output_dir":    str(mdir.parent),
        "piper_sample_generator_path": str(WORKSPACE / "piper-sample-generator"),
        "target_phrase": target_phrases,
        "n_samples":     n_samples,
        "n_samples_val": _calc_val_samples(hw),
        "steps":         n_steps,
        "device":        "cuda" if hw.has_cuda else "cpu",
        "model_type":    "dnn",
        "layer_size":    64 if arch == "micro" else 192,  # micro=64 (mobile/edge), open=192 (default)
        "tts_batch_size": 50,
        "augmentation_rounds":     1,
        "augmentation_batch_size": 4,   # keep small — augment runs on CPU
        "target_accuracy": TARGET_ACCURACY,
        "target_recall":   TARGET_RECALL,
        "target_false_positives_per_hour": 0.2,
        "max_negative_weight": penalty,
        "custom_negative_phrases": _build_hard_negatives(en_text, he_text),
        "background_paths": bg_paths,
        "background_paths_duplication_rate": [1] * len(bg_paths),
        "rir_paths": [str(rir_dir)] if rir_dir.exists() and fcount(rir_dir) > 0 else [],
        "feature_data_files": {},
        "batch_n_per_class": {
            "ACAV100M_sample":      1024,
            "adversarial_negative": 50,
            "positive":             50,
        },
        "false_positive_validation_data_path": str(val),
    }
    if acav.exists():
        cfg["feature_data_files"]["ACAV100M_sample"] = str(acav)

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)
    log_info(f"YAML written: {yaml_path.name}")


def _fix_missing_test_npy(mdir: Path):
    """OWW sometimes skips *_test.npy — create from train slice if missing."""
    try:
        import numpy as np
        for kind in ("positive", "negative"):
            test = mdir / f"{kind}_features_test.npy"
            train = mdir / f"{kind}_features_train.npy"
            if not test.exists() and train.exists():
                data = np.load(str(train))
                np.save(str(test), data[:min(1000, data.shape[0])])
                log_info(f"Created {test.name} from train slice")
    except Exception as e:
        log_warn(f"Could not fix test NPY: {e}")


def _env(hw) -> dict:
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if hw.has_cuda:
        env["CUDA_VISIBLE_DEVICES"] = "0"
        # Force onnxruntime to use CUDA (needed for augment_clips speed)
        env["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "0"
        env["ONNXRUNTIME_PROVIDERS"] = "CUDAExecutionProvider,CPUExecutionProvider"
    return env


def _find_output(mdir: Path, ext: str):
    hits = list(mdir.rglob(f"*{ext}"))
    return hits[0] if hits else None


def _convert_onnx_to_tflite_native(onnx_path: Path, py: str, env: dict):
    """
    Convert OWW's ONNX model to TFLite by rebuilding in TensorFlow.

    onnx2tf transposes the 3D input [batch, n_frames, features] to
    [batch, features, n_frames], which scrambles the Flatten element order
    and produces wrong predictions.  This function extracts weights from the
    ONNX file and rebuilds the identical network in Keras, preserving axes.

    Returns: tflite bytes, or None on failure.
    """
    script = f"""
import sys, os, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import onnx
from onnx import numpy_helper

m = onnx.load({repr(str(onnx_path))})
ws = {{t.name: numpy_helper.to_array(t) for t in m.graph.initializer}}

# Read input shape from ONNX model
inp = m.graph.input[0]
shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]  # e.g. [1, 20, 96]
n_frames, n_feats = shape[1], shape[2]

import tensorflow as tf

# Extract Gemm weights (ONNX Gemm: Y = X @ B^T + C, transB=1 by default)
W1 = ws["layer1.weight"]          # shape [hidden, n_frames*n_feats]
b1 = ws.get("layer1.bias", None)
LN1_g = ws["layernorm1.weight"]
LN1_b = ws["layernorm1.bias"]

# Detect which block key is used
block_key = [k for k in ws if "fcn_layer.weight" in k]
if not block_key:
    print("ERROR: could not find fcn_layer weights", file=sys.stderr)
    sys.exit(1)
block_prefix = block_key[0].rsplit(".fcn_layer.weight", 1)[0]
W2 = ws[f"{{block_prefix}}.fcn_layer.weight"]
b2 = ws.get(f"{{block_prefix}}.fcn_layer.bias", None)
LN2_g = ws[f"{{block_prefix}}.layer_norm.weight"]
LN2_b = ws[f"{{block_prefix}}.layer_norm.bias"]

W3 = ws["last_layer.weight"]
b3 = ws.get("last_layer.bias", None)

hidden = W1.shape[0]

# Build Keras model mirroring the ONNX graph exactly
inp_layer = tf.keras.Input(shape=(n_frames, n_feats), batch_size=1, name="input_0")
x = tf.keras.layers.Flatten()(inp_layer)
x = tf.keras.layers.Dense(hidden, use_bias=(b1 is not None))(x)
x = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Dense(hidden, use_bias=(b2 is not None))(x)
x = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x)
x = tf.keras.layers.ReLU()(x)
out = tf.keras.layers.Dense(1, use_bias=(b3 is not None), activation="sigmoid")(x)
model = tf.keras.Model(inputs=inp_layer, outputs=out)

# Assign weights layer by layer
dense_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
ln_layers    = [l for l in model.layers if isinstance(l, tf.keras.layers.LayerNormalization)]

for dl, (W, b) in zip(dense_layers, [(W1, b1), (W2, b2), (W3, b3)]):
    wlist = [W.T]  # Keras Dense uses W^T convention (n_in, n_out)
    if b is not None:
        wlist.append(b)
    dl.set_weights(wlist)

for ln, (g, b) in zip(ln_layers, [(LN1_g, LN1_b), (LN2_g, LN2_b)]):
    ln.set_weights([g, b])

# Verify: compare outputs on random input
x_test = np.random.randn(1, n_frames, n_feats).astype(np.float32)
import onnxruntime as ort
sess = ort.InferenceSession({repr(str(onnx_path))})
onnx_out = sess.run(None, {{sess.get_inputs()[0].name: x_test}})[0]
keras_out = model(x_test, training=False).numpy()
max_diff = float(np.abs(onnx_out - keras_out).max())
if max_diff > 0.01:
    print(f"WARNING: Keras/ONNX output mismatch: max_diff={{max_diff:.4f}}", file=sys.stderr)
else:
    print(f"Verified: max diff ONNX vs Keras = {{max_diff:.6f}}")

# Convert to TFLite — write to a temp file, print path on stdout
import tempfile
with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as fh:
    out_path = fh.name
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_bytes = converter.convert()
open(out_path, "wb").write(tflite_bytes)
print(out_path)
"""
    try:
        r = subprocess.run(
            [py, "-c", script],
            capture_output=True, text=True, env=env, timeout=120)
        # Find the temp file path printed on the last stdout line
        out_lines = [l.strip() for l in r.stdout.splitlines() if l.strip()]
        tflite_path = None
        for line in reversed(out_lines):
            if line.endswith(".tflite") and os.path.exists(line):
                tflite_path = line
                break
            if "Verified:" in line:
                log_info(f"  {line}")
        if tflite_path:
            data = Path(tflite_path).read_bytes()
            os.unlink(tflite_path)
            return data
        stderr = r.stderr
        log_warn(f"  TFLite native convert failed: {stderr[-400:] if stderr else '(no output)'}")
    except Exception as e:
        log_warn(f"  TFLite native convert exception: {e}")
    return None


def _create_mww_manifest(model_name: str, tflite_path: Path,
                          deploy_dir: Path, n_steps: int) -> Path:
    """
    Generate the microWakeWord JSON manifest required by ESPHome.

    ESPHome's micro_wake_word component will NOT load a bare .tflite file —
    it needs a JSON manifest that contains:
      - model metadata (wake word name, version, classes)
      - the TFLite model embedded as a base64 string

    The generated JSON can be referenced directly in ESPHome YAML:
      micro_wake_word:
        model: /config/custom_components/micro_wake_word/models/{model_name}.json
    """
    import json as _json, base64 as _b64

    tflite_bytes = tflite_path.read_bytes()
    model_b64 = _b64.b64encode(tflite_bytes).decode("ascii")

    wake_word_label = model_name.replace("_", " ")

    manifest = {
        "type": "micro_wake_word",
        "wake_word": wake_word_label,
        "version": 1,
        "minimum_esphome_version": "2024.2.0",
        "micro": {
            "model": model_b64,
            "quantization_scheme": "s8",
            "trained_steps": n_steps,
            "classes": [
                {
                    "id": "wake_word",
                    "label": wake_word_label,
                }
            ],
            "stride_size": 1,
        },
    }

    json_path = deploy_dir / f"{model_name}.json"
    json_path.write_text(_json.dumps(manifest, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    return json_path


def _create_ha_output(model_name: str, onnx_path, tflite_path,
                      n_steps: int = DEFAULT_N_STEPS):
    """
    Copy trained model files to workspace/ha_deploy/{model_name}/
    and write a deployment README so the user knows exactly what to do.
    """
    import shutil as _sh2
    deploy_dir = WORKSPACE / "ha_deploy" / model_name
    deploy_dir.mkdir(parents=True, exist_ok=True)

    files_ready = []
    if onnx_path and Path(onnx_path).exists():
        dst = deploy_dir / Path(onnx_path).name
        _sh2.copy2(str(onnx_path), str(dst))
        files_ready.append(f"  ONNX  (openWakeWord / wyoming): {dst.name}")

    mww_json_path = None
    if tflite_path and Path(tflite_path).exists():
        dst = deploy_dir / Path(tflite_path).name
        _sh2.copy2(str(tflite_path), str(dst))
        files_ready.append(f"  TFLite (microWakeWord):          {dst.name}")
        # Generate the JSON manifest — ESPHome requires this alongside the TFLite
        mww_json_path = _create_mww_manifest(model_name, dst, deploy_dir, n_steps)
        files_ready.append(f"  JSON manifest (ESPHome):         {mww_json_path.name}")

    if not files_ready:
        return

    mww_yaml = (
        f"  micro_wake_word:\n"
        f"    model: /config/custom_components/micro_wake_word/models/{model_name}.json\n"
        if mww_json_path else
        f"  # TFLite not available\n"
    )

    readme = deploy_dir / "DEPLOY.txt"
    readme.write_text(
        f"Wake word model: {model_name}\n"
        f"{'=' * 50}\n\n"
        f"Files ready for Home Assistant:\n"
        + "\n".join(files_ready) + "\n\n"
        f"── openWakeWord (wyoming-openwakeword / HA server) ──────\n"
        f"1. Copy {model_name}.onnx to the wyoming-openwakeword\n"
        f"   custom_models/ directory on your HA server.\n"
        f"2. Restart the wyoming-openwakeword add-on.\n"
        f"3. In HA: Settings → Voice Assistants → select wake word.\n\n"
        f"── microWakeWord (ESP32-S3 / ESPHome) ───────────────────\n"
        f"1. Copy {model_name}.json to your ESPHome config dir:\n"
        f"   /config/custom_components/micro_wake_word/models/\n"
        f"   (The JSON contains the model embedded — no separate .tflite needed)\n"
        f"2. Reference it in your ESPHome device YAML:\n"
        + mww_yaml +
        f"\nNote: {model_name}.tflite is also provided for manual inspection\n"
        f"or use with tools that accept raw TFLite files.\n",
        encoding="utf-8"
    )

    log_ok(f"HA deploy files ready: {deploy_dir}/")
    for f in files_ready:
        log_ok(f)
    if mww_json_path:
        log_ok(f"  microWakeWord manifest: {mww_json_path.name} (TFLite embedded as base64)")
