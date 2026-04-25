"""
Step 3b — STT Quality Filter + Hard Negative Mining

Two operations:
1. STT Filter: scan all TTS clips with Whisper
   - Clip sounds like wake word → keep in wav_base (positive)
   - Clip sounds garbled/wrong → move to negative_custom (hard negative!)

2. Hard Negative Generation: use "bad" Piper settings to deliberately
   generate garbled clips as additional hard negatives.
   (norman + high noise, jenny_dioco + certain speakers)
"""
import os
import subprocess
import shutil
from pathlib import Path

from .common import (
    log_section, log_step, log_ok, log_info, log_warn, log_err,
    WORKSPACE, py_bin, wav_base, fcount, mark_done,
    PIPER_NEG_VOICES, PIPER_NEG_PHRASES,
)


def run(model_name: str,
        en_text: str = "",
        he_text: str = "",
        force: bool = False) -> bool:
    log_section("Step 3b — STT Filter + Hard Negatives")

    base    = wav_base(model_name)
    neg_dir = WORKSPACE / "negative_custom"
    neg_dir.mkdir(parents=True, exist_ok=True)

    # Build acceptable transcript variants
    variants = _build_variants(en_text, he_text)
    log_info(f"  Acceptable variants: {variants[:4]}{'...' if len(variants)>4 else ''}")

    # 1. STT Filter
    _stt_filter(base, neg_dir, variants, force)

    # 2. Hard negative generation (wake-word-specific garbled variants)
    neg_phrases = _build_neg_phrases(en_text, he_text)
    _generate_hard_negatives(neg_dir, neg_phrases, force)

    mark_done("stt_filter", model_name)
    return True


def _build_variants(en_text: str, he_text: str) -> list:
    """All phonetic forms that count as a CORRECT wake word."""
    v = set()
    for txt in [en_text, he_text]:
        if not txt.strip(): continue
        words = txt.strip().split()
        v.add(txt.strip())
        if len(words) >= 2:
            v.add(" ".join(words[-2:]))
            v.add(" ".join(words[-1:]))
        # Key phonemes — if any appear it's probably correct
        for w in words:
            if len(w) > 3: v.add(w)
    return sorted(v)


def _transcript_matches(transcript: str, variants: list) -> bool:
    t = transcript.lower().strip()
    for v in variants:
        v = v.lower().strip()
        if not v: continue
        if v in t or t in v: return True
        v_words = set(v.split())
        t_words = set(t.split())
        if v_words and len(v_words & t_words) / len(v_words) >= 0.5:
            return True
    return False


def _stt_filter(base: Path, neg_dir: Path,
                variants: list, force: bool):
    """Run Whisper on all TTS clips. Move bad ones to negative_custom."""

    tts_prefixes = ("en_", "he_", "piper_", "google_", "speecht5_")
    clips = sorted(
        f for f in os.listdir(base)
        if f.endswith(".wav") and any(f.startswith(p) for p in tts_prefixes)
    )
    if not clips:
        log_info("  No TTS clips to filter"); return

    log_step(f"  STT filtering {len(clips)} clips (Whisper tiny, GPU)...")

    # Write clips list to a temp JSON file to avoid "argument list too long"
    import tempfile, json as _json
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                     delete=False, encoding="utf-8") as fh:
        _json.dump({"clips": clips, "variants": variants,
                    "base": str(base), "neg_dir": str(neg_dir)}, fh)
        cfg_file = fh.name

    script = f"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    import whisper
except ImportError:
    print(json.dumps({{"error": "whisper not installed"}}))
    sys.exit(0)

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("small.en", device=device)

with open({repr(cfg_file)}) as f:
    cfg = json.load(f)
clips    = cfg["clips"]
variants = cfg["variants"]
base     = cfg["base"]
neg_dir  = cfg["neg_dir"]
os.unlink({repr(cfg_file)})

def matches(transcript):
    t = transcript.lower().strip()
    for v in variants:
        v = v.lower().strip()
        if not v: continue
        if v in t or t in v: return True
        vw = set(v.split()); tw = set(t.split())
        if vw and len(vw & tw) / len(vw) >= 0.5: return True
    return False

kept = moved = errors = 0
for fname in clips:
    src = os.path.join(base, fname)
    if not os.path.exists(src): continue
    try:
        result = model.transcribe(src, language="en", fp16=(device=="cuda"))
        transcript = result["text"].strip()
        if matches(transcript):
            kept += 1
        else:
            dst = os.path.join(neg_dir, "hard_neg_" + fname)
            os.rename(src, dst)
            moved += 1
    except Exception as e:
        errors += 1

print(json.dumps({{"kept": kept, "moved": moved, "errors": errors}}))
"""
    r = subprocess.run([py_bin(), "-c", script],
                       capture_output=True, text=True, timeout=3600)
    try:
        import json
        last_line = [l for l in r.stdout.strip().splitlines() if l.startswith("{")]
        if last_line:
            res = json.loads(last_line[-1])
            if "error" in res:
                log_warn(f"  {res['error']}")
            else:
                log_ok(f"  Kept {res['kept']} positives, "
                       f"moved {res['moved']} → hard negatives "
                       f"({'errors: '+str(res['errors']) if res['errors'] else 'no errors'})")
    except Exception:
        if r.stdout: log_info(f"  {r.stdout[-300:]}")
        if r.stderr: log_warn(f"  {r.stderr[-200:]}")


def _build_neg_phrases(en_text: str, he_text: str) -> list:
    """Build wake-word-specific garbled phrases for hard negative generation."""
    phrases = list(PIPER_NEG_PHRASES)
    for text in [en_text]:   # English only — Piper generates English
        text = text.strip()
        if not text:
            continue
        words = text.split()
        phrases.append(text)
        if len(words) > 1:
            phrases.append(" ".join(words[1:]))   # drop first word
            phrases.append(" ".join(words[:-1]))  # drop last word
        if text.lower().startswith("hey "):
            rest = text[4:]
            phrases.append(f"okay {rest}")
        phrases.append(f"play {text}")
    # Deduplicate, keep non-empty
    seen = set()
    result = []
    for p in phrases:
        p = p.strip()
        if p and p not in seen:
            seen.add(p)
            result.append(p)
    return result[:20]


def _generate_hard_negatives(neg_dir: Path, neg_phrases: list, force: bool):
    """
    Deliberately generate garbled clips using 'bad' Piper settings.
    norman + high noise_scale = stutters/lisps
    jenny_dioco + high noise = corrupted phonemes
    These are excellent hard negatives because they're acoustically
    similar to the wake word but phonetically wrong.
    """
    prefix = "deliberate_neg"
    existing = sum(1 for f in os.listdir(neg_dir)
                   if f.startswith(prefix) and f.endswith(".wav"))
    n_expected = len(PIPER_NEG_VOICES) * len(neg_phrases) * 3

    if not force and existing >= int(n_expected * 0.8):
        log_info(f"  Hard neg clips: {existing} (skip)"); return

    voices_dir = WORKSPACE / "piper_voices"
    piper_dir  = WORKSPACE / "piper-sample-generator"

    if not piper_dir.exists():
        log_warn("  Piper repo not found — skipping hard neg generation")
        return

    log_step(f"  Generating hard negative clips with 'bad' Piper settings ({len(neg_phrases)} phrases)")

    for voice_stem, noise_scales, length_scales in PIPER_NEG_VOICES:
        onnx = voices_dir / f"{voice_stem}.onnx"
        if not onnx.exists():
            log_info(f"    {voice_stem}: not downloaded — skipping")
            continue

        script = f"""
import sys, logging
logging.disable(logging.WARNING)
sys.path.insert(0, {repr(str(piper_dir))})
from generate_samples import generate_samples_onnx

texts = []
fnames = []
for ni, ns in enumerate({repr(noise_scales)}):
    for li, ls in enumerate({repr(length_scales)}):
        for pi, phrase in enumerate({repr(neg_phrases)}):
            fnames.append("{prefix}_{voice_stem}_n{{ni}}_l{{li}}_p{{pi}}.wav")
            texts.append(phrase)

generate_samples_onnx(
    text        = texts,
    output_dir  = {repr(str(neg_dir))},
    model       = [{repr(str(onnx))}],
    max_samples = len(texts),
    file_names  = fnames,
    length_scales  = {repr(length_scales)},
    noise_scales   = {repr(noise_scales)},
    noise_scale_ws = [0.9, 1.1],
    max_speakers   = 50,
)
print("done")
""".replace("{prefix}", prefix).replace("{voice_stem}", voice_stem)

        r = subprocess.run([py_bin(), "-c", script], capture_output=True, text=True)
        n = sum(1 for f in os.listdir(neg_dir)
                if f.startswith(f"{prefix}_{voice_stem}") and f.endswith(".wav"))
        log_ok(f"    {voice_stem}: {n} hard neg clips")

    total = sum(1 for f in os.listdir(neg_dir)
                if f.startswith(prefix) and f.endswith(".wav"))
    log_ok(f"  Hard negatives total: {total} clips in negative_custom/")
