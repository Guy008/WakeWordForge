"""
forge/step3_generate.py
Step 3 — Sample Generation

3a. TTS base clips: edge-tts (HE + EN voices) + optional Piper
3b. Import user's custom recordings from positive_custom/
3c. Augmentation: volume × tempo grid → wav_aug/

The number of augmented samples is determined by n_target.
Everything is idempotent — re-running skips completed work.
"""
from __future__ import annotations
import os, random, shutil, subprocess, sys, tempfile
from itertools import product
from pathlib import Path

from .common import (
    WORKSPACE, CUSTOM_POS_DIR, MY_RECORDINGS_DIR,
    EDGE_HE_VOICES, EDGE_EN_VOICES, EDGE_RATES, EDGE_PITCHES,
    AUG_VOLUMES, AUG_TEMPOS, TTS_PER_VOICE,
    PERSONAL_TEMPOS, PERSONAL_VOLUMES, PERSONAL_PITCHES, PERSONAL_EQ,
    PIPER_PT_MODEL_NAME, PIPER_LENGTH_SCALES, PIPER_NOISE_SCALES,
    PIPER_NOISE_SCALE_WS, PIPER_SLERP_WEIGHTS, PIPER_MAX_SPEAKERS, PIPER_SAMPLES_PER_TEXT,
    WYOMING_HOST, WYOMING_PORT,
    CLIP_MIN_BYTES, CLIP_MAX_BYTES, CLIP_MIN_DUR, CLIP_MAX_DUR,
    log_ok, log_info, log_warn, log_err, log_step, log_section,
    py_bin, fcount, delete_broken, ffcmd, mark_done, save_state,
    wav_base, wav_aug,
)


def run(model_name: str, he_text: str, en_text: str,
        n_target: int, force=False, ipa_text: str = "") -> bool:
    log_section("Step 3 — Sample Generation")

    base = wav_base(model_name)
    aug  = wav_aug(model_name)
    base.mkdir(parents=True, exist_ok=True)

    # 3a: Personal recordings — primary source (my_recordings/ + positive_custom/)
    _import_personal(base)
    _import_custom(base)
    n_personal = fcount(base)
    log_info(f"Personal recordings: {n_personal}")

    # 3b: TTS clips
    if n_personal == 0:
        # TTS-only mode — no personal recordings provided.
        # Use ALL available edge-tts voices (10 EN + 2 HE) for maximum diversity.
        log_warn("No personal recordings found — running in TTS-only mode")
        log_info("  Tip: add WAV files to my_recordings/ for a much better model")
        tts_quota = 200
        _tts_minimal(he_text, en_text, base, tts_quota, force, use_all_voices=True)
    else:
        # 3:1 ratio — 1 TTS clip per 3 personal recordings
        tts_quota = max(10, n_personal // 3)
        log_info(f"TTS quota: {tts_quota} clips (ratio 3:1 vs {n_personal} personal)")
        _tts_minimal(he_text, en_text, base, tts_quota, force)

    # 3b+: IPA synthesis — generates clips for phonemes English TTS cannot produce
    if ipa_text.strip():
        _generate_espeak_ipa(ipa_text.strip(), base, force)

    # 3c: STT quality filter — Whisper validates TTS clips, moves bad ones to hard negatives
    try:
        from . import step3b_stt_filter
        step3b_stt_filter.run(model_name,
                              en_text=en_text, he_text=he_text,
                              force=force)
    except Exception as e:
        log_warn(f"STT filter skipped: {e}")

    n_base = fcount(base)
    n_tts = n_base - n_personal
    if n_personal == 0:
        log_ok(f"Base samples: {n_base} (TTS-only: {n_tts} clips from {len(EDGE_EN_VOICES) + len(EDGE_HE_VOICES)} voices)")
    else:
        log_ok(f"Base samples: {n_base} ({n_personal} personal + {n_tts} TTS)")
    if n_base == 0:
        log_err("No base samples — add recordings to my_recordings/ or positive_custom/")
        return False

    # 3d: Intensive augmentation
    #     Personal recordings: heavy aug (tempo × volume × pitch × EQ)
    #     TTS clips: light aug (tempo × volume)
    n_existing = fcount(aug)
    if not force and n_existing >= int(n_target * 0.9):
        log_info(f"Augmentation already complete: {n_existing:,}")
    else:
        _augment(base, aug, n_target)

    save_state(**{
        f"n_base_{model_name}": fcount(base),
        f"n_aug_{model_name}":  fcount(aug),
    })
    mark_done("generate", model_name)
    log_ok(f"Step 3 complete: {fcount(aug):,} augmented samples")
    return True


# ══════════════════════════════════════════════════════════
#  Text variants + Punctuation combos
# ══════════════════════════════════════════════════════════
# Punctuation appended to each word independently.
# Splitting the text with | gives multiple base phrases.
# Example: "mirror mirror|mirror mirror on the wall"
#          → two base phrases, each gets all punct combos.
# edge-tts punct variants — keep minimal (speed > variety for MS engine)
PUNCT = ["", ".", "?"]
MAX_PUNCT_COMBOS = 20   # very small — edge-tts is slow, Piper PT does the heavy lifting


def _all_variants(text: str) -> list:
    """
    Expand text into all punctuation combos across all | variants.

    "A B | A B C" → variants of "A B" (36) + variants of "A B C" (216) = 252 total
    Each variant has every word independently punctuated.
    """
    import itertools, random as _rnd

    base_phrases = [p.strip() for p in text.split("|") if p.strip()]
    if not base_phrases:
        return [text]

    all_variants = []
    seen = set()

    for phrase in base_phrases:
        words = phrase.split()
        if not words:
            continue
        combos = list(itertools.product(PUNCT, repeat=len(words)))

        # Uniform sample if too many
        if len(combos) > MAX_PUNCT_COMBOS:
            step = len(combos) // MAX_PUNCT_COMBOS
            combos = combos[::step][:MAX_PUNCT_COMBOS]

        for puncts in combos:
            variant = " ".join(w + p for w, p in zip(words, puncts))
            if variant not in seen:
                seen.add(variant)
                all_variants.append(variant)

    # Always include all bare phrases first (no punct)
    for phrase in base_phrases:
        if phrase not in seen:
            seen.add(phrase)
            all_variants.insert(0, phrase)

    return all_variants





# ══════════════════════════════════════════════════════════
#  IPA synthesis (espeak-ng)
# ══════════════════════════════════════════════════════════
def _generate_espeak_ipa(ipa_text: str, out_dir: Path, force: bool) -> int:
    """
    Generate TTS clips from IPA phonetic notation using espeak-ng.

    espeak-ng accepts IPA via SSML <phoneme alphabet="ipa" ph="...">,
    enabling synthesis of sounds that English orthography cannot represent —
    e.g. Hebrew ʁ (uvular fricative), ʔ (glottal stop), ħ, χ, etc.

    Produces: len(voices) × len(rates) × len(pitches) clips → wav_base/
    Each clip is resampled to 16 kHz mono for OWW compatibility.
    """
    import html as _html
    import shutil as _sh

    if not _sh.which("espeak-ng"):
        log_warn("espeak-ng not found — skipping IPA synthesis")
        log_warn("  Install:  sudo apt install espeak-ng   (Debian/Ubuntu/WSL)")
        log_warn("            brew install espeak-ng        (macOS)")
        return 0

    PREFIX  = "espeak_ipa"
    voices  = ["en+m1", "en+m2", "en+m3", "en+m4", "en+m5",
               "en+f1", "en+f2", "en+f3", "en+f4"]
    rates   = [130, 150, 170]
    pitches = [40, 55, 70]

    existing = sum(1 for e in os.scandir(out_dir)
                   if e.name.startswith(PREFIX) and e.name.endswith(".wav"))
    n_expected = len(voices) * len(rates) * len(pitches)

    if not force and existing >= int(n_expected * 0.8):
        log_info(f"  espeak-ng IPA: {existing} clips (skip)")
        return existing

    log_step(f"  espeak-ng IPA [{ipa_text}]  "
             f"{len(voices)} voices × {len(rates)} rates × {len(pitches)} pitches "
             f"= {n_expected} clips")

    ph_attr = _html.escape(ipa_text, quote=True)
    ssml    = f'<speak><phoneme alphabet="ipa" ph="{ph_attr}">word</phoneme></speak>'

    count = 0
    for vi, voice in enumerate(voices):
        for ri, rate in enumerate(rates):
            for pi, pitch in enumerate(pitches):
                out_wav = out_dir / f"{PREFIX}_v{vi}_r{ri}_p{pi}.wav"
                if not force and out_wav.exists() and out_wav.stat().st_size > CLIP_MIN_BYTES:
                    count += 1
                    continue
                tmp = out_dir / f"_ipa_tmp_{vi}_{ri}_{pi}.wav"
                r = subprocess.run(
                    ["espeak-ng", "--ssml",
                     f"-v{voice}", f"-s{rate}", f"-p{pitch}",
                     "-w", str(tmp), ssml],
                    capture_output=True, timeout=15,
                )
                if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size < 500:
                    if tmp.exists():
                        tmp.unlink()
                    continue
                ok = ffcmd(["-i", str(tmp), "-ar", "16000", "-ac", "1", str(out_wav)])
                tmp.unlink()
                if ok and out_wav.exists() and out_wav.stat().st_size > CLIP_MIN_BYTES:
                    count += 1

    log_ok(f"  espeak-ng IPA: {count} clips")
    return count


# ══════════════════════════════════════════════════════════
#  TTS
# ══════════════════════════════════════════════════════════
def _generate_combo_negatives(neg_combos: list, neg_dir: Path, force: bool):
    """
    Generate TTS clips from negative word_parts combinations.
    Uses edge-tts (fast, no GPU) for quick generation.
    Also applies aggressive silence removal to further corrupt the clips.
    """
    import asyncio, tempfile
    prefix = "combo_neg"
    existing = sum(1 for e in os.scandir(neg_dir)
                   if e.name.startswith(prefix) and e.name.endswith(".wav"))
    n_expected = len(neg_combos) * 3  # 3 EN voices

    if not force and existing >= int(n_expected * 0.8):
        log_info(f"  Combo negatives: {existing} clips (skip)"); return

    log_step(f"  Combo negatives: {len(neg_combos)} phrases × 3 voices = {n_expected} clips")

    # Use 3 fast edge-tts voices for negative combos
    neg_voices = ["en-US-GuyNeural", "en-GB-RyanNeural", "en-AU-NatashaNeural"]
    py = py_bin()
    n = 0
    for vi, voice in enumerate(neg_voices):
        for ci, text in enumerate(neg_combos):
            out_wav = neg_dir / f"{prefix}_v{vi}_c{ci}.wav"
            if not force and out_wav.exists() and out_wav.stat().st_size > 500:
                n += 1; continue
            mp3_tmp = out_wav.with_suffix(".mp3")
            script = f"""
import asyncio, edge_tts
async def run():
    c = edge_tts.Communicate({repr(text)}, {repr(voice)})
    await c.save({repr(str(mp3_tmp))})
asyncio.run(run())
"""
            r = subprocess.run([py, "-c", script], capture_output=True, timeout=30)
            if mp3_tmp.exists() and mp3_tmp.stat().st_size > 500:
                ffcmd(["-i", str(mp3_tmp), "-ar", "16000", "-ac", "1", str(out_wav)])
                mp3_tmp.unlink()
                # Aggressive silence removal on negative clips
                _apply_aggressive_silence(out_wav)
                n += 1
    log_ok(f"  Combo negatives: {n} clips")


def _apply_aggressive_silence(wav_path: Path):
    """
    Apply aggressive silence removal to a clip.
    This corrupts the wake word by removing pauses between syllables,
    making it sound garbled — perfect for hard negatives.
    """
    tmp = wav_path.with_suffix(".sil.wav")
    try:
        ffcmd([
            "-i", str(wav_path),
            "-af", (
                "silenceremove="
                "start_periods=1:start_duration=0.01:start_threshold=-30dB"
                ":stop_periods=-1:stop_duration=0.05:stop_threshold=-30dB"
            ),
            str(tmp)
        ])
        if tmp.exists() and tmp.stat().st_size > 200:
            wav_path.unlink()
            tmp.rename(wav_path)
        elif tmp.exists():
            tmp.unlink()
    except Exception:
        if tmp.exists(): tmp.unlink()


def _tts_all(he_text: str, en_text: str, base: Path, force: bool,
             model_cfg: dict = None):
    py = py_bin()

    he_variants = _all_variants(he_text)
    en_variants = _all_variants(en_text)

    # Expand word_parts combinatorics if defined in model config
    if model_cfg:
        from forge.model_config import expand_word_parts
        pos_combos, neg_combos = expand_word_parts(model_cfg)
        if pos_combos:
            # Add to positive variants (deduplicated)
            all_pos = list(dict.fromkeys(en_variants + pos_combos))
            en_variants = all_pos
            log_info(f"  +{len(pos_combos)} positive combos from word_parts")
        if neg_combos:
            # Generate negative clips directly
            neg_dir = WORKSPACE / "negative_custom"
            neg_dir.mkdir(parents=True, exist_ok=True)
            _generate_combo_negatives(neg_combos, neg_dir, force)

    log_info(f"  Text variants: {len(he_variants)} HE, {len(en_variants)} EN")

    # edge-tts (online, multiple voices)
    for voice in EDGE_HE_VOICES:
        _tts_voice(py, he_variants, voice, "he", base, force)
    for voice in EDGE_EN_VOICES:
        _tts_voice(py, en_variants, voice, "en", base, force)

    # Piper ONNX voices (local, multi-speaker)
    _tts_piper_pt(en_variants, base, force)

    # Google Translate TTS (free, no API key)
    _tts_google(en_variants, base, "en", force)
    if he_text.strip():
        _tts_google(he_variants, base, "iw", force)  # iw = Hebrew in Google

    # Direct hard negatives — garbled voices → negative_custom/ directly
    _generate_direct_negatives(en_variants, force)

    # Wyoming Piper — local server with custom voices (trump, eminem, carlin, etc.)
    from .common import WYOMING_HOST, WYOMING_PORT
    if WYOMING_HOST and WYOMING_PORT:
        _tts_wyoming(en_variants, base, "en", WYOMING_HOST, WYOMING_PORT, force)
        if he_text.strip():
            _tts_wyoming(he_variants, base, "he", WYOMING_HOST, WYOMING_PORT, force)



def _tts_voice(py: str, texts: list, voice: str, lang: str,
               base: Path, force: bool):
    """Generate one TTS clip per text variant for this voice."""
    prefix = f"{lang}_{voice}"
    existing = sum(1 for e in os.scandir(base)
                   if e.name.startswith(prefix) and e.name.endswith(".wav"))
    expected = len(texts)
    if not force and existing >= expected:
        log_info(f"  {voice}: {existing} clips (skip)"); return

    mp3_tmp = base.parent / f"_mp3_{lang}_{voice}"
    mp3_tmp.mkdir(parents=True, exist_ok=True)

    script = f"""
import asyncio
from pathlib import Path
from tqdm import tqdm
import edge_tts

RATES   = {repr(EDGE_RATES)}
PITCHES = {repr(EDGE_PITCHES)}
texts   = {repr(texts)}
voice   = {repr(voice)}
out     = Path({repr(str(mp3_tmp))})

async def one(text, rate, pitch, path):
    try:
        c = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
        await c.save(str(path))
        return Path(path).exists() and Path(path).stat().st_size > 500
    except Exception:
        return False

async def main():
    ok = 0
    for i, text in enumerate(tqdm(texts, desc=voice)):
        p = out / f"et_{{i:04d}}.mp3"
        if p.exists() and p.stat().st_size > 500: ok += 1; continue
        rate   = RATES[i % len(RATES)]
        pitch  = PITCHES[i % len(PITCHES)]
        if await one(text, rate, pitch, p): ok += 1
    print(f"{{voice}}: {{ok}}/{{len(texts)}} clips")

asyncio.run(main())
"""
    _run_script(script)

    # Convert mp3 → wav (no silence trimming — preserves natural speech)
    for mp3 in mp3_tmp.glob("*.mp3"):
        wav = base / f"{prefix}_{mp3.stem}.wav"
        if not wav.exists():
            ffcmd(["-i", str(mp3), "-ar", "16000", "-ac", "1", str(wav)])

    delete_broken(base)
    n = sum(1 for e in os.scandir(base)
            if e.name.startswith(prefix) and e.name.endswith(".wav"))
    log_ok(f"  {voice}: {n} WAV files ({expected} variants × 1 voice)")


# ══════════════════════════════════════════════════════════
#  Piper ONNX voices  (English, multi-speaker where available)
# ══════════════════════════════════════════════════════════
# We use generate_samples_onnx — simpler, no phoneme_id_map needed.
# Voices are downloaded to workspace/piper_voices/
# Speaker diversity via length_scales × noise_scales cycling.
# ── POSITIVE voices — clean output → wav_base/ ──────────────────────────────
PIPER_ONNX_VOICES = [
    # ── en_US (American English) ──────────────────────────────────────────────
    ("en_US-libritts-high",          "en_US-libritts-high",          "en/en_US/libritts/high"),
    ("en_US-libritts_r-medium",      "en_US-libritts_r-medium",      "en/en_US/libritts_r/medium"),
    ("en_US-lessac-medium",          "en_US-lessac-medium",          "en/en_US/lessac/medium"),
    ("en_US-lessac-high",            "en_US-lessac-high",            "en/en_US/lessac/high"),
    ("en_US-joe-medium",             "en_US-joe-medium",             "en/en_US/joe/medium"),
    ("en_US-john-medium",            "en_US-john-medium",            "en/en_US/john/medium"),
    ("en_US-bryce-medium",           "en_US-bryce-medium",           "en/en_US/bryce/medium"),
    ("en_US-kusal-medium",           "en_US-kusal-medium",           "en/en_US/kusal/medium"),
    ("en_US-kristin-medium",         "en_US-kristin-medium",         "en/en_US/kristin/medium"),
    ("en_US-amy-medium",             "en_US-amy-medium",             "en/en_US/amy/medium"),
    ("en_US-amy-low",                "en_US-amy-low",                "en/en_US/amy/low"),
    ("en_US-danny-low",              "en_US-danny-low",              "en/en_US/danny/low"),
    ("en_US-kathleen-low",           "en_US-kathleen-low",           "en/en_US/kathleen/low"),
    ("en_US-arctic-medium",          "en_US-arctic-medium",          "en/en_US/arctic/medium"),
    ("en_US-ljspeech-medium",        "en_US-ljspeech-medium",        "en/en_US/ljspeech/medium"),
    ("en_US-ljspeech-high",          "en_US-ljspeech-high",          "en/en_US/ljspeech/high"),
    ("en_US-hfc_female-medium",      "en_US-hfc_female-medium",      "en/en_US/hfc_female/medium"),
    ("en_US-hfc_male-medium",        "en_US-hfc_male-medium",        "en/en_US/hfc_male/medium"),
    ("en_US-sam-medium",             "en_US-sam-medium",             "en/en_US/sam/medium"),
    ("en_US-reza_ibrahim-medium",    "en_US-reza_ibrahim-medium",    "en/en_US/reza_ibrahim/medium"),
    # ── en_GB (British English) ───────────────────────────────────────────────
    ("en_GB-alan-medium",            "en_GB-alan-medium",            "en/en_GB/alan/medium"),
    ("en_GB-alba-medium",            "en_GB-alba-medium",            "en/en_GB/alba/medium"),
    ("en_GB-aru-medium",             "en_GB-aru-medium",             "en/en_GB/aru/medium"),
    ("en_GB-cori-medium",            "en_GB-cori-medium",            "en/en_GB/cori/medium"),
    ("en_GB-cori-high",              "en_GB-cori-high",              "en/en_GB/cori/high"),
    ("en_GB-northern_english_male-medium", "en_GB-northern_english_male-medium", "en/en_GB/northern_english_male/medium"),
    ("en_GB-southern_english_female-low",  "en_GB-southern_english_female-low",  "en/en_GB/southern_english_female/low"),
    ("en_GB-vctk-medium",            "en_GB-vctk-medium",            "en/en_GB/vctk/medium"),
    # ── Community / celebrity voices (DIRECT download) ────────────────────────
    ("en_US-glados-high",       "en_US-glados-high",
     "DIRECT:https://huggingface.co/csukuangfj/vits-piper-en_US-glados-high/resolve/main/en_US-glados-high.onnx"),
    ("en_US-hal-medium",        "en_US-hal-medium",
     "DIRECT:https://huggingface.co/campwill/HAL-9000-Piper-TTS/resolve/main/hal.onnx"),
    ("en_US-bobby-medium",      "en_US-bobby-medium",
     "DIRECT:https://github.com/simoniz0r/piper-voice-models/releases/download/bobby/en_US-bobby-medium.onnx"),
    ("en_US-carl-medium",       "en_US-carl-medium",
     "DIRECT:https://github.com/simoniz0r/piper-voice-models/releases/download/carl/en_US-carl-medium.onnx"),
    ("en_US-eminem-medium",     "en_US-eminem-medium",
     "DIRECT:https://github.com/simoniz0r/piper-voice-models/releases/download/eminem/en_US-eminem-medium.onnx"),
    ("en_US-patrick-medium",    "en_US-patrick-medium",
     "DIRECT:https://github.com/simoniz0r/piper-voice-models/releases/download/patrick/en_US-patrick-medium.onnx"),
]
# ── NEGATIVE voices — garbled params → negative_custom/ DIRECTLY ─────────────
# We KNOW these settings produce stutters/corruption — no STT needed.
# norman: high noise_scale → lip smacking, stuttering
# jenny_dioco: certain speaker IDs → "jent swis" phoneme corruption
PIPER_NEG_VOICES_DIRECT = [
    # (stem, hf_path, noise_scales_for_neg, length_scales_for_neg)
    ("en_US-norman-medium",     "en/en_US/norman/medium",
     [1.2, 1.5, 1.8], [1.2, 1.5]),
    ("en_GB-jenny_dioco-medium","en/en_GB/jenny_dioco/medium",
     [1.1, 1.4], [1.0, 1.3]),
]
PIPER_VOICE_BLACKLIST: set = set()
HF_PIPER_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


# ══════════════════════════════════════════════════════════
#  Piper Voice QA — play one sample from each voice
# ══════════════════════════════════════════════════════════
def _qa_piper_voices(texts: list, base: Path):
    """
    Play one sample clip from each Piper voice so the user can
    blacklist voices with poor diction. Writes blacklist to
    workspace/piper_voice_blacklist.txt
    """
    blacklist_file = WORKSPACE / "piper_voice_blacklist.txt"
    if blacklist_file.exists():
        PIPER_VOICE_BLACKLIST.update(
            l.strip() for l in blacklist_file.read_text().splitlines() if l.strip()
        )
    voices_dir = WORKSPACE / "piper_voices"
    for _, stem, _ in PIPER_ONNX_VOICES:
        if stem in PIPER_VOICE_BLACKLIST:
            continue
        # Find one existing sample
        sample = next(
            (base / f for f in os.listdir(base)
             if f.startswith(f"piper_pt_{stem}") and f.endswith(".wav")),
            None
        )
        if sample is None:
            continue
        print(f"\n  Playing sample: {stem}")
        subprocess.run(["aplay", "-q", str(sample)], capture_output=True)
        ans = input(f"  Keep this voice? [Y/n]: ").strip().lower()
        if ans in ("n", "no"):
            PIPER_VOICE_BLACKLIST.add(stem)
            log_warn(f"  Blacklisted: {stem}")
    blacklist_file.write_text("\n".join(sorted(PIPER_VOICE_BLACKLIST)))
    if PIPER_VOICE_BLACKLIST:
        log_info(f"  Blacklisted voices: {', '.join(sorted(PIPER_VOICE_BLACKLIST))}")


# ══════════════════════════════════════════════════════════
#  Google Translate TTS  (free, no API key, 1 voice)
# ══════════════════════════════════════════════════════════
def _tts_google(texts: list, base: Path, lang: str, force: bool):
    """
    Download TTS clips from Google Translate (no API key needed).
    One clip per text variant. Used as a bonus voice.
    """
    try:
        import urllib.request, urllib.parse
    except ImportError:
        return

    prefix = f"google_{lang}"
    existing = sum(1 for e in os.scandir(base)
                   if e.name.startswith(prefix) and e.name.endswith(".wav"))
    if not force and existing >= len(texts):
        log_info(f"  Google TTS ({lang}): {existing} clips (skip)"); return

    log_step(f"  Google TTS ({lang}): {len(texts)} clips")
    n = 0
    for i, text in enumerate(texts):
        out_mp3 = base / f"{prefix}_{i}.mp3"
        out_wav = base / f"{prefix}_{i}.wav"
        if not force and out_wav.exists() and out_wav.stat().st_size > 500:
            n += 1; continue
        q = urllib.parse.quote(text)
        url = f"https://translate.google.com/translate_tts?ie=UTF-8&q={q}&tl={lang}&client=tw-ob"
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as r:
                out_mp3.write_bytes(r.read())
            if out_mp3.stat().st_size > 500:
                ffcmd(["-i", str(out_mp3), "-ar", "16000", "-ac", "1", str(out_wav)])
                out_mp3.unlink()
                n += 1
        except Exception as e:
            log_warn(f"  Google TTS failed for '{text}': {e}")
            if out_mp3.exists(): out_mp3.unlink()
    log_ok(f"  Google TTS ({lang}): {n} clips")


# ══════════════════════════════════════════════════════════
#  Wyoming Piper TTS  (local Docker, Hebrew + English)
# ══════════════════════════════════════════════════════════
# Wyoming voice names per language
WYOMING_EN_VOICES = ["eminem", "carlin", "trump", "rocket", "picard", "bobby", "carl", "patrick"]
WYOMING_HE_VOICES = ["shaul"]


def _tts_wyoming(texts: list, base: Path, lang_prefix: str,
                 host: str, port: int, force: bool):
    """
    Generate clips from a local Wyoming Piper server.
    Iterates over all available voices and generates clips per voice.
    Uses Wyoming TCP protocol with wyoming-client or falls back to HTTP.
    """
    import urllib.request, urllib.error, urllib.parse

    voices = WYOMING_HE_VOICES if lang_prefix == "he" else WYOMING_EN_VOICES

    # Check connectivity
    try:
        urllib.request.urlopen(f"http://{host}:{port}/api/tts?text=test&voice=test",
                               timeout=3)
    except urllib.error.HTTPError:
        pass
    except Exception as e:
        log_warn(f"  Wyoming ({lang_prefix}): not reachable at {host}:{port} — {e}")
        return

    total = 0
    for voice in voices:
        prefix = f"wyoming_{lang_prefix}_{voice}"
        existing = sum(1 for e in os.scandir(base)
                       if e.name.startswith(prefix) and e.name.endswith(".wav"))
        if not force and existing >= len(texts):
            log_info(f"  Wyoming {voice}: {existing} clips (skip)"); total += existing; continue

        log_step(f"  Wyoming {voice}: {len(texts)} clips")
        n = 0
        for i, text in enumerate(texts):
            out_wav = base / f"{prefix}_{i}.wav"
            if not force and out_wav.exists() and out_wav.stat().st_size > 500:
                n += 1; continue
            url = (f"http://{host}:{port}/api/tts"
                   f"?text={urllib.parse.quote(text)}&voice={voice}")
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=15) as r:
                    raw = r.read()
                if len(raw) < 100: continue
                tmp = out_wav.with_suffix(".raw.wav")
                tmp.write_bytes(raw)
                ffcmd(["-i", str(tmp), "-ar", "16000", "-ac", "1", str(out_wav)])
                if tmp.exists(): tmp.unlink()
                n += 1
            except Exception as e:
                pass  # voice may not exist on this server
        if n > 0:
            log_ok(f"  Wyoming {voice}: {n} clips")
        total += n
    log_ok(f"  Wyoming ({lang_prefix}): {total} clips total")


def _ensure_piper_voices(force: bool = False) -> list:
    """Download missing Piper ONNX voices. Returns list of existing .onnx paths."""
    voices_dir = WORKSPACE / "piper_voices"
    voices_dir.mkdir(parents=True, exist_ok=True)
    available = []
    for _, stem, hf_path in PIPER_ONNX_VOICES:
        onnx = voices_dir / f"{stem}.onnx"
        json_ = voices_dir / f"{stem}.onnx.json"
        if not force and onnx.exists() and json_.exists() and onnx.stat().st_size > 10000:
            available.append(onnx)
            continue
        log_step(f"  Downloading Piper voice: {stem}")
        ok = True
        import subprocess as _sp
        is_direct = hf_path.startswith("DIRECT:")
        for fname, dst in [(f"{stem}.onnx", onnx), (f"{stem}.onnx.json", json_)]:
            if is_direct:
                # Direct URL — onnx URL given, json URL = onnx_url + ".json"
                base_url = hf_path[7:]  # strip "DIRECT:"
                url = base_url if fname.endswith(".onnx") else base_url + ".json"
                # HAL has different json name
                if "hal" in stem and fname.endswith(".json"):
                    url = "https://huggingface.co/campwill/HAL-9000-Piper-TTS/resolve/main/hal.onnx.json"
                elif "glados" in stem and fname.endswith(".json"):
                    url = "https://huggingface.co/csukuangfj/vits-piper-en_US-glados-high/resolve/main/en_US-glados-high.onnx.json"
            else:
                url = f"{HF_PIPER_BASE}/{hf_path}/{fname}?download=true"
            import shutil as _sh
            _dl = ["wget", "-q", "-O"] if _sh.which("wget") else ["curl", "-sL", "-o"]
            r = _sp.run(_dl + [str(dst), url], capture_output=True)
            if r.returncode != 0 or not dst.exists() or dst.stat().st_size < 100:
                log_warn(f"  Failed: {fname}")
                ok = False; break
        if ok:
            log_ok(f"  Piper voice: {stem}")
            available.append(onnx)
    return available


def _tts_piper_pt(texts: list, base: Path, force: bool):
    """
    Generate English clips using Piper ONNX voices via generate_samples_onnx.
    Each voice gets its OWN subdirectory under base/piper/{voice_stem}/
    so the user can audit each voice separately and blacklist bad ones.

    Directory structure:
      wav_base/
        piper/
          en_US-libritts-high/  ← 247 speakers × texts × params
          en_US-lessac-medium/  ← 1 speaker × texts × params
          ...
    All clips are ALSO symlinked/copied to wav_base/ for training.
    """
    piper_dir = WORKSPACE / "piper-sample-generator"
    if not (piper_dir / "generate_samples.py").exists():
        log_info("  Piper: repo not found — skipping (run step 1)")
        return

    # Load blacklist
    blacklist_file = WORKSPACE / "piper_voice_blacklist.txt"
    blacklist = set()
    if blacklist_file.exists():
        blacklist.update(l.strip() for l in blacklist_file.read_text().splitlines() if l.strip())

    voices = _ensure_piper_voices(force)
    if not voices:
        log_warn("  Piper: no voices available — skipping")
        return

    # Root dir for organized voice output
    piper_root = base / "piper"
    piper_root.mkdir(exist_ok=True)

    prefix = "piper_pt"
    total_expected = sum(
        len(texts) * PIPER_SAMPLES_PER_TEXT
        for onnx in voices
        if onnx.stem not in blacklist
    )
    existing = sum(1 for e in os.scandir(base)
                   if e.name.startswith(prefix) and e.name.endswith(".wav"))
    if not force and existing >= int(total_expected * 0.9):
        log_info(f"  Piper: {existing} clips (skip)"); return

    log_step(f"  Piper: {len(voices)} voices × {len(texts)} texts × {PIPER_SAMPLES_PER_TEXT} clips")

    for vi, onnx in enumerate(voices):
        stem = onnx.stem
        if stem in blacklist:
            log_info(f"    {stem}: BLACKLISTED — skip")
            continue

        # Voice-specific subdir for auditing
        voice_dir = piper_root / stem
        voice_dir.mkdir(exist_ok=True)

        v_expected = len(texts) * PIPER_SAMPLES_PER_TEXT
        v_existing = sum(1 for e in os.scandir(voice_dir)
                         if e.name.endswith(".wav"))
        if not force and v_existing >= int(v_expected * 0.9):
            log_info(f"    {stem}: {v_existing} clips (skip)")
            # Still ensure clips are in base/
            _link_voice_clips(voice_dir, base, stem, vi)
            continue

        file_names = [f"{stem}_t{i}_s{j}.wav"
                      for i in range(len(texts))
                      for j in range(PIPER_SAMPLES_PER_TEXT)]

        script = f"""
import sys, logging
logging.disable(logging.WARNING)
sys.path.insert(0, {repr(str(piper_dir))})
from generate_samples import generate_samples_onnx
from pathlib import Path

generate_samples_onnx(
    text    = {repr(texts)},
    output_dir = {repr(str(voice_dir))},
    model   = [{repr(str(onnx))}],
    max_samples = {len(texts) * PIPER_SAMPLES_PER_TEXT},
    file_names  = {repr(file_names)},
    length_scales  = {repr(PIPER_LENGTH_SCALES)},
    noise_scales   = {repr(PIPER_NOISE_SCALES)},
    noise_scale_ws = {repr(PIPER_NOISE_SCALE_WS)},
    max_speakers   = {PIPER_MAX_SPEAKERS},
)
print("done")
"""
        _run_script(script)

        n = sum(1 for e in os.scandir(voice_dir) if e.name.endswith(".wav"))
        log_ok(f"    {stem}: {n} clips → {voice_dir.relative_to(base)}")

        # STT filter — duration+RMS+phonetic match
        _stt_filter_voice(voice_dir, stem, model_cfg=None)

        # Copy surviving clips to wav_base/
        _link_voice_clips(voice_dir, base, stem, vi)

    delete_broken(base)
    n = sum(1 for e in os.scandir(base)
            if e.name.startswith(prefix) and e.name.endswith(".wav"))
    log_ok(f"  Piper total: {n} clips in wav_base/")


def _stt_filter_voice(voice_dir: Path, stem: str, model_cfg: dict = None):
    """
    3-layer quality filter. Runs on per-voice dir (~1000 clips) BEFORE copying to wav_base.
    
    Layer 1 — Duration (fast, no AI):
      Too short (<0.5s) or too long (>6s) → hard negative immediately
    
    Layer 2 — RMS energy (fast, no AI):
      Too quiet → clip is silent/broken → hard negative
    
    Layer 3 — Whisper STT (only on clips that passed layers 1+2):
      Transcript vs positive/negative phrases → precise accept/reject
    """
    import tempfile, json as _json, subprocess as _sp
    neg_dir = WORKSPACE / "negative_custom"
    neg_dir.mkdir(parents=True, exist_ok=True)

    clips = [f for f in os.listdir(voice_dir) if f.endswith(".wav")]
    if not clips:
        return

    stt_cfg = {}
    if model_cfg:
        stt_cfg = {
            "positive_en": model_cfg.get("positive_en", []),
            "positive_he": model_cfg.get("positive_he", []),
            "negative_en": model_cfg.get("negative_en", []),
            "negative_he": model_cfg.get("negative_he", []),
            "stt_keywords": model_cfg.get("stt_keywords", []),
            "stt_model":    model_cfg.get("stt_model", "small.en"),
            "stt_device":   model_cfg.get("stt_device", "cuda"),
            "stt_enabled":  model_cfg.get("stt_enabled", True),
        }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                     delete=False, encoding="utf-8") as fh:
        _json.dump({"clips": clips, "voice_dir": str(voice_dir),
                    "neg_dir": str(neg_dir), "stem": stem,
                    "stt_cfg": stt_cfg}, fh, ensure_ascii=False)
        cfg_path = fh.name

    script = f"""
import sys, os, json, wave, struct, math, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

with open({repr(cfg_path)}, encoding="utf-8") as f:
    data = json.load(f)
clips     = data["clips"]
voice_dir = data["voice_dir"]
neg_dir   = data["neg_dir"]
stem      = data["stem"]
stt_cfg   = data["stt_cfg"]
os.unlink({repr(cfg_path)})

import sys
sys.path.insert(0, "/dev/null")  # dummy
MIN_BYTES = 20000
MAX_BYTES = 250000
MIN_DUR   = 0.5
MAX_DUR   = 4.5
MIN_RMS   = 0.008
MIN_BYTES = 20 * 1024
MAX_BYTES = 300 * 1024

def wav_duration(path):
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / w.getframerate()
    except: return 0

def wav_rms(path):
    try:
        with wave.open(path, "rb") as w:
            frames = w.readframes(w.getnframes())
            sampwidth = w.getsampwidth()
            if sampwidth == 2:
                samples = struct.unpack(f"{{len(frames)//2}}h", frames)
                rms = math.sqrt(sum(s*s for s in samples) / len(samples))
                return rms / 32768.0
            return 0.1  # assume ok for other formats
    except: return 0

# STT setup
positive_all = stt_cfg.get("positive_en", []) + stt_cfg.get("positive_he", [])
negative_all = stt_cfg.get("negative_en", []) + stt_cfg.get("negative_he", [])
keywords     = stt_cfg.get("stt_keywords", [])
stt_enabled  = stt_cfg.get("stt_enabled", True)

whisper_model = None
if stt_enabled:
    try:
        import whisper, torch
        device = stt_cfg.get("stt_device", "cuda")
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        whisper_model = whisper.load_model(stt_cfg.get("stt_model", "small.en"), device=device)
    except ImportError:
        stt_enabled = False

# Phonetic tokens — derived from the actual wake word keywords in stt_cfg.
# Fall back to individual words of length > 3 from the positive phrases.
_pos_phrases = stt_cfg.get("positive_en", []) + stt_cfg.get("positive_he", [])
_kw_set = set()
for _ph in _pos_phrases:
    for _w in _ph.lower().split():
        if len(_w) > 3:
            _kw_set.add(_w)
PHONETIC_TOKENS = stt_cfg.get("stt_keywords", sorted(_kw_set)) or sorted(_kw_set)

def stt_accepts(transcript):
    t = transcript.lower().strip()
    if not t: return False

    # 1. Forbidden words
    fw_list = stt_cfg.get("forbidden_words", ["just","meat","please","play","hate","page","sweet"])
    if any(fw in t.split() for fw in fw_list): return False

    # 2. Phonetic rules
    rules = stt_cfg.get("phonetic_rules", {{}})
    must_end = rules.get("must_end_with", [])
    if must_end:
        # Strip ALL punctuation before checking last word
        import re as _re
        clean = _re.sub(r"[^a-z ]", "", t).strip()
        last = clean.split()[-1] if clean.split() else ""
        if not any(last.endswith(s) for s in must_end): return False
    for seq in rules.get("forbidden_sequences", []):
        if seq in t: return False

    # 3. Phonetic token match
    all_tokens = stt_cfg.get("phonetic_tokens", PHONETIC_TOKENS)
    min_m = rules.get("min_token_matches", 1)
    if sum(1 for tok in all_tokens if tok in t) >= min_m: return True

    # 4. Keyword fallback
    for kw in keywords:
        if len(kw) >= 4 and kw[:4] in t: return True

    return False

kept = neg_dur = neg_rms = neg_stt = 0
sample_good = ""
sample_bad  = ""
for fname in clips:
    src = os.path.join(voice_dir, fname)
    if not os.path.exists(src): continue

    # Layer 1a: File size check (fast, no reading)
    fsize = os.path.getsize(src)
    if fsize < MIN_BYTES or fsize > MAX_BYTES:
        dst = os.path.join(neg_dir, f"hard_neg_{{stem}}_{{fname}}")
        os.rename(src, dst)
        neg_dur += 1
        continue

    # Layer 1b: Duration check
    dur = wav_duration(src)
    if dur < MIN_DUR or dur > MAX_DUR:
        dst = os.path.join(neg_dir, f"hard_neg_{{stem}}_{{fname}}")
        os.rename(src, dst)
        neg_dur += 1
        continue

    # Layer 2: RMS energy check
    rms = wav_rms(src)
    if rms < MIN_RMS:
        dst = os.path.join(neg_dir, f"hard_neg_{{stem}}_{{fname}}")
        os.rename(src, dst)
        neg_rms += 1
        continue

    # Layer 3: Whisper STT with confidence scoring
    if whisper_model:
        try:
            res = whisper_model.transcribe(src, language="en",
                                           fp16=(stt_cfg.get("stt_device","cuda")=="cuda"))
            t = res["text"].strip()
            # Get confidence score (avg_logprob: 0=perfect, -1=poor, <-1.5=garbage)
            segs = res.get("segments", [])
            avg_logp = sum(s.get("avg_logprob", -1) for s in segs) / max(len(segs), 1)

            if not stt_accepts(t):
                # Clearly wrong transcript
                dst = os.path.join(neg_dir, f"hard_neg_{{stem}}_{{fname}}")
                os.rename(src, dst)
                neg_stt += 1
                continue
            elif avg_logp < -0.8:
                # Borderline — low confidence even though transcript matches
                # Run second pass with no language hint for confirmation
                try:
                    res2 = whisper_model.transcribe(src, fp16=(stt_cfg.get("stt_device","cuda")=="cuda"))
                    t2 = res2["text"].strip()
                    if not stt_accepts(t2):
                        # Second pass also fails — reject
                        dst = os.path.join(neg_dir, f"hard_neg_{{stem}}_{{fname}}")
                        os.rename(src, dst)
                        neg_stt += 1
                        continue
                except Exception:
                    pass
        except Exception:
            pass  # on error, keep the clip

    kept += 1
    if not sample_good:
        try:
            res_s = whisper_model.transcribe(src, language="en", fp16=False) if whisper_model else None
            if res_s: sample_good = res_s["text"].strip()[:60]
        except: pass

total_neg = neg_dur + neg_rms + neg_stt
print(json.dumps({{"kept": kept, "neg_dur": neg_dur, "neg_rms": neg_rms, "neg_stt": neg_stt, "sample_good": sample_good}}))
"""
    r = _sp.run([py_bin(), "-c", script], capture_output=True, text=True, timeout=900)
    try:
        res = _json.loads([l for l in r.stdout.strip().splitlines()
                           if l.startswith("{")][-1])
        log_info(
            f"  Filter {stem}: ✓{res['kept']} kept | "
            f"dur×{res.get('neg_dur',0)} rms×{res.get('neg_rms',0)} "
            f"stt×{res.get('neg_stt',0)} → hard_neg"
        )
        if res.get("sample_good"):
            log_info(f"    ✓ Whisper heard: \"{res['sample_good']}\"")
    except Exception:
        if r.stdout: log_info(f"  Filter {stem}: {r.stdout.strip()[-150:]}")


def _link_voice_clips(voice_dir: Path, base: Path, stem: str, vi: int):
    """Copy clips from voice subdir to base/ with standardized prefix."""
    for f in os.listdir(voice_dir):
        if not f.endswith(".wav"): continue
        dst = base / f"piper_pt_v{vi}_{f}"
        if not dst.exists():
            import shutil as _shutil
            _shutil.copy2(str(voice_dir / f), str(dst))


# ══════════════════════════════════════════════════════════
#  SpeechT5 TTS  (HuggingFace, English, 7000+ speaker embeddings)
# ══════════════════════════════════════════════════════════
SPEECHT5_N_SPEAKERS = 100

# ══════════════════════════════════════════════════════════
#  Phonikud-TTS  (Hebrew, local Piper ONNX, GPU)
# ══════════════════════════════════════════════════════════
def _tts_phonikud(texts: list, base: Path, force: bool):
    """
    Generate Hebrew clips using a local Piper ONNX voice + phonikud diacritization.
    The user provides the path to their Hebrew ONNX model via PHONIKUD_ONNX_PATH in common.py.
    Uses PiperVoice.synthesize_wav directly — same API as English Piper voices.
    Multiple length_scales and noise_scales for variety.
    """
    from .common import PHONIKUD_ONNX_PATH, PHONIKUD_MODEL_PATH
    import importlib

    onnx_path = Path(PHONIKUD_ONNX_PATH) if PHONIKUD_ONNX_PATH else None
    if not onnx_path or not onnx_path.exists():
        log_info("  Phonikud-TTS: PHONIKUD_ONNX_PATH not set or file missing — skipping")
        log_info("  Set PHONIKUD_ONNX_PATH in forge/common.py to enable Hebrew TTS")
        return

    prefix = "phonikud"
    length_scales = [0.85, 1.0, 1.15, 1.3]
    noise_scales   = [0.667, 0.8, 1.0]
    n_expected = len(texts) * len(length_scales) * len(noise_scales)
    existing = sum(1 for e in os.scandir(base)
                   if e.name.startswith(prefix) and e.name.endswith(".wav"))
    if not force and existing >= int(n_expected * 0.9):
        log_info(f"  Phonikud-TTS: {existing} clips (skip)"); return

    log_step(f"  Phonikud-TTS: {len(texts)} texts × {len(length_scales)} speeds × {len(noise_scales)} noise = {n_expected} clips")

    script = f"""
import sys, warnings, wave
warnings.filterwarnings("ignore")

try:
    from piper.voice import PiperVoice, SynthesisConfig
except ImportError:
    try:
        from piper import PiperVoice
        SynthesisConfig = None
    except ImportError:
        print("piper-tts not installed")
        sys.exit(0)

try:
    from phonikud_onnx import Phonikud
    from phonikud import phonemize as phonikud_phonemize
    phonikud_model = Phonikud({repr(str(PHONIKUD_MODEL_PATH))}) if {repr(str(PHONIKUD_MODEL_PATH))} else None
except ImportError:
    phonikud_model = None

from pathlib import Path

onnx_path = {repr(str(onnx_path))}
base = Path({repr(str(base))})
texts = {repr(texts)}
length_scales = {repr(length_scales)}
noise_scales = {repr(noise_scales)}
prefix = "phonikud"

voice = PiperVoice.load(onnx_path, use_cuda=True)
n = 0
for ti, text in enumerate(texts):
    # Apply diacritization if phonikud available
    if phonikud_model:
        try:
            text = phonikud_model.add_diacritics(text)
        except Exception:
            pass
    for li, ls in enumerate(length_scales):
        for ni, ns in enumerate(noise_scales):
            out = base / f"{{prefix}}_t{{ti}}_l{{li}}_n{{ni}}.wav"
            if out.exists() and out.stat().st_size > 500:
                n += 1; continue
            try:
                with wave.open(str(out), "wb") as wf:
                    if SynthesisConfig:
                        cfg = SynthesisConfig(length_scale=ls, noise_scale=ns)
                        voice.synthesize_wav(text, wf, syn_config=cfg)
                    else:
                        voice.synthesize(text, wf)
                n += 1
            except Exception as e:
                print(f"  skip t{{ti}} l{{li}} n{{ni}}: {{e}}")
print(f"done: {{n}} clips")
"""
    _run_script(script)
    n = sum(1 for e in os.scandir(base)
            if e.name.startswith(prefix) and e.name.endswith(".wav"))
    log_ok(f"  Phonikud-TTS: {n} clips")


def _tts_speecht5(texts: list, base: Path, force: bool):
    """
    Generate English clips using Microsoft SpeechT5 + CMU ARCTIC xvectors.
    100 different speaker embeddings × len(texts) clips.
    Requires: pip install transformers soundfile
    """
    prefix = "speecht5"
    n_expected = len(texts) * SPEECHT5_N_SPEAKERS
    existing = sum(1 for e in os.scandir(base)
                   if e.name.startswith(prefix) and e.name.endswith(".wav"))
    if not force and existing >= int(n_expected * 0.9):
        log_info(f"  SpeechT5: {existing} clips (skip)"); return

    log_step(f"  SpeechT5: {SPEECHT5_N_SPEAKERS} speakers × {len(texts)} texts = {n_expected} clips")

    script = f"""
import sys, os, warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import torch

try:
    from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
    from datasets import load_dataset
    import soundfile as sf
except ImportError:
    print("transformers/soundfile not installed - skipping SpeechT5")
    print("Install: pip install transformers soundfile")
    sys.exit(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"SpeechT5 on {{device}}")

base = {repr(str(base))}
texts = {repr(texts)}
n_speakers = {SPEECHT5_N_SPEAKERS}
prefix = "speecht5"

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

emb_ds = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
total_emb = len(emb_ds)
indices = [int(i * total_emb / n_speakers) for i in range(n_speakers)]

n = 0
for si, emb_idx in enumerate(indices):
    spk_emb = torch.tensor(emb_ds[emb_idx]["xvector"]).unsqueeze(0).to(device)
    for ti, text in enumerate(texts):
        out = f"{{base}}/{{prefix}}_s{{si}}_t{{ti}}.wav"
        if os.path.exists(out) and os.path.getsize(out) > 500:
            n += 1; continue
        try:
            inputs = processor(text=text, return_tensors="pt").to(device)
            with torch.no_grad():
                speech = model.generate_speech(inputs["input_ids"], spk_emb, vocoder=vocoder)
            sf.write(out, speech.cpu().numpy(), samplerate=16000)
            n += 1
        except Exception as e:
            print(f"  skip s{{si}} t{{ti}}: {{e}}")

print(f"done: {{n}} clips")
"""
    _run_script(script)
    n = sum(1 for e in os.scandir(base)
            if e.name.startswith(prefix) and e.name.endswith(".wav"))
    log_ok(f"  SpeechT5: {n} clips")


# ══════════════════════════════════════════════════════════
#  Direct Hard Negative Generation
# ══════════════════════════════════════════════════════════
def _generate_direct_negatives(texts: list, force: bool, model_cfg: dict = None):
    """
    Generate hard negative clips DIRECTLY to negative_custom/ using
    voices and params known to produce garbled/stuttered output.
    Reads voice config from model YAML (tts.piper_negative.voices) if available,
    falls back to PIPER_NEG_VOICES_DIRECT from common.py.
    """
    from .common import PIPER_NEG_VOICES_DIRECT, PIPER_NEG_PHRASES

    # Load voice list from YAML if available
    yaml_voices = []
    if model_cfg:
        for v in model_cfg.get("tts", {}).get("piper_negative", {}).get("voices", []):
            stem = v.get("stem", "")
            ns   = v.get("noise_scales", [1.2])
            ls   = v.get("length_scales", [1.2])
            # Look up hf_path from PIPER_NEG_VOICES_DIRECT or PIPER_ONNX_VOICES
            hf_path = next(
                (h for _, s, h in PIPER_NEG_VOICES_DIRECT if s == stem),
                next((h for _, s, h in PIPER_ONNX_VOICES if s == stem), None)
            )
            if hf_path:
                yaml_voices.append((stem, hf_path, ns, ls))
    neg_voices = yaml_voices if yaml_voices else PIPER_NEG_VOICES_DIRECT

    piper_dir  = WORKSPACE / "piper-sample-generator"
    voices_dir = WORKSPACE / "piper_voices"
    neg_dir    = WORKSPACE / "negative_custom"
    neg_dir.mkdir(parents=True, exist_ok=True)

    if not (piper_dir / "generate_samples.py").exists():
        return

    # Use the actual target texts (garbled) + dedicated negative phrases
    all_neg_texts = list(texts) + list(PIPER_NEG_PHRASES)
    prefix = "piper_neg"

    for stem, hf_path, noise_scales, length_scales in neg_voices:
        onnx  = voices_dir / f"{stem}.onnx"
        json_ = voices_dir / f"{stem}.onnx.json"

        # Download voice if missing
        if not onnx.exists() or onnx.stat().st_size < 10000:
            log_step(f"  Downloading neg voice: {stem}")
            ok = True
            for fname, dst in [(f"{stem}.onnx", onnx), (f"{stem}.onnx.json", json_)]:
                import subprocess as _sp, shutil as _sh2
                _dl = ["wget", "-q", "-O"] if _sh2.which("wget") else ["curl", "-sL", "-o"]
                r = _sp.run(_dl + [str(dst), f"{HF_PIPER_BASE}/{hf_path}/{fname}?download=true"],
                            capture_output=True)
                if r.returncode != 0 or dst.stat().st_size < 100:
                    log_warn(f"  Failed: {fname}"); ok = False; break
            if not ok:
                continue

        n_expected = len(all_neg_texts) * len(noise_scales) * len(length_scales)
        v_prefix = f"{prefix}_{stem}"
        existing = sum(1 for e in os.scandir(neg_dir)
                       if e.name.startswith(v_prefix) and e.name.endswith(".wav"))
        if not force and existing >= int(n_expected * 0.8):
            log_info(f"  Neg {stem}: {existing} clips (skip)"); continue

        log_step(f"  Neg {stem}: {n_expected} garbled clips → negative_custom/")
        fnames = [f"{v_prefix}_n{ni}_l{li}_t{ti}.wav"
                  for ni in range(len(noise_scales))
                  for li in range(len(length_scales))
                  for ti in range(len(all_neg_texts))]
        all_texts_expanded = all_neg_texts * (len(noise_scales) * len(length_scales))

        script = f"""
import sys, logging
logging.disable(logging.WARNING)
sys.path.insert(0, {repr(str(piper_dir))})
from generate_samples import generate_samples_onnx
from pathlib import Path
from itertools import product as iproduct

fnames = []
texts_out = []
for ni, ns in enumerate({repr(noise_scales)}):
    for li, ls in enumerate({repr(length_scales)}):
        for ti, txt in enumerate({repr(all_neg_texts)}):
            fnames.append(f"{v_prefix}_n{{ni}}_l{{li}}_t{{ti}}.wav")
            texts_out.append(txt)

generate_samples_onnx(
    text       = texts_out,
    output_dir = {repr(str(neg_dir))},
    model      = [{repr(str(onnx))}],
    max_samples = len(texts_out),
    file_names  = fnames,
    length_scales  = {repr(length_scales)},
    noise_scales   = {repr(noise_scales)},
    noise_scale_ws = [0.9, 1.1],
    max_speakers   = 30,
)
print("done")
"""
        _run_script(script)
        n = sum(1 for e in os.scandir(neg_dir)
                if e.name.startswith(v_prefix) and e.name.endswith(".wav"))
        log_ok(f"  Neg {stem}: {n} hard negative clips")


def _run_script(code: str) -> int:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                     delete=False, encoding="utf-8") as fh:
        fh.write(code); tmp = fh.name
    try:
        return subprocess.run([py_bin(), tmp]).returncode
    finally:
        try: os.unlink(tmp)
        except Exception: pass


# ══════════════════════════════════════════════════════════
#  Silence trimming
# ══════════════════════════════════════════════════════════
def _trim_silence(wav_path: Path) -> bool:
    """
    Remove silence from a WAV clip using ffmpeg silenceremove.
    Settings match Audacity "Truncate Silence":
      - threshold: -80 dB
      - min silence duration: 0.1s
      - leave: 0.1s of silence after trimming
    Replaces the file in-place. Returns True if file still valid after trim.
    """
    tmp = wav_path.with_suffix(".trimtmp.wav")
    try:
        # silenceremove: remove leading silence, then trailing silence
        # 1:0.1:-80dB = start_periods:start_duration:start_threshold
        ffcmd([
            "-i", str(wav_path),
            "-af", (
                "silenceremove="
                "start_periods=1:start_duration=0.05:start_threshold=-20dB"
                ":stop_periods=-1:stop_duration=0.1:stop_threshold=-20dB"
                ",apad=pad_dur=0.1"  # -20dB threshold, 0.1s pad
            ),
            "-ar", "16000", "-ac", "1",
            str(tmp)
        ])
        if tmp.exists() and tmp.stat().st_size > 500:
            wav_path.unlink()
            tmp.rename(wav_path)
            return True
        else:
            if tmp.exists(): tmp.unlink()
            return wav_path.exists()
    except Exception:
        if tmp.exists(): tmp.unlink()
        return wav_path.exists()


# ══════════════════════════════════════════════════════════
#  Custom positive recordings
# ══════════════════════════════════════════════════════════
def _import_personal(base: Path):
    """
    Import personal recordings from my_recordings/ (project root).
    Files are converted to 16kHz mono WAV and named with 'my_rec_' prefix
    so _augment() can identify them and apply intensive augmentation.
    """
    dirs_to_check = [MY_RECORDINGS_DIR]
    # Also check nested subdirs (speaker01/, speaker02/, etc.)
    if MY_RECORDINGS_DIR.exists():
        for sub in MY_RECORDINGS_DIR.iterdir():
            if sub.is_dir():
                dirs_to_check.append(sub)

    n = 0
    for d in dirs_to_check:
        if not d.exists():
            continue
        for ext in ("*.wav", "*.mp3", "*.ogg", "*.flac", "*.m4a"):
            for f in d.glob(ext):
                dst = base / f"my_rec_{f.stem}.wav"
                if not dst.exists():
                    if ffcmd(["-i", str(f), "-ar", "16000", "-ac", "1",
                              "-sample_fmt", "s16", str(dst)]):
                        n += 1
    if n:
        log_ok(f"Personal recordings: {n} imported from my_recordings/")
    elif not any(base.glob("my_rec_*.wav")):
        log_warn("my_recordings/ is empty — add your own recordings for best results")


def _tts_minimal(he_text: str, en_text: str, base: Path,
                 quota: int, force: bool, use_all_voices: bool = False):
    """
    Generate a small number of TTS clips (quota total).
    Uses only fast edge-tts voices — no Piper, no Google, no Wyoming.

    use_all_voices=False (default, personal-recording mode):
        2 EN voices + 1 HE voice.  Clips get 'tts_' prefix → light augmentation.

    use_all_voices=True (TTS-only mode, no personal recordings):
        ALL available EN + HE voices.  Clips get NO 'tts_' prefix so _augment()
        treats them as personal recordings and applies intensive augmentation.
    """
    py = py_bin()
    en_variants = _all_variants(en_text) if en_text.strip() else []
    he_variants = _all_variants(he_text) if he_text.strip() else []

    # Pick voices based on mode
    if use_all_voices:
        # TTS-only: use every available voice for maximum diversity
        voices = []
        if en_variants:
            voices += [(v, "en", en_variants) for v in EDGE_EN_VOICES]
        if he_variants and he_text.strip():
            voices += [(v, "he", he_variants) for v in EDGE_HE_VOICES]
    else:
        # Minimal: 2 EN + 1 HE
        voices = []
        if en_variants:
            voices += [(EDGE_EN_VOICES[0], "en", en_variants),
                       (EDGE_EN_VOICES[2], "en", en_variants)]  # AriaNeural + JennyNeural
        if he_variants and he_text.strip():
            voices += [(EDGE_HE_VOICES[0], "he", he_variants)]

    if not voices:
        return

    clips_per_voice = max(1, quota // len(voices))
    log_info(f"  TTS: {len(voices)} voices × {clips_per_voice} clips each")

    for voice, lang, variants in voices:
        # Use the first clips_per_voice variants (diverse punctuation)
        selected = variants[:clips_per_voice]

        # In TTS-only mode: no 'tts_' prefix → _augment() applies intensive aug
        # In normal mode:   'tts_' prefix   → _augment() applies light aug
        prefix = f"{lang}_{voice}" if use_all_voices else f"tts_{lang}_{voice}"
        existing = sum(1 for e in os.scandir(base)
                       if e.name.startswith(prefix) and e.name.endswith(".wav"))
        if not force and existing >= len(selected):
            log_info(f"  {voice}: {existing} clips (skip)"); continue

        mp3_tmp = base.parent / f"_mp3_tts_{lang}_{voice}"
        mp3_tmp.mkdir(parents=True, exist_ok=True)

        script = f"""
import asyncio, edge_tts
texts  = {repr(selected)}
voice  = {repr(voice)}
out    = {repr(str(mp3_tmp))}
RATES  = {repr(EDGE_RATES)}
PITCHES = {repr(EDGE_PITCHES)}

async def main():
    ok = 0
    for i, text in enumerate(texts):
        p = __import__("pathlib").Path(out) / f"et_{{i:04d}}.mp3"
        if p.exists() and p.stat().st_size > 500:
            ok += 1; continue
        rate  = RATES[i % len(RATES)]
        pitch = PITCHES[i % len(PITCHES)]
        try:
            c = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
            await c.save(str(p))
            ok += 1
        except Exception:
            pass
    print(f"{{voice}}: {{ok}}/{{len(texts)}}")

asyncio.run(main())
"""
        _run_script(script)

        n = 0
        for mp3 in mp3_tmp.glob("*.mp3"):
            wav = base / f"{prefix}_{mp3.stem}.wav"
            if not wav.exists():
                if ffcmd(["-i", str(mp3), "-ar", "16000", "-ac", "1", str(wav)]):
                    n += 1
            mp3.unlink(missing_ok=True)
        try:
            mp3_tmp.rmdir()
        except Exception:
            pass
        if n:
            log_ok(f"  {voice}: {n} TTS clips")

    delete_broken(base)


def _import_custom(base: Path):
    CUSTOM_POS_DIR.mkdir(parents=True, exist_ok=True)
    files = (list(CUSTOM_POS_DIR.glob("*.wav")) +
             list(CUSTOM_POS_DIR.glob("*.mp3")))
    if not files:
        log_info("positive_custom/: empty (add your own recordings here)")
        return

    log_step(f"Importing {len(files)} custom recordings")
    n = 0
    for f in files:
        if f.suffix.lower() == ".mp3":
            dst = base / f"custom_{f.stem}.wav"
            if not dst.exists():
                ffcmd(["-i", str(f), "-ar", "16000", "-ac", "1", str(dst)])
                n += 1
        else:
            # WAV files: always re-encode to 16000Hz mono
            # (never copy as-is — OWW crashes on wrong sample rate)
            # NOTE: no silence trimming on custom recordings — user's recordings
            # are already clean and trimming may cut leading/trailing phonemes
            dst = base / f"custom_{f.stem}.wav"
            if not dst.exists():
                ffcmd(["-i", str(f), "-ar", "16000", "-ac", "1", str(dst)])
                n += 1

    log_ok(f"Custom: {n} imported (base total: {fcount(base)})")


# ══════════════════════════════════════════════════════════
#  Augmentation
# ══════════════════════════════════════════════════════════
def _augment(base: Path, aug: Path, n_target: int):
    """
    Intensive augmentation strategy:
    - Personal recordings (my_rec_* / custom_*): heavy aug — tempo × volume × pitch × EQ
    - TTS clips (tts_*): light aug — tempo × volume only
    - Target split: 75% personal, 25% TTS (or 100% personal if no TTS clips)
    """
    aug.mkdir(parents=True, exist_ok=True)

    all_wavs = [Path(e.path) for e in os.scandir(base) if e.name.endswith(".wav")]
    if not all_wavs:
        log_warn("No WAVs to augment"); return

    # Separate personal from TTS by prefix
    personal = [w for w in all_wavs
                if w.name.startswith("my_rec_") or w.name.startswith("custom_")]
    tts = [w for w in all_wavs if w not in set(personal)]

    if not personal:
        # Fallback: no prefix distinction — treat all as personal
        personal = all_wavs
        tts = []

    existing = sum(1 for e in os.scandir(aug) if e.name.endswith(".wav"))
    remaining = n_target - existing
    if remaining <= 0:
        log_info(f"  Augmentation already complete: {existing:,}")
        return

    # Allocate budget
    if tts:
        n_pers = int(remaining * 0.75)
        n_tts  = remaining - n_pers
    else:
        n_pers = remaining
        n_tts  = 0

    log_step(f"Augmentation: {n_pers:,} personal + {n_tts:,} TTS = {remaining:,} total")
    log_info(f"  Sources: {len(personal)} personal, {len(tts)} TTS base clips")

    try:
        from tqdm import tqdm as _tqdm
        bar = _tqdm(total=remaining, desc="Augmenting", unit="clip")
    except ImportError:
        bar = None

    done = 0

    # ── Personal recordings: intensive augmentation ───────────────────────────
    # For each clip: randomly sample (tempo, volume) then optionally add pitch / EQ
    idx = existing
    _rnd = random
    while done < n_pers:
        src = _rnd.choice(personal)
        tempo  = _rnd.choice(PERSONAL_TEMPOS)
        vol    = _rnd.choice(PERSONAL_VOLUMES)
        pitch  = _rnd.choice(PERSONAL_PITCHES)  # cents
        eq     = _rnd.choice(PERSONAL_EQ)

        dst = aug / f"aug_p_{idx:08d}.wav"
        idx += 1
        if dst.exists() and dst.stat().st_size > 500:
            done += 1
            if bar: bar.update(1)
            continue

        # Build ffmpeg filter chain
        fparts = [f"atempo={tempo}", f"volume={vol}"]
        if pitch != 0:
            factor = 2 ** (pitch / 1200)
            new_rate = int(16000 * factor)
            correction = 1.0 / factor
            fparts += [f"asetrate={new_rate}",
                       f"atempo={correction:.5f}",
                       "aresample=16000"]
        if eq:
            fparts.append(eq)

        if ffcmd(["-i", str(src), "-af", ",".join(fparts),
                  "-ar", "16000", "-ac", "1", str(dst)]):
            done += 1
            if bar: bar.update(1)

    # ── TTS clips: light augmentation (tempo × volume only) ──────────────────
    tts_done = 0
    while tts and tts_done < n_tts:
        src = _rnd.choice(tts)
        vol_str = _rnd.choice(AUG_VOLUMES)
        tem_str = _rnd.choice(AUG_TEMPOS)

        dst = aug / f"aug_t_{idx:08d}.wav"
        idx += 1
        if dst.exists() and dst.stat().st_size > 500:
            tts_done += 1
            if bar: bar.update(1)
            continue
        if ffcmd(["-i", str(src), "-af", f"{vol_str},{tem_str}",
                  "-ar", "16000", "-ac", "1", str(dst)]):
            tts_done += 1
            if bar: bar.update(1)

    if bar: bar.close()
    total = sum(1 for e in os.scandir(aug) if e.name.endswith(".wav"))
    log_ok(f"Augmentation done: {total:,} files ({done} personal + {tts_done} TTS)")
