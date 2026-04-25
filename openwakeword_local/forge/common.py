"""
forge/common.py
Shared constants, logging, state, path helpers.
Every other module imports only from here.
"""
from __future__ import annotations
import json, os, re, shutil, subprocess, sys
from datetime import datetime
from pathlib import Path

# ═══════════════════════════════════════════════════════
#  Project paths
# ═══════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE    = PROJECT_ROOT / "workspace"
STATE_FILE   = PROJECT_ROOT / "run_state.json"
LOG_DIR      = WORKSPACE / "logs"

# Per-model directories
def model_dir(name: str) -> Path:  return WORKSPACE / "models" / name
def wav_base(name: str)  -> Path:  return model_dir(name) / "wav_base"
def wav_aug(name: str)   -> Path:  return model_dir(name) / "wav_aug"
def npy_dir(name: str)   -> Path:  return model_dir(name) / "features"

# Shared negative features (model-independent, built once)
SHARED_NEG_TRAIN = WORKSPACE / "shared_neg_train.npy"
SHARED_NEG_TEST  = WORKSPACE / "shared_neg_test.npy"

# User custom recordings
CUSTOM_POS_DIR   = WORKSPACE / "positive_custom"
CUSTOM_NEG_DIR   = WORKSPACE / "negative_custom"
# Personal recordings folder at project root (my_recordings/)
MY_RECORDINGS_DIR = PROJECT_ROOT / "my_recordings"

# ═══════════════════════════════════════════════════════
#  Remote URLs
# ═══════════════════════════════════════════════════════
OWW_REPO     = "https://github.com/dscripka/openWakeWord.git"
PIPER_REPO   = "https://github.com/rhasspy/piper-sample-generator.git"

# Piper .pt generator (LibriTTS-R, 904 English speakers, SLERP mixing)
# This is what the OWW notebook uses — NOT the ONNX voices
# Downloaded to workspace/piper-sample-generator/models/
PIPER_PT_MODEL_URL  = "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt"
PIPER_PT_MODEL_NAME = "en_US-libritts_r-medium.pt"
PIPER_PT_JSON_URL   = "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt.json"

# Generation parameters
# Key insight: speaker diversity comes from batch_size and max_samples, NOT parameter combos.
# Each batch advances speakers_iter by batch_size steps through product(range(N), range(N)).
# So: more samples = more unique speaker pairs = more voice diversity.
# Keep parameter combos small — let speaker cycling do the heavy lifting.
PIPER_LENGTH_SCALES  = (0.85, 1.0, 1.15)   # 3 speeds
PIPER_NOISE_SCALES   = (0.667, 0.9)         # 2 noise levels
PIPER_NOISE_SCALE_WS = (0.8,)               # fixed
PIPER_SLERP_WEIGHTS  = (0.5,)               # fixed midpoint — speaker variety via cycling
PIPER_MAX_SPEAKERS   = 200                  # cap at 200 (later speakers have artifacts)
PIPER_SAMPLES_PER_TEXT = 50

# Hard negative generation params — these settings produce garbled/stuttered output
# which is PERFECT for negative training examples.
# norman + high noise + slow speed = stutters and lip-smacking sounds
# jenny_dioco + certain speakers = "hey jent swis" type corruptions
PIPER_NEG_VOICES = [
    # voice stem, noise_scale (high=garbled), length_scale (slow=weird)
    ("en_US-norman-medium",    [1.2, 1.4], [1.3, 1.5]),   # stutters/lisps
    ("en_GB-jenny_dioco-medium", [1.0, 1.2], [1.0, 1.2]), # "jent swis" type
]
PIPER_NEG_PHRASES = [
    # Generic confusable utterances — step5 adds wake-word-specific variants
    # These teach the model to reject near-miss phonemes and prefix-less triggers
    "hey",
    "hey hey",
    "okay",
    "okay computer",
    "hey stop",
    "play music",
    "turn on the lights",
]
# Voices + params known to produce garbled output → go directly to negative_custom
# (stem, hf_path, noise_scales, length_scales)
PIPER_NEG_VOICES_DIRECT = [
    ("en_US-norman-medium",     "en/en_US/norman/medium",
     [1.2, 1.5, 1.8], [1.2, 1.5]),
    ("en_GB-jenny_dioco-medium","en/en_GB/jenny_dioco/medium",
     [1.1, 1.4], [1.0, 1.3]),
]                 # how many clips per text variant from Piper

# Wyoming Piper local server (Docker)
# Set WYOMING_HOST to the IP of your Home Assistant / Docker host
# Set WYOMING_PORT to the Wyoming HTTP port (default 10200)
WYOMING_HOST = "localhost"   # change to your HA/Docker IP e.g. "192.168.1.100"
WYOMING_PORT = 11400         # your voice hub port

# Phonikud-TTS Hebrew voice (local Piper ONNX)
# Set PHONIKUD_ONNX_PATH to your Hebrew ONNX model file
# e.g. "/path/to/he_IL-your-voice.onnx"
# Set PHONIKUD_MODEL_PATH to your phonikud-1.0.int8.onnx diacritization model
# Leave as "" to skip phonikud diacritization (still works with raw Hebrew text)
PHONIKUD_ONNX_PATH  = ""   # e.g. "/path/to/your/he_IL-ohr-medium.onnx"
PHONIKUD_MODEL_PATH = ""   # e.g. "/path/to/your/phonikud-1.0.int8.onnx"                 # how many clips per text variant from Piper
                                            # 50 samples × batch_size=32 = ~1.5 batches per text
                                            # = ~48 unique speaker pairs per text
ACAV_URL     = ("https://huggingface.co/datasets/davidscripka/openwakeword_features"
                "/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy")
VAL_URL      = ("https://huggingface.co/datasets/davidscripka/openwakeword_features"
                "/resolve/main/validation_set_features.npy")
AUDIOSET_URL = ("https://huggingface.co/datasets/agkphysics/AudioSet/resolve/"
                "196c0900867eff791b8f4d4be57db277e9a5b131/bal_train09.tar")
MIT_RIR_REPO = ("https://huggingface.co/datasets/davidscripka/"
                "MIT_environmental_impulse_responses")

# ═══════════════════════════════════════════════════════
#  Training defaults
# ═══════════════════════════════════════════════════════
DEFAULT_N_SAMPLES = 100_000
DEFAULT_N_STEPS   = 100_000
DEFAULT_PENALTY   = 5_000
TARGET_ACCURACY   = 0.5        # OWW early-stop threshold — don't raise, let it train fully
TARGET_RECALL     = 0.25       # same — actual recall is much higher in real usage
TTS_PER_VOICE     = 5          # base clips per TTS voice (augmentation multiplies)
HOME_NOISE_CLIP_S = 3.0        # seconds per noise clip (shorter = more variety)

# ── Hard clip quality thresholds ─────────────────────────────────────────────
CLIP_MIN_KB   = 20     # smaller = silence/broken
CLIP_MAX_KB   = 300    # larger = too long/garbled (wake word at 16kHz mono WAV)
CLIP_MIN_SEC  = 0.5    # seconds — shorter = no speech
CLIP_MAX_SEC  = 4.5    # seconds — longer = stutter (wake word is 1.5-3s)
CLIP_MIN_RMS  = 0.008  # normalized RMS — quieter = silent clip

# ── Clip quality thresholds ────────────────────────────────
# Applied during STT filter before copying to wav_base
CLIP_MIN_BYTES  = 20_000       # ~0.6s at 16kHz 16-bit — shorter = silence/broken
CLIP_MAX_BYTES  = 250_000      # ~8s — longer = stutter/corruption
CLIP_MIN_DUR    = 1.0          # seconds
CLIP_MAX_DUR    = 5.0          # seconds

EDGE_HE_VOICES = ["he-IL-AvriNeural", "he-IL-HilaNeural"]
EDGE_EN_VOICES = [
    "en-US-AriaNeural", "en-US-GuyNeural",      "en-US-JennyNeural",
    "en-US-EricNeural", "en-US-ChristopherNeural",
    "en-GB-SoniaNeural","en-GB-RyanNeural",
    "en-AU-NatashaNeural","en-CA-ClaraNeural",  "en-IN-NeerjaNeural",
]
# edge-tts: minimum variants — we only need it for Hebrew (no local HE engine)
# Just 2 rates × 2 pitches = 4 combos max, faster is better
EDGE_RATES   = ["+0%", "-15%"]
EDGE_PITCHES = ["+0Hz", "-5Hz"]

# Augmentation: 3 volumes × 9 tempos = 27 combos per base file (TTS clips, fallback)
AUG_VOLUMES = ["volume=0.5", "volume=1.0", "volume=2.0"]
AUG_TEMPOS  = [
    "atempo=0.75", "atempo=0.85", "atempo=0.93",
    "atempo=1.0",
    "atempo=1.1",  "atempo=1.2",  "atempo=1.3",
    "atempo=1.5",  "atempo=1.75",
]

# ── Intensive augmentation for personal recordings (from test2.py) ──────────
# 71 tempo values × 18 volume values = 1,278 base combos per recording
# + 13 pitch values + 8 EQ presets = thousands of unique augmentations
PERSONAL_TEMPOS  = [round(0.70 + i * 0.01, 2) for i in range(71)]   # 0.70 → 1.40
PERSONAL_VOLUMES = [round(0.3  + i * 0.1,  2) for i in range(18)]   # 0.3 → 2.0
PERSONAL_PITCHES = list(range(-300, 350, 50))  # -300 to +300 cents (13 steps)
PERSONAL_EQ = [
    None,
    "bass=gain=4",                     # warm / bassy
    "bass=gain=-4",                    # thin
    "treble=gain=3",                   # bright
    "treble=gain=-3",                  # dark
    "bass=gain=3,treble=gain=2",       # warm + bright
    "bass=gain=-2,treble=gain=3",      # crisp
    "bass=gain=5,treble=gain=-2",      # heavy bass
]

# ═══════════════════════════════════════════════════════
#  ANSI colours
# ═══════════════════════════════════════════════════════
class C:
    RESET  = "\033[0m";  BOLD   = "\033[1m";  DIM    = "\033[2m"
    RED    = "\033[91m"; GREEN  = "\033[92m"; YELLOW = "\033[93m"
    BLUE   = "\033[94m"; CYAN   = "\033[96m"; WHITE  = "\033[97m"

    @staticmethod
    def strip(t: str) -> str:
        return re.sub(r"\033\[[0-9;]*m", "", t)

# Windows: enable VT sequences; fall back to plain text if not supported
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleMode(
            ctypes.windll.kernel32.GetStdHandle(-11), 7)
    except Exception:
        for _a in ("RESET","BOLD","DIM","RED","GREEN","YELLOW","BLUE","CYAN","WHITE"):
            setattr(C, _a, "")

# ═══════════════════════════════════════════════════════
#  Logging  (terminal + log file)
# ═══════════════════════════════════════════════════════
_log_fh = None

def _open_log():
    global _log_fh
    if _log_fh is None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _log_fh = open(LOG_DIR / f"run_{ts}.log", "a", encoding="utf-8")

def _emit(sym: str, col: str, msg: str):
    _open_log()
    line = f"  {col}{sym}{C.RESET}  {msg}"
    print(line)
    _log_fh.write(f"[{datetime.now().strftime('%H:%M:%S')}] {C.strip(line)}\n")
    _log_fh.flush()

def log_ok(m: str):    _emit("OK ", C.GREEN,  m)
def log_info(m: str):  _emit(".. ", C.CYAN,   m)
def log_warn(m: str):  _emit("!! ", C.YELLOW, m)
def log_err(m: str):   _emit("XX ", C.RED,    m)
def log_step(m: str):  _emit(">> ", C.BLUE,   m)

def log_title(m: str):
    bar = "=" * 60
    s = f"\n{C.BOLD}{C.CYAN}{bar}\n  {m}\n{bar}{C.RESET}\n"
    print(s); _open_log()
    _log_fh.write(C.strip(s) + "\n"); _log_fh.flush()

def log_section(m: str):
    bar = "-" * 60
    s = f"\n{C.BOLD}{bar}\n  {m}\n{bar}{C.RESET}"
    print(s); _open_log()
    _log_fh.write(C.strip(s) + "\n"); _log_fh.flush()

def log_box(lines: list):
    w = max(len(C.strip(l)) for l in lines) + 4
    print(f"  +{'-'*w}+")
    for l in lines:
        pad = w - len(C.strip(l)) - 2
        print(f"  |  {l}{' '*pad}|")
    print(f"  +{'-'*w}+\n")

# ═══════════════════════════════════════════════════════
#  State management  (run_state.json — resume support)
# ═══════════════════════════════════════════════════════
def _state_load() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text("utf-8"))
        except Exception:
            pass
    return {}

def _state_dump(d: dict):
    d["_updated"] = datetime.now().isoformat()
    STATE_FILE.write_text(json.dumps(d, indent=2, ensure_ascii=False), "utf-8")

def save_state(**kwargs):
    d = _state_load(); d.update(kwargs); _state_dump(d)

def get_state(key: str, default=None):
    return _state_load().get(key, default)

def mark_done(step: str, model: str = ""):
    k = f"done_{step}" + (f"_{model}" if model else "")
    save_state(**{k: True, f"{k}_at": datetime.now().isoformat()})

def is_done(step: str, model: str = "") -> bool:
    k = f"done_{step}" + (f"_{model}" if model else "")
    return bool(get_state(k))

def reset_step(step: str, model: str = ""):
    k = f"done_{step}" + (f"_{model}" if model else "")
    d = _state_load(); d.pop(k, None); d.pop(f"{k}_at", None); _state_dump(d)

# ═══════════════════════════════════════════════════════
#  Venv helpers
# ═══════════════════════════════════════════════════════
def venv_root() -> Path:
    return WORKSPACE / "venv"

def py_bin() -> str:
    v = venv_root()
    if sys.platform == "win32":
        return str(v / "Scripts" / "python.exe")
    return str(v / "bin" / "python3")

def run_in_venv(script: str, check=False) -> subprocess.CompletedProcess:
    """Execute a Python script string inside the venv."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                     delete=False, encoding="utf-8") as fh:
        fh.write(script); tmp = fh.name
    try:
        return subprocess.run([py_bin(), tmp], check=check)
    finally:
        try: os.unlink(tmp)
        except Exception: pass

# ═══════════════════════════════════════════════════════
#  File utilities
# ═══════════════════════════════════════════════════════
def fcount(d, ext=".wav") -> int:
    """Fast file count — os.scandir instead of glob (works on 700k+ files)."""
    if not d: return 0
    p = Path(d)
    if not p.exists(): return 0
    return sum(1 for e in os.scandir(p) if e.name.endswith(ext))

def delete_broken(d, min_bytes=500) -> int:
    """Delete files smaller than min_bytes. Returns count removed."""
    if not d: return 0
    p = Path(d)
    if not p.exists(): return 0
    n = 0
    for e in os.scandir(p):
        if e.is_dir(): continue  # skip subdirectories
        if e.stat().st_size < min_bytes:
            os.unlink(e.path); n += 1
    return n

def ffcmd(args: list, silent=True) -> bool:
    """Run ffmpeg with -y prefix. Returns True on success."""
    r = subprocess.run(["ffmpeg", "-y"] + args,
                       capture_output=silent)
    return r.returncode == 0

def symlink_dir(link, target):
    """Create a directory symlink link → target. Windows falls back to junction."""
    link, target = Path(link), Path(target)
    if link.is_symlink():
        if link.resolve() == target.resolve(): return
        link.unlink()
    elif link.exists():
        shutil.rmtree(str(link))
    try:
        link.symlink_to(target.resolve())
    except (OSError, NotImplementedError):
        try:
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link), str(target)],
                capture_output=True)
        except Exception:
            shutil.copytree(str(target), str(link))

def symlink_file(link, target):
    """Create a file symlink. Falls back to copy."""
    link, target = Path(link), Path(target)
    if link.exists() or link.is_symlink():
        if link.is_symlink() and link.resolve() == target.resolve(): return
        link.unlink()
    try:
        link.symlink_to(target.resolve())
    except (OSError, NotImplementedError):
        shutil.copy2(str(target), str(link))

def confirm(prompt: str, auto=False) -> bool:
    if auto: return True
    ans = input(f"\n  ? {prompt} [y/N]: ").strip().lower()
    return ans in ("y", "yes", "\u05db\u05df", "1")
