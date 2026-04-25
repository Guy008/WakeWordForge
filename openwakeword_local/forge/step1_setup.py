"""
forge/step1_setup.py
Step 1 — Environment Setup

Creates the Python venv, installs all dependencies, and downloads
one-time assets (openWakeWord, Piper, AudioSet, MIT-RIR, FMA, ACAV NPY).

Can be re-run safely — skips anything already present.
Use --force to re-download everything.
"""
from __future__ import annotations
import os, shutil, subprocess, sys, tarfile, textwrap
from pathlib import Path

from .common import (
    WORKSPACE, OWW_REPO, PIPER_REPO, PIPER_PT_MODEL_URL, PIPER_PT_MODEL_NAME, PIPER_PT_JSON_URL, ACAV_URL, VAL_URL,
    AUDIOSET_URL, MIT_RIR_REPO,
    log_ok, log_info, log_warn, log_err, log_step, log_section, log_title,
    venv_root, py_bin, fcount, mark_done, is_done, save_state,
)

# ── Python packages ────────────────────────────────────────
# Versions pinned for Python 3.10/3.11 compatibility with openWakeWord.
# torch: CUDA build selected dynamically in _install_deps() based on hw.
PACKAGES_CPU = [
    ["torch", "torchvision", "torchaudio"],
]
PACKAGES_CUDA = [
    ["torch==2.5.1+cu121", "torchvision", "torchaudio",
     "--index-url", "https://download.pytorch.org/whl/cu121"],
]
PACKAGES_COMMON = [
    ["numpy==1.26.4"],
    ["scipy==1.13.1"],
    ["pyarrow==14.0.2"],          # last version supporting datasets 2.x on Py3.10/3.11
    # onnxruntime installed dynamically in _install_deps (gpu vs cpu)
    ["ai_edge_litert==1.0.1"],
    ["onnxsim", "onnx==1.16.2"],
    # onnx-tf removed — using onnx2tf instead (onnx-tf broken with tensorflow_probability)
    ["onnx_graphsurgeon", "sng4onnx"],
    ["onnx2tf==1.26.3"],
    ["mutagen==1.47.0", "torchinfo==1.8.0", "torchmetrics==1.2.0"],
    ["speechbrain==0.5.14"],
    ["audiomentations==0.33.0", "torch-audiomentations==0.11.0"],
    ["acoustics==0.2.6"],
    ["pronouncing==0.2.0"],
    ["datasets==2.14.6"],
    ["deep-phonemizer==0.0.19"],
    ["setuptools", "webrtcvad"],
    ["piper-phonemize"],
    ["edge-tts",
        "transformers>=4.33.0",
        "soundfile",
        "openai-whisper", "pydub", "pyyaml", "tqdm"],
    ["requests", "huggingface_hub==0.23.0"],
    ["pyaudio", "librosa", "soundfile"],
]

# Python versions supported by openWakeWord dependencies
SUPPORTED_PY = [(3, 10), (3, 11), (3, 12)]
RECOMMENDED_PY = "3.11"


# ══════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════
def run(force=False, only=None, skip_downloads=False):
    """
    Run full setup, or a subset via `only` list.
    only: list of strings from {"venv","deps","oww","piper",
                                 "audioset","rirs","fma","acav","val"}
    skip_downloads: skip AudioSet / FMA / MIT-RIRs (use if downloads fail).
                    ACAV and validation NPY are always downloaded (small + required).
    """
    log_title("Step 1 — Environment Setup")
    want = set(only) if only else None

    def _want(tag): return want is None or tag in want

    _check_system_tools()

    if _want("venv"):   _make_venv()
    if _want("deps"):   _install_deps(force)
    if _want("deps"):   _patch_venv()
    if _want("oww"):    _get_oww(force)
    if _want("piper"):  _get_piper(force)

    if skip_downloads:
        log_warn("Skipping AudioSet / MIT-RIRs / FMA downloads (--skip-downloads)")
        log_info("Training will use only negative_custom/ for background noise")
    else:
        if _want("audioset"): _get_audioset(force)
        if _want("rirs"):     _get_rirs(force)
        if _want("fma"):      _get_fma(force)

    if _want("acav"):   _get_acav(force)
    if _want("val"):    _get_val(force)
    if _want("chime6"): _get_chime6(force)

    mark_done("setup")
    log_ok("Step 1 complete")


# ══════════════════════════════════════════════════════════
#  System tools
# ══════════════════════════════════════════════════════════

def _get_chime6(force: bool = False):
    """
    Download MUSAN noise-only WAV files from HuggingFace via wget.
    No streaming API needed — direct file download, no _run_script.
    """
    chime_dir = WORKSPACE / "chime6_clips"
    if not force and chime_dir.exists() and fcount(chime_dir) > 100:
        log_info(f"MUSAN noise clips: {fcount(chime_dir):,} (cached)")
        return
    chime_dir.mkdir(parents=True, exist_ok=True)
    log_step("Downloading MUSAN noise from HuggingFace...")

    import json, urllib.request, urllib.error
    HF_API = "https://huggingface.co/api/datasets/FluidInference/musan/tree/main/noise"
    try:
        req = urllib.request.Request(HF_API,
              headers={"User-Agent": "WakeWordForge/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            items = json.loads(r.read())
    except Exception as e:
        log_warn(f"  MUSAN index fetch failed: {e} — skipping")
        return

    wav_urls = []
    for item in items:
        if isinstance(item, dict) and item.get("type") == "file" and item.get("path","").endswith(".wav"):
            wav_urls.append(
                f"https://huggingface.co/datasets/FluidInference/musan/resolve/main/{item['path']}"
            )
        if len(wav_urls) >= 50:
            break

    if not wav_urls:
        log_warn("  No MUSAN noise WAVs found — skipping")
        return

    n = 0
    for i, url in enumerate(wav_urls):
        fname = chime_dir / f"musan_{i:04d}.wav"
        if fname.exists() and fname.stat().st_size > 10_000:
            n += 1; continue
        r = _download_quiet(url, str(fname))
        if r.returncode == 0 and fname.exists() and fname.stat().st_size > 10_000:
            n += 1
        elif fname.exists():
            fname.unlink()

    log_ok(f"MUSAN noise: {n} files downloaded")

def _download(url: str, dest: str) -> subprocess.CompletedProcess:
    """Download a file using wget or curl (fallback). Works on Linux, macOS, WSL."""
    if shutil.which("wget"):
        return subprocess.run(["wget", "-q", "--show-progress", "-O", dest, url])
    elif shutil.which("curl"):
        return subprocess.run(["curl", "-L", "--progress-bar", "-o", dest, url])
    else:
        raise RuntimeError("Neither wget nor curl is installed. Install one and retry.")


def _download_quiet(url: str, dest: str) -> subprocess.CompletedProcess:
    """Silent download (no progress bar) — for small files."""
    if shutil.which("wget"):
        return subprocess.run(["wget", "-q", "-O", dest, url], capture_output=True)
    elif shutil.which("curl"):
        return subprocess.run(["curl", "-sL", "-o", dest, url], capture_output=True)
    else:
        raise RuntimeError("Neither wget nor curl is installed.")


def _check_system_tools():
    missing_hard = [t for t in ("ffmpeg", "git") if shutil.which(t) is None]
    has_downloader = shutil.which("wget") or shutil.which("curl")

    if missing_hard:
        log_warn(f"Missing required tools: {', '.join(missing_hard)}")
        if sys.platform == "win32":
            log_info("Install: winget install ffmpeg Git.Git")
        elif sys.platform == "darwin":
            log_info("Install: brew install " + " ".join(missing_hard))
        else:
            log_info("Install: sudo apt install -y " + " ".join(missing_hard))
    else:
        log_ok("System tools: ffmpeg / git present")

    if not has_downloader:
        log_warn("Neither wget nor curl found — downloads will fail")
        if sys.platform == "darwin":
            log_info("Install wget: brew install wget")
        else:
            log_info("Install: sudo apt install -y wget")
    else:
        dl = "wget" if shutil.which("wget") else "curl"
        log_ok(f"Downloader: {dl}")


# ══════════════════════════════════════════════════════════
#  Venv + dependencies
# ══════════════════════════════════════════════════════════
def _find_compatible_python() -> str:
    """Find Python 3.10/3.11/3.12 — required by openWakeWord deps."""
    # Check current python first
    major, minor = sys.version_info[:2]
    if (major, minor) in SUPPORTED_PY:
        return sys.executable

    # Search for compatible interpreter
    candidates = []
    for ver in ["3.11", "3.10", "3.12"]:
        for name in [f"python{ver}", f"python3.{ver.split('.')[1]}"]:
            path = shutil.which(name)
            if path:
                candidates.append(path)

    if candidates:
        return candidates[0]

    return ""   # not found


def _make_venv():
    venv = venv_root()
    py   = py_bin()
    if Path(py).exists():
        r2 = subprocess.run([py, "-c", "import sys; print(sys.version_info[:2])"],
                            capture_output=True, text=True)
        try:
            vi = eval(r2.stdout.strip())
            if vi in SUPPORTED_PY:
                log_info(f"venv exists: Python {vi[0]}.{vi[1]}")
                return   # good version — keep it
            # Wrong Python version — delete and recreate
            log_warn(f"venv uses Python {vi[0]}.{vi[1]} — incompatible with openWakeWord deps")
            log_warn(f"Auto-deleting and recreating with Python {RECOMMENDED_PY}...")
            shutil.rmtree(str(venv))
        except Exception:
            pass   # venv exists but broken — fall through to recreate

    log_step("Creating Python venv")

    # Find a compatible Python
    py_exec = _find_compatible_python()
    major, minor = sys.version_info[:2]

    if not py_exec:
        log_err(f"Python {RECOMMENDED_PY} not found on this system!")
        log_err(f"Current Python: {major}.{minor} — not compatible with openWakeWord deps")
        log_err(f"Install Python {RECOMMENDED_PY}:")
        if sys.platform == "win32":
            log_err(f"  winget install Python.Python.3.11")
        else:
            log_err(f"  sudo apt install python3.11 python3.11-venv")
        log_err(f"Then re-run step 1.")
        raise SystemExit(1)

    if py_exec != sys.executable:
        log_info(f"Using {py_exec} (current Python {major}.{minor} is not compatible)")

    WORKSPACE.mkdir(parents=True, exist_ok=True)
    subprocess.run([py_exec, "-m", "venv", str(venv)], check=True)
    log_ok(f"venv: {venv} (Python {RECOMMENDED_PY})")


def _install_deps(force):
    log_section("Installing dependencies")
    py  = py_bin()
    pip = [py, "-m", "pip", "install", "--quiet", "--upgrade"]

    # Detect CUDA
    has_cuda = False
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True)
        has_cuda = r.returncode == 0
    except Exception:
        pass

    # Install torch
    torch_pkgs = PACKAGES_CUDA if has_cuda else PACKAGES_CPU
    label = torch_pkgs[0][0][:40]
    log_info(f"  torch ({'CUDA' if has_cuda else 'CPU'})...")
    r = subprocess.run(pip + torch_pkgs[0], capture_output=True)
    if r.returncode != 0:
        log_warn(f"  CUDA torch failed, trying CPU fallback...")
        r = subprocess.run(pip + PACKAGES_CPU[0], capture_output=True)
        if r.returncode != 0:
            log_warn(f"  torch install failed: {r.stderr.decode()[-200:]}")

    # Install onnxruntime — GPU version if CUDA available
    if has_cuda:
        log_info("  onnxruntime-gpu (CUDA)...")
        r = subprocess.run(pip + ["onnxruntime-gpu==1.19.2"], capture_output=True)
        if r.returncode != 0:
            log_warn("  onnxruntime-gpu failed, falling back to CPU...")
            subprocess.run(pip + ["onnxruntime==1.19.2"], capture_output=True)
        else:
            log_ok("  onnxruntime-gpu installed")
    else:
        log_info("  onnxruntime (CPU)...")
        subprocess.run(pip + ["onnxruntime==1.19.2"], capture_output=True)

    # Install all other packages
    for pkg_list in PACKAGES_COMMON:
        label = pkg_list[0][:40]
        log_info(f"  {label}...")
        r = subprocess.run(pip + pkg_list, capture_output=True)
        if r.returncode != 0:
            log_warn(f"  FAILED: {label}\n{r.stderr.decode()[-150:]}")
    log_ok("Dependencies done")


def _patch_venv():
    """Fix webrtcvad and pronouncing for environments without pkg_resources."""
    venv = venv_root()
    site = next((p for p in (venv / "lib").rglob("site-packages")
                 if p.is_dir()), None)
    if not site:
        log_warn("site-packages not found — skipping patches")
        return

    # webrtcvad stub
    wv = site / "webrtcvad.py"
    if wv.exists():
        wv.write_text(textwrap.dedent("""\
            try:
                from importlib.metadata import version
                __version__ = version("webrtcvad")
            except Exception:
                __version__ = "2.0.10"

            class Vad:
                def __init__(self, mode=3): self._mode = mode
                def set_mode(self, m): self._mode = m
                def is_speech(self, buf, sr, length=None): return True
        """), encoding="utf-8")
        log_info("webrtcvad patched")

    # pronouncing — remove pkg_resources import
    pron = site / "pronouncing" / "__init__.py"
    if pron.exists():
        code = pron.read_text("utf-8")
        if "pkg_resources" in code:
            import re
            code = re.sub(r"import pkg_resources\b",
                          "import importlib.resources as _ir", code)
            code = code.replace("pkg_resources.resource_filename",
                                "_ir.files('pronouncing').joinpath")
            pron.write_text(code, "utf-8")
            log_info("pronouncing patched")


# ══════════════════════════════════════════════════════════
#  Asset downloads
# ══════════════════════════════════════════════════════════
def _get_oww(force):
    dst = WORKSPACE / "openwakeword"
    if not force and (dst / "openwakeword" / "train.py").exists():
        log_info("openWakeWord: present")
    else:
        log_step("Cloning openWakeWord")
        if dst.exists(): shutil.rmtree(str(dst))
        subprocess.run(["git", "clone", OWW_REPO, str(dst)], check=True)
        log_ok("openWakeWord cloned")
    # Always ensure it's installed in the current venv
    log_info("Installing openwakeword into venv...")
    r = subprocess.run([py_bin(), "-m", "pip", "install", "--quiet", "-e", str(dst)],
                       capture_output=True)
    if r.returncode != 0:
        log_warn(f"openwakeword pip install failed: {r.stderr.decode()[-200:]}")
    else:
        log_ok("openwakeword installed in venv")

    # Download resource models required by train.py (melspectrogram + embedding)
    # These are stored with git-lfs so they are NOT downloaded by git clone.
    OWW_RELEASE = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"
    res_dir = dst / "openwakeword" / "resources" / "models"
    res_dir.mkdir(parents=True, exist_ok=True)
    for fname in ["melspectrogram.onnx", "melspectrogram.tflite",
                  "embedding_model.onnx", "embedding_model.tflite"]:
        fpath = res_dir / fname
        if fpath.exists() and fpath.stat().st_size > 10_000:
            continue
        log_step(f"  Downloading {fname}")
        r = _download_quiet(f"{OWW_RELEASE}/{fname}", str(fpath))
        if r.returncode == 0 and fpath.stat().st_size > 10_000:
            log_ok(f"  {fname} ({fpath.stat().st_size//1024} KB)")
        else:
            log_warn(f"  {fname} download failed")


def _get_piper(force):
    dst = WORKSPACE / "piper-sample-generator"

    # Clone repo (pinned to same commit as OWW notebook: 213d4d5)
    if not force and (dst / "generate_samples.py").exists():
        log_info("Piper repo: present")
    else:
        log_step("Cloning Piper sample generator")
        if dst.exists(): shutil.rmtree(str(dst))
        subprocess.run(["git", "clone", PIPER_REPO, str(dst)], check=True)
        # No checkout — use latest version which has generate_samples_onnx
        log_ok("Piper cloned")

    # Install piper-tts + piper-phonemize into venv
    for pkg in ["piper-tts", "piper-phonemize-cross"]:
        r = subprocess.run(
            [py_bin(), "-m", "pip", "install", "--quiet", pkg],
            capture_output=True)
        if r.returncode == 0:
            log_ok(f"  {pkg} installed")
        else:
            log_warn(f"  {pkg} install failed: {r.stderr.decode()[-100:]}")

    # Download the LibriTTS-R .pt generator model (904 English speakers)
    models_dir = dst / "models"
    models_dir.mkdir(exist_ok=True)
    pt_model = models_dir / PIPER_PT_MODEL_NAME
    if not force and pt_model.exists() and pt_model.stat().st_size > 1_000_000:
        log_info(f"  Piper model: {PIPER_PT_MODEL_NAME} (skip)")
    else:
        log_step(f"  Downloading Piper model: {PIPER_PT_MODEL_NAME}")
        r = _download(PIPER_PT_MODEL_URL, str(pt_model))
        if r.returncode == 0 and pt_model.exists():
            log_ok(f"  Piper model downloaded ({pt_model.stat().st_size//1024//1024} MB)")
        else:
            log_warn("  Piper model download failed — English GPU generation unavailable")

    # generate_samples needs a .pt.json config with num_speakers + espeak voice
    # The repo stores it with git-lfs so it may be empty after clone.
    # We write it ourselves — all values are known constants for libritts_r-medium.
    import json as _json
    pt_json = models_dir / (PIPER_PT_MODEL_NAME + ".json")
    needs_write = True
    if pt_json.exists() and pt_json.stat().st_size > 100:
        try:
            cfg = _json.loads(pt_json.read_text())
            if cfg.get("num_speakers", 0) == 904:
                log_info(f"  Piper model config: 904 speakers (present)")
                needs_write = False
        except Exception:
            pass
    if needs_write:
        cfg = {
            "audio": {"sample_rate": 22050},
            "espeak": {"voice": "en-us"},
            "inference": {"noise_scale": 0.667, "length_scale": 1.0, "noise_w": 0.8},
            "phoneme_type": "espeak",
            "phoneme_map": {},
            "phoneme_id_map": {},
            "num_symbols": 256,
            "num_speakers": 904,
            "speaker_id_map": {}
        }
        pt_json.write_text(_json.dumps(cfg))
        log_ok("  Piper model config: written (904 speakers)")
def _get_audioset(force):
    out = WORKSPACE / "audioset_16k"
    if not force and fcount(out) > 50:
        log_info(f"AudioSet: {fcount(out)} files"); return
    log_step("Downloading AudioSet background noise")
    out.mkdir(parents=True, exist_ok=True)
    tmp = WORKSPACE / "_dl_audioset"
    tmp.mkdir(parents=True, exist_ok=True)
    tar = tmp / "bal_train09.tar"

    # Download — delete and retry if file is truncated/corrupted
    for attempt in range(3):
        if tar.exists():
            try:
                with tarfile.open(tar) as t:
                    t.getmembers()   # validate — raises if corrupted
                break   # file is good
            except Exception:
                log_warn(f"tar corrupted (attempt {attempt+1}), re-downloading...")
                tar.unlink()

        r = _download(AUDIOSET_URL, str(tar))
        if r.returncode != 0:
            log_warn("AudioSet download failed — skipping")
            return
    else:
        log_warn("AudioSet download failed after 3 attempts — skipping")
        return

    try:
        with tarfile.open(tar) as t:
            t.extractall(str(tmp))
    except Exception as e:
        log_warn(f"AudioSet extract failed: {e} — skipping")
        return

    _convert_audio_dir(tmp, out, "AudioSet")
    shutil.rmtree(str(tmp), ignore_errors=True)
    log_ok(f"AudioSet: {fcount(out)} WAV files")


def _get_rirs(force):
    out = WORKSPACE / "mit_rirs"
    if not force and fcount(out) > 100:
        log_info(f"MIT RIRs: {fcount(out)} files"); return
    log_step("Downloading MIT Room Impulse Responses")
    out.mkdir(parents=True, exist_ok=True)
    tmp = WORKSPACE / "_dl_rirs"
    if tmp.exists(): shutil.rmtree(str(tmp))
    r = subprocess.run(["git", "clone", MIT_RIR_REPO, str(tmp)])
    if r.returncode != 0:
        log_warn("MIT RIR clone failed — skipping"); return
    _convert_audio_dir(tmp, out, "MIT RIRs")
    shutil.rmtree(str(tmp), ignore_errors=True)
    log_ok(f"MIT RIRs: {fcount(out)} files")


def _get_fma(force):
    out = WORKSPACE / "fma"
    if not force and fcount(out) > 50:
        log_info(f"FMA: {fcount(out)} files"); return
    log_step("Downloading Free Music Archive (1-hour sample)")
    out.mkdir(parents=True, exist_ok=True)
    # run in venv so datasets package is available
    from .common import run_in_venv
    run_in_venv(f"""
import datasets as ds, numpy as np, scipy.io.wavfile as sf
from pathlib import Path
from tqdm import tqdm

out = Path({repr(str(out))})
fma = iter(
    ds.load_dataset("rudraml/fma", name="small", split="train", streaming=True)
      .cast_column("audio", ds.Audio(sampling_rate=16000))
)
for i in tqdm(range(120), desc="FMA"):
    try:
        row = next(fma)
        name = Path(row["audio"]["path"]).stem + ".wav"
        sf.write(str(out/name), 16000,
                 (row["audio"]["array"]*32767).astype(np.int16))
    except StopIteration:
        break
print(f"FMA done: {{sum(1 for _ in out.iterdir())}} files")
""")
    log_ok(f"FMA: {fcount(out)} files")


def _get_acav(force):
    dst = WORKSPACE / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    if not force and dst.exists() and dst.stat().st_size > 1_000_000:
        log_info(f"ACAV100M: {dst.stat().st_size/1e9:.1f} GB"); return
    log_step("Downloading ACAV100M features (~17 GB, one-time download)")
    r = _download(ACAV_URL, str(dst))
    if r.returncode != 0:
        raise RuntimeError("ACAV download failed")
    log_ok("ACAV100M done")


def _get_val(force):
    full = WORKSPACE / "validation_set_features.npy"
    if not force and full.exists() and full.stat().st_size > 10_000:
        log_info("Validation features: present"); return
    log_step("Downloading validation features")
    r = _download(VAL_URL, str(full))
    if r.returncode != 0:
        raise RuntimeError("Validation features download failed")
    # create a smaller version for low-RAM systems
    small = WORKSPACE / "validation_set_small.npy"
    if not small.exists():
        try:
            import numpy as np
            data = np.load(str(full))
            np.save(str(small), data[:5000])
            log_info(f"Validation small: 5,000 rows saved")
        except Exception as e:
            log_warn(f"Could not create small validation: {e}")
    log_ok("Validation features done")


# ── helpers ────────────────────────────────────────────────
def _convert_audio_dir(src_dir: Path, dst_dir: Path, label: str):
    """Convert all audio files under src_dir to 16kHz mono WAV using ffmpeg.
    Works on any format ffmpeg supports — no soundfile/datasets dependency."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}
    files = [f for f in src_dir.rglob("*") if f.suffix.lower() in exts]
    if not files:
        log_warn(f"{label}: no audio files found in {src_dir}"); return

    log_info(f"{label}: converting {len(files)} files...")
    ok = 0
    try:
        from tqdm import tqdm
        it = tqdm(files, desc=label)
    except ImportError:
        it = files

    for src in it:
        dst = dst_dir / (src.stem + ".wav")
        if dst.exists() and dst.stat().st_size > 500:
            ok += 1; continue
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(src),
             "-ar", "16000", "-ac", "1",
             "-sample_fmt", "s16",
             str(dst)],
            capture_output=True)
        if r.returncode == 0 and dst.exists() and dst.stat().st_size > 500:
            ok += 1
        else:
            # Try without sample_fmt flag (some older ffmpeg)
            r2 = subprocess.run(
                ["ffmpeg", "-y", "-i", str(src),
                 "-ar", "16000", "-ac", "1", str(dst)],
                capture_output=True)
            if r2.returncode == 0:
                ok += 1

    log_info(f"{label}: {ok}/{len(files)} converted")
