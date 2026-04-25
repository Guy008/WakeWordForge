#!/usr/bin/env bash
# WakeWordForge — one-command installer for Linux / macOS / WSL
# Usage:  bash install.sh
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC}  $*"; }
warn() { echo -e "${YELLOW}[!!]${NC}  $*"; }
err()  { echo -e "${RED}[ERR]${NC} $*"; exit 1; }

echo "=================================================="
echo "  WakeWordForge — Environment Setup"
echo "=================================================="
echo

# ── Detect WSL ──────────────────────────────────────────
IS_WSL=false
if grep -qiE "microsoft|wsl" /proc/version 2>/dev/null; then
    IS_WSL=true
    ok "WSL detected — running in Windows Subsystem for Linux"
    echo "    Audio recording requires a PulseAudio or PipeWire server."
    echo "    See: https://github.com/guy008/WakeWordForge/wiki/WSL-Audio"
    echo
fi

# ── Python ─────────────────────────────────────────────
if command -v python3 &>/dev/null; then
    PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PY_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
    PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
    ok "Python $PY_VER"
    if [ "$PY_MAJOR" -lt 3 ] || [ "$PY_MINOR" -lt 10 ]; then
        err "Python 3.10+ required (found $PY_VER). Install via pyenv or your package manager."
    fi
else
    err "python3 not found. Install Python 3.10+ first:
  Ubuntu/Debian:  sudo apt install python3 python3-pip python3-venv
  macOS:          brew install python@3.11
  Other:          https://www.python.org/downloads/"
fi

# ── git ────────────────────────────────────────────────
command -v git &>/dev/null && ok "git" || err "git not found. Install git first."

# ── ffmpeg ─────────────────────────────────────────────
if command -v ffmpeg &>/dev/null; then
    ok "ffmpeg"
else
    warn "ffmpeg not found — trying to install automatically..."
    if command -v apt-get &>/dev/null; then
        sudo apt-get install -y ffmpeg && ok "ffmpeg installed"
    elif command -v brew &>/dev/null; then
        brew install ffmpeg && ok "ffmpeg installed"
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y ffmpeg && ok "ffmpeg installed"
    elif command -v pacman &>/dev/null; then
        sudo pacman -S --noconfirm ffmpeg && ok "ffmpeg installed"
    else
        warn "Cannot install ffmpeg automatically. Install manually:"
        warn "  Ubuntu/Debian: sudo apt install ffmpeg"
        warn "  macOS:         brew install ffmpeg"
        warn "  Other:         https://ffmpeg.org/download.html"
    fi
fi

# ── wget / curl (one is required for downloads) ────────
if command -v wget &>/dev/null; then
    ok "wget"
elif command -v curl &>/dev/null; then
    ok "curl (wget not found, will use curl for downloads)"
else
    warn "Neither wget nor curl found — trying to install wget..."
    if command -v apt-get &>/dev/null; then
        sudo apt-get install -y wget && ok "wget installed"
    elif command -v brew &>/dev/null; then
        brew install wget && ok "wget installed"
    elif command -v pacman &>/dev/null; then
        sudo pacman -S --noconfirm wget && ok "wget installed"
    else
        warn "Could not install wget automatically."
        warn "Install with:  sudo apt install wget  or  brew install wget"
    fi
fi

# ── portaudio (for microphone recording) ───────────────
if python3 -c "import pyaudio" &>/dev/null 2>&1; then
    ok "pyaudio"
else
    if command -v apt-get &>/dev/null; then
        sudo apt-get install -y portaudio19-dev python3-dev 2>/dev/null || true
    elif command -v brew &>/dev/null; then
        brew install portaudio 2>/dev/null || true
    fi
fi

# ── GPU detection ───────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    ok "NVIDIA GPU detected — CUDA training will be used"
elif command -v rocminfo &>/dev/null 2>&1; then
    warn "AMD GPU (ROCm) detected — install PyTorch ROCm build for GPU acceleration"
    warn "See: https://pytorch.org/get-started/locally/ → select ROCm"
else
    warn "No NVIDIA GPU detected — CPU training (slower but works)"
    warn "Tip: use --samples 20000 --steps 50000 to reduce training time on CPU"
fi

echo
echo "  Running Step 1 — creating Python venv and installing all dependencies..."
echo "  (This downloads ~2-4 GB of models and data — takes 20-60 minutes)"
echo

python3 run.py --step 1

echo
ok "WakeWordForge is ready!"
echo
echo "  To train your first wake word:"
echo "    python3 run.py"
echo
echo "  For non-interactive training (example):"
echo "    python3 run.py --model hey_gadi --he \"היי גדי\" --en \"hey gadi\""
echo
