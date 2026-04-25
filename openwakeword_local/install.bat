@echo off
:: WakeWordForge — one-command installer for Windows
:: Usage:  install.bat
setlocal EnableDelayedExpansion

echo ==================================================
echo   WakeWordForge -- Environment Setup (Windows)
echo ==================================================
echo.

:: ── Python ─────────────────────────────────────────────
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERR] Python not found.
    echo       Install Python 3.10+ from https://www.python.org/downloads/
    echo       Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYVER=%%i
echo [OK]  %PYVER%

:: ── git ────────────────────────────────────────────────
where git >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERR] git not found.
    echo       Install git from https://git-scm.com/download/win
    pause
    exit /b 1
)
echo [OK]  git

:: ── ffmpeg ─────────────────────────────────────────────
where ffmpeg >nul 2>&1
if %errorlevel% neq 0 (
    echo [!!]  ffmpeg not found.
    echo       Install with winget:  winget install ffmpeg
    echo       Or download from https://ffmpeg.org/download.html
    echo       (TTS preview and audio playback require ffmpeg)
    echo.
) else (
    echo [OK]  ffmpeg
)

:: ── CUDA check ─────────────────────────────────────────
where nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK]  NVIDIA GPU detected -- CUDA training will be used
) else (
    echo [!!]  No NVIDIA GPU detected -- CPU training (slower but works)
)

echo.
echo   Running Step 1 -- creating Python venv and installing all dependencies...
echo   (This downloads 2-4 GB -- takes 20-60 minutes)
echo.

python run.py --step 1

echo.
echo [OK]  WakeWordForge is ready!
echo.
echo   To train your first wake word:
echo     python run.py
echo.
echo   For non-interactive training (example):
echo     python run.py --model my_word --he "my hebrew" --en "my english"
echo.
pause
