"""
forge/recorder.py
Interactive microphone recording wizard.

Records wake word samples directly from the mic and saves them to
my_recordings/ as speaker01_take01.wav ... speaker01_takeNN.wav.

Called from run.py --record or from the interactive wizard when the user
chooses to record before training.
"""
from __future__ import annotations
import os, struct, subprocess, sys, time, wave
from pathlib import Path

from .common import MY_RECORDINGS_DIR, log_ok, log_info, log_warn, log_err, log_section, fcount

# Recording parameters
RATE       = 16000
CHANNELS   = 1
CHUNK      = 1024          # frames per read
DURATION   = 3.0           # max seconds per take
FORMAT_INT = 8             # pyaudio.paInt16 = 8  (avoid importing pyaudio at module level)
QUIET_RMS  = 150           # below this = microphone not picking up anything


# ══════════════════════════════════════════════════════════
#  Public entry point
# ══════════════════════════════════════════════════════════
def run(model_name: str = "", target: int = 50) -> int:
    """
    Interactively record up to `target` wake word samples.
    Saves WAV files to my_recordings/.
    Returns total number of recordings in my_recordings/ after session.
    Gracefully handles missing pyaudio or no microphone.
    """
    try:
        import pyaudio   # noqa — available inside venv
    except ImportError:
        log_warn("pyaudio not found — run  python run.py --step 1  first")
        return _count()

    out_dir = MY_RECORDINGS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    n_existing = _count()
    if n_existing >= target:
        log_ok(f"Already have {n_existing} recordings — nothing to record")
        return n_existing

    needed = target - n_existing

    log_section("Recording Wizard")
    _print_intro(needed, target)

    pa = pyaudio.PyAudio()
    device_idx = _choose_device(pa)
    if device_idx is None:
        pa.terminate()
        log_warn("No microphone detected — skipping recording session")
        return n_existing

    made  = 0
    n     = n_existing + 1

    try:
        while n <= target:
            out_path = out_dir / f"speaker01_take{n:02d}.wav"
            print(f"\n  ── Take {n}/{target} ──", flush=True)
            try:
                prompt = input("  Press ENTER to record (or q to quit): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break

            if prompt == "q":
                break

            ok = _record_one(pa, device_idx, out_path)
            if not ok:
                log_warn("  Recording failed — check your microphone")
                continue

            rms = _rms(out_path)
            bar = _bar(rms)
            print(f"  Captured  {bar}")

            if rms < QUIET_RMS:
                log_warn("  Very quiet — microphone may not be picking up audio")

            try:
                action = input("  [ENTER=keep | r=redo | p=play | q=quit]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                if out_path.exists():
                    made += 1
                    n += 1
                break

            if action == "r":
                out_path.unlink(missing_ok=True)
                continue

            if action == "p":
                _play(out_path)
                try:
                    action = input("  [ENTER=keep | r=redo]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    action = ""
                if action == "r":
                    out_path.unlink(missing_ok=True)
                    continue

            if action == "q":
                if out_path.exists():
                    made += 1
                    n += 1
                break

            made += 1
            n += 1
            log_ok(f"  Take saved  ({made} new,  {n_existing + made} total)")

    except Exception as exc:
        log_warn(f"Recording error: {exc}")
    finally:
        pa.terminate()

    total = _count()
    print()
    log_ok(f"Session complete — {total} recordings in my_recordings/")
    return total


# ══════════════════════════════════════════════════════════
#  Internal helpers
# ══════════════════════════════════════════════════════════
def _record_one(pa, device_idx: int, out_path: Path) -> bool:
    """Record one take: countdown → record → save WAV. Returns True on success."""
    import pyaudio

    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_idx,
            frames_per_buffer=CHUNK,
        )
    except Exception as e:
        log_warn(f"  Cannot open mic: {e}")
        return False

    # Countdown
    for i in (3, 2, 1):
        print(f"\r  Starting in {i}...  ", end="", flush=True)
        time.sleep(0.6)
    print(f"\r  RECORDING — say your wake word now!          ", flush=True)

    frames   = []
    n_chunks = int(RATE / CHUNK * DURATION)

    for i in range(n_chunks):
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
        except Exception:
            break
        frames.append(data)

        # Live level bar
        samples = struct.unpack(f"<{len(data) // 2}h", data)
        rms_live = _rms_samples(samples)
        bar = _bar(rms_live, width=25)
        elapsed = (i + 1) * CHUNK / RATE
        print(f"\r  {bar}  {elapsed:.1f}s", end="", flush=True)

    print()  # newline after bar
    stream.stop_stream()
    stream.close()

    if not frames:
        return False

    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)   # 16-bit
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))

    return out_path.exists() and out_path.stat().st_size > 1000


def _choose_device(pa) -> int | None:
    """Return microphone device index. Tries default first, then lists options."""
    # Try default input device
    try:
        info = pa.get_default_input_device_info()
        name = info["name"]
        print(f"\n  Microphone: {name}")
        return int(info["index"])
    except Exception:
        pass

    # List available input devices
    devices = []
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info.get("maxInputChannels", 0) > 0:
            devices.append((i, info["name"]))

    if not devices:
        return None

    if len(devices) == 1:
        print(f"\n  Microphone: {devices[0][1]}")
        return devices[0][0]

    print("\n  Available microphones:")
    for idx, (dev_i, name) in enumerate(devices):
        print(f"    {idx}: {name}")

    try:
        choice = input(f"  Select [0-{len(devices)-1}]: ").strip()
        return devices[int(choice)][0]
    except (ValueError, IndexError, EOFError):
        return devices[0][0]


def _rms(path: Path) -> float:
    """Compute RMS from a saved WAV file."""
    try:
        with wave.open(str(path)) as wf:
            raw = wf.readframes(wf.getnframes())
        n = len(raw) // 2
        if n == 0:
            return 0.0
        samples = struct.unpack(f"<{n}h", raw)
        return _rms_samples(samples)
    except Exception:
        return 0.0


def _rms_samples(samples) -> float:
    n = len(samples)
    if n == 0:
        return 0.0
    return (sum(s * s for s in samples) / n) ** 0.5


def _bar(rms: float, max_rms: float = 6000, width: int = 20) -> str:
    """ANSI-coloured level bar."""
    ratio  = min(rms / max_rms, 1.0)
    filled = int(ratio * width)
    bar    = "\u2588" * filled + "\u2591" * (width - filled)

    if rms < 150:
        color = "\033[91m"   # red — too quiet
    elif ratio > 0.85:
        color = "\033[93m"   # yellow — clipping risk
    else:
        color = "\033[92m"   # green — good level
    return f"{color}[{bar}]\033[0m  RMS={rms:.0f}"


def _play(path: Path):
    """Play back a WAV file using the system player."""
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
    print(f"  (Cannot play — open manually: {path})")


def _count() -> int:
    return len(list(MY_RECORDINGS_DIR.glob("*.wav"))) if MY_RECORDINGS_DIR.exists() else 0


def _print_intro(needed: int, target: int):
    print(f"""
  Goal: record {needed} take(s)  (total target: {target})

  Tips:
    - Say your wake word naturally, as you would in real use
    - Vary distance to mic: try close, normal, and slightly far
    - Vary speed: some takes a bit faster, some slower
    - Record in the room where you'll use the device
    - Even 20 takes beats TTS-only mode significantly

  Controls during session:
    ENTER  = start recording, then keep the take
    r      = redo this take
    p      = play back before deciding
    q      = stop session (keeps all takes recorded so far)
""")
