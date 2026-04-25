"""
forge/step4_features.py  (rewritten)
Step 4 — Prepare Training Directories

What OWW --augment_clips actually needs:
  model_dir/positive_train/   WAV files (our augmented clips)
  model_dir/positive_test/    WAV files (our base clips)
  model_dir/negative_train/   WAV files (background noise)
  model_dir/negative_test/    WAV files (background noise)

We use symlinks to point these at existing directories.
The NPY feature files are created by --augment_clips in step 5 — not here.
"""
from __future__ import annotations
import os, random, shutil, subprocess
from pathlib import Path

from .common import (
    WORKSPACE, CUSTOM_NEG_DIR, HOME_NOISE_CLIP_S,
    log_ok, log_info, log_warn, log_step, log_section,
    fcount, ffcmd, symlink_dir,
    mark_done, save_state, get_state,
    wav_base, wav_aug,
)

NEG_TRAIN_DIR = WORKSPACE / "_neg_train"
NEG_TEST_DIR  = WORKSPACE / "_neg_test"


def run(model_name: str, n_samples: int = 100_000, force=False):
    log_section("Step 4 — Prepare Training Directories")

    mdir = WORKSPACE / "models" / model_name
    mdir.mkdir(parents=True, exist_ok=True)

    base = wav_base(model_name)
    aug  = wav_aug(model_name)

    n_aug  = fcount(aug)
    n_base = fcount(base)
    if n_aug == 0:
        log_warn("wav_aug/ empty — run step 3 first"); return False
    if n_base == 0:
        log_warn("wav_base/ empty — run step 3 first"); return False

    log_info(f"wav_aug : {n_aug:,} files")
    log_info(f"wav_base: {n_base:,} files")

    # positive_train: sample exactly n_samples files from wav_aug
    # OWW scans EVERY file in positive_train — 700k files = 5 hours on CPU.
    # We give it only what it needs: n_samples files (randomly sampled).
    pos_train = mdir / "positive_train"
    _populate_positive_train(pos_train, aug, n_samples, force)
    log_ok(f"positive_train -> {fcount(pos_train):,} files (sampled from {n_aug:,} in wav_aug/)")
    log_info(f"  wav_aug/ still has all {n_aug:,} files — not deleted")

    # positive_test: use all base clips (small, always fast)
    # Validate sample rates BEFORE symlinking — OWW crashes on wrong SR
    _validate_wav_samplerate(base)
    symlink_dir(mdir / "positive_test", base)
    n_base = fcount(base)  # recount after possible removals
    log_ok(f"positive_test  -> wav_base ({n_base:,})")

    # negative dirs: shared pool
    _ensure_neg_pool(force)
    # Clean any broken symlinks before OWW tries to read them
    _clean_broken_symlinks(NEG_TRAIN_DIR)
    _clean_broken_symlinks(NEG_TEST_DIR)
    symlink_dir(mdir / "negative_train", NEG_TRAIN_DIR)
    symlink_dir(mdir / "negative_test",  NEG_TEST_DIR)
    log_ok(f"negative_train -> shared ({fcount(NEG_TRAIN_DIR):,})")
    log_ok(f"negative_test  -> shared ({fcount(NEG_TEST_DIR):,})")

    mark_done("features", model_name)
    log_ok("Step 4 complete")
    return True


def _populate_positive_train(pos_train: Path, aug: Path,
                              n_samples: int, force: bool):
    """
    Symlink exactly n_samples files from wav_aug into positive_train.
    This prevents OWW from scanning 700k+ files (= hours of wasted I/O).
    Rebuilds only if the dir is missing, empty, or --force.
    """
    existing = fcount(pos_train) if pos_train.exists() else 0

    # Already populated with approximately the right number of files?
    # Accept if within 5% of target. Reject if way too many (old symlink to full wav_aug).
    if not force and existing > 0:
        ratio = existing / n_samples
        if 0.90 <= ratio <= 1.10:
            log_info(f"positive_train: {existing:,} files (cached)")
            return
        elif ratio > 1.10:
            log_info(f"positive_train has {existing:,} files but target is {n_samples:,} — rebuilding")
        else:
            log_info(f"positive_train has only {existing:,} / {n_samples:,} files — rebuilding")

    # Rebuild
    if pos_train.exists():
        if pos_train.is_symlink():
            pos_train.unlink()
        else:
            shutil.rmtree(str(pos_train))
    pos_train.mkdir(parents=True, exist_ok=True)

    # Collect all available WAV files
    all_wavs = [Path(e.path) for e in os.scandir(aug)
                if e.name.endswith(".wav")]
    if not all_wavs:
        log_warn("wav_aug/ is empty"); return

    # Sample n_samples (or all if fewer available)
    random.shuffle(all_wavs)
    selected = all_wavs[:n_samples]
    log_step(f"Populating positive_train: {len(selected):,} / {len(all_wavs):,} files")

    for w in selected:
        lnk = pos_train / w.name
        if not lnk.exists():
            try:
                lnk.symlink_to(w.resolve())
            except Exception:
                shutil.copy2(str(w), str(lnk))

    log_ok(f"positive_train populated: {fcount(pos_train):,} files")


def _ensure_neg_pool(force: bool):
    # Convert MP3s in custom_neg and split ALL long files to 5s clips
    CUSTOM_NEG_DIR.mkdir(parents=True, exist_ok=True)
    _split_long_files(CUSTOM_NEG_DIR, WORKSPACE / "negative_custom_clips")
    _split_home_noise()

    noise_dirs = _noise_sources()
    total = sum(fcount(d) for d in noise_dirs)
    cached = int(get_state("neg_pool_count") or 0)

    if (not force
            and fcount(NEG_TRAIN_DIR) > 0
            and fcount(NEG_TEST_DIR) > 0
            and cached == total):
        log_info(f"Negative pool cached: {fcount(NEG_TRAIN_DIR):,} + {fcount(NEG_TEST_DIR):,}")
        return

    log_step(f"Building negative pool ({total} noise files from {len(noise_dirs)} sources)")

    all_wavs = []
    import wave as _wave, contextlib as _ctx
    bad_sr = 0
    for d in noise_dirs:
        for e in os.scandir(d):
            if not e.name.endswith(".wav"):
                continue
            p = Path(e.path)
            try:
                with _ctx.closing(_wave.open(str(p), "rb")) as wf:
                    if wf.getframerate() == 16000:
                        all_wavs.append(p)
                    else:
                        bad_sr += 1
            except Exception:
                bad_sr += 1
    if bad_sr:
        log_warn(f"  Skipped {bad_sr} noise files with wrong sample rate (not 16kHz)")

    if not all_wavs:
        log_warn("No noise WAVs found"); return

    random.shuffle(all_wavs)
    split = int(len(all_wavs) * 0.8)

    for d in [NEG_TRAIN_DIR, NEG_TEST_DIR]:
        if d.exists() and not d.is_symlink():
            if force:
                log_info(f"Rebuilding {d.name}/ (--force)")
                shutil.rmtree(str(d))
            # else: keep existing files, just add new ones
        d.mkdir(parents=True, exist_ok=True)

    for wavs, d in [(all_wavs[:split], NEG_TRAIN_DIR),
                    (all_wavs[split:], NEG_TEST_DIR)]:
        for w in wavs:
            lnk = d / f"{w.parent.name}__{w.name}"
            if not lnk.exists():
                try: lnk.symlink_to(w.resolve())
                except Exception: shutil.copy2(str(w), str(lnk))

    save_state(neg_pool_count=len(all_wavs))
    _clean_broken_symlinks(NEG_TRAIN_DIR)
    _clean_broken_symlinks(NEG_TEST_DIR)
    log_ok(f"Negative pool: {fcount(NEG_TRAIN_DIR):,} train + {fcount(NEG_TEST_DIR):,} test")


def _clean_broken_symlinks(d: Path):
    """Remove symlinks pointing to missing files and invalid WAVs."""
    if not d.exists(): return
    removed = 0
    for entry in os.scandir(d):
        p = Path(entry.path)
        # Broken symlink
        if p.is_symlink() and not p.exists():
            p.unlink(); removed += 1; continue
        # Zero-size or tiny file
        if p.is_file() and p.stat().st_size < 100:
            p.unlink(); removed += 1; continue
    if removed:
        log_info(f"  Cleaned {removed} broken entries from {d.name}/")


def _validate_wav_samplerate(d: Path, expected_sr: int = 16000):
    """
    Scan all WAV files in d and remove (or re-encode) any that are not expected_sr.
    OWW raises ValueError if any clip has wrong sample rate.
    """
    if not d.exists():
        return
    import wave, contextlib
    bad = []
    for entry in os.scandir(d):
        p = Path(entry.path)
        if not p.name.endswith(".wav"):
            continue
        try:
            with contextlib.closing(wave.open(str(p), "rb")) as wf:
                sr = wf.getframerate()
            if sr != expected_sr:
                bad.append(p)
        except Exception:
            bad.append(p)  # unreadable = corrupt

    if not bad:
        return

    log_warn(f"  {len(bad)} WAV(s) with wrong sample rate in {d.name}/ — re-encoding to {expected_sr}Hz")
    for p in bad:
        tmp = p.with_suffix(".tmp.wav")
        ffcmd(["-i", str(p), "-ar", str(expected_sr), "-ac", "1", str(tmp)])
        if tmp.exists() and tmp.stat().st_size > 100:
            p.unlink()
            tmp.rename(p)
        else:
            log_warn(f"    Could not fix {p.name} — removing")
            p.unlink()
            if tmp.exists():
                tmp.unlink()


def _noise_sources():
    return [d for d in [
        WORKSPACE / "negative_custom_clips",  # long files already split
        WORKSPACE / "audioset_16k",
        WORKSPACE / "fma",
        WORKSPACE / "home_noise_clips",
        WORKSPACE / "chime6_clips",           # MUSAN noise (harder negatives)
    ] if d.exists() and fcount(d) > 0]


def _split_long_files(src_dir: Path, clips_dir: Path):
    """Convert + split all audio files in src_dir into 5s WAV clips."""
    audio_files = (list(src_dir.glob("*.wav")) +
                   list(src_dir.glob("*.mp3")) +
                   list(src_dir.glob("*.flac")) +
                   list(src_dir.glob("*.ogg")) +
                   list(src_dir.glob("*.m4a")))

    if not audio_files:
        return

    clips_dir.mkdir(parents=True, exist_ok=True)
    already = fcount(clips_dir)

    # estimate expected clips — skip if already looks complete
    if already > len(audio_files) * 10 and already > 100:
        log_info(f"negative_custom clips: {already:,} (cached)")
        return

    log_step(f"Splitting {len(audio_files)} files from negative_custom -> 5s clips")

    try:
        from tqdm import tqdm
        files_iter = tqdm(audio_files, desc="negative_custom")
    except ImportError:
        files_iter = audio_files

    for src in files_iter:
        # get duration
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(src)],
            capture_output=True, text=True)
        try:
            dur = float(r.stdout.strip())
        except Exception:
            dur = 0.0

        if dur < 1.0:
            continue

        n_clips = int(dur / HOME_NOISE_CLIP_S)
        if n_clips == 0:
            # short file — just convert as-is
            out = clips_dir / f"{src.stem}_full.wav"
            if not out.exists():
                ffcmd(["-i", str(src), "-ar", "16000", "-ac", "1", str(out)])
            continue

        for i in range(n_clips):
            out = clips_dir / f"{src.stem}_{i:05d}.wav"
            if out.exists():
                continue
            ffcmd(["-ss", str(i * HOME_NOISE_CLIP_S),
                   "-t",  str(HOME_NOISE_CLIP_S),
                   "-i",  str(src),
                   "-ar", "16000", "-ac", "1", str(out)])

    log_ok(f"negative_custom_clips: {fcount(clips_dir):,} clips")


def _split_home_noise():
    candidates = [
        WORKSPACE / "home_noise.wav",
        WORKSPACE.parent / "home_noise.wav",
    ]
    src = next((c for c in candidates
                if c.exists() and c.stat().st_size > 10_000_000), None)
    if src is None: return

    clips = WORKSPACE / "home_noise_clips"
    clips.mkdir(parents=True, exist_ok=True)
    if fcount(clips) > 100:
        log_info(f"Home noise clips: {fcount(clips):,} (cached)"); return

    log_step(f"Splitting home noise: {src.name}")
    r = subprocess.run(["ffprobe", "-v", "quiet",
                        "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(src)],
                       capture_output=True, text=True)
    try: total_s = float(r.stdout.strip())
    except Exception: return

    n = int(total_s / HOME_NOISE_CLIP_S)
    log_info(f"  {total_s/3600:.1f}h -> {n:,} clips")

    try:
        from tqdm import tqdm
        rng = tqdm(range(min(n, 10_000)), desc="Noise clips")
    except ImportError:
        rng = range(min(n, 10_000))

    for i in rng:
        out = clips / f"home_{i:05d}.wav"
        if out.exists(): continue
        ffcmd(["-ss", str(i * HOME_NOISE_CLIP_S), "-t", str(HOME_NOISE_CLIP_S),
               "-i", str(src), "-ar", "16000", "-ac", "1", str(out)])

    log_ok(f"Home noise: {fcount(clips):,} clips")
