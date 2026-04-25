#!/usr/bin/env python3
"""
WakeWordForge — run.py
======================
Main entry point. Can be used as:

  Interactive wizard:
    python run.py

  Full pipeline:
    python run.py --model agent_smith --he "אייג'נט סמית" --en "agent smith"

  Single step:
    python run.py --step 3 --model agent_smith --he "..." --en "..."

  Resume from last crash:
    python run.py --resume

  Force-redo a step:
    python run.py --step 4 --force

  Setup only:
    python run.py --step 1
    python run.py --step 1 --only venv deps oww piper

Steps:
  1  setup      — venv, deps, one-time downloads
  2  verify     — TTS preview + pronunciation check
  3  generate   — TTS base samples + augmentation
  4  features   — compile WAVs to NPY feature files
  5  train       — openWakeWord model training
  6  test        — live microphone test
  7  cleanup    — free disk space
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

# ── Make forge importable without installing ──────────────
sys.path.insert(0, str(Path(__file__).parent))

from forge.common import (
    log_title, log_section, log_ok, log_info, log_warn, log_err, log_box,
    is_done, get_state, save_state, reset_step,
    DEFAULT_N_SAMPLES, DEFAULT_N_STEPS, DEFAULT_PENALTY,
    WORKSPACE, CUSTOM_POS_DIR, CUSTOM_NEG_DIR, MY_RECORDINGS_DIR,
)
from forge import hardware

STEP_NAMES = {
    1: "Setup",
    2: "Verify",
    3: "Generate",
    4: "Features",
    5: "Train",
    6: "Test",
    7: "Cleanup",
}


# ══════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════
def main():
    args = _parse_args()

    log_title("WakeWordForge v1.0")

    # Detect hardware
    hw = hardware.detect()
    hardware.print_summary(hw)

    # Load or ask for config
    cfg = _resolve_config(args)

    if args.resume:
        cfg = _resume(cfg)

    # Ensure workspace dirs exist
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    CUSTOM_POS_DIR.mkdir(parents=True, exist_ok=True)
    CUSTOM_NEG_DIR.mkdir(parents=True, exist_ok=True)

    steps_to_run = _steps_to_run(args, cfg)
    log_box([f"Model    : {cfg['model_name']}",
             f"Hebrew   : {cfg['he_text']}",
             f"English  : {cfg['en_text']}",
             f"Samples  : {cfg['n_samples']:,}",
             f"Steps    : {cfg['n_steps']:,}",
             f"Running  : steps {steps_to_run}"])

    _run_steps(steps_to_run, cfg, hw, args)


# ══════════════════════════════════════════════════════════
#  Step dispatcher
# ══════════════════════════════════════════════════════════
def _run_steps(steps: list, cfg: dict, hw, args):
    name = cfg["model_name"]
    auto = args.auto

    # If --record flag given, run recording wizard before step 3
    if getattr(args, "record", False) and 3 in steps:
        from forge import recorder
        recorder.run(model_name=name, target=50)

    for step in steps:
        log_section(f"--- Step {step}: {STEP_NAMES[step]} ---")

        if step == 1:
            from forge import step1_setup
            only = args.only.split() if args.only else None
            step1_setup.run(force=args.force, only=only, skip_downloads=getattr(args,'skip_downloads',False))

        elif step == 2:
            from forge import step2_verify
            ok = step2_verify.run(
                model_name=name,
                he_text=cfg["he_text"],
                en_text=cfg["en_text"],
                auto=auto,
            )
            if not ok and not auto:
                log_warn("Stopping — fix pronunciation and re-run step 2")
                sys.exit(0)

        elif step == 3:
            from forge import step3_generate
            ok = step3_generate.run(
                model_name=name,
                he_text=cfg["he_text"],
                en_text=cfg["en_text"],
                n_target=cfg["n_samples"],
                force=args.force,
                ipa_text=cfg.get("ipa_text", ""),
            )
            if not ok:
                log_err("Step 3 failed"); sys.exit(1)

        elif step == 4:
            from forge import step4_features
            step4_features.run(
                model_name=name,
                n_samples=cfg.get("n_samples", 100_000),
                force=args.force,
            )

        elif step == 5:
            from forge import step5_train
            # target: CLI --target overrides wizard cfg["output_target"]
            output_target = (getattr(args, "target", None)
                             or cfg.get("output_target", "both"))
            ok = step5_train.run(
                model_name=name,
                hw=hw,
                n_samples=cfg["n_samples"],
                n_steps=cfg["n_steps"],
                penalty=cfg["penalty"],
                force=args.force,
                en_text=cfg.get("en_text", ""),
                he_text=cfg.get("he_text", ""),
                arch=getattr(args, "arch", "open"),
                target=output_target,
            )
            if not ok:
                log_err("Step 5 failed"); sys.exit(1)

        elif step == 6:
            from forge import step6_test
            step6_test.run(
                model_name=name,
                threshold=cfg.get("threshold", 0.5),
                duration=cfg.get("test_duration", 60),
            )

        elif step == 7:
            from forge import step7_cleanup
            step7_cleanup.run(
                model_name=name,
                level=cfg.get("cleanup_level", "medium"),
                auto=auto,
            )

    log_ok("All steps complete")


# ══════════════════════════════════════════════════════════
#  Config resolution
# ══════════════════════════════════════════════════════════
def _resolve_config(args) -> dict:
    """Return config dict from CLI args, state file, or interactive wizard."""
    # If all required args provided, use them
    if args.model and args.he and args.en:
        from forge.model_config import load as _load_cfg
        cfg = _load_cfg(
            model_name  = args.model.strip().replace(" ", "_"),
            cli_en      = args.en,
            cli_he      = args.he,
            cli_samples = args.samples,
            cli_steps   = args.steps,
        )
        cfg["n_samples"] = cfg["samples"]
        cfg["n_steps"]   = cfg["steps"]
        cfg["penalty"]   = getattr(args, "penalty", cfg.get("penalty", 5000))
        cfg["ipa_text"]  = getattr(args, "ipa", "") or ""
        return cfg

    # Try to load from state (resume scenario)
    if args.resume:
        cfg = {
            "model_name": get_state("model_name"),
            "he_text":    get_state("he_text"),
            "en_text":    get_state("en_text"),
            "n_samples":  get_state("n_samples", DEFAULT_N_SAMPLES),
            "n_steps":    get_state("n_steps",   DEFAULT_N_STEPS),
            "penalty":    get_state("penalty",   DEFAULT_PENALTY),
        }
        if cfg["model_name"]:
            log_info(f"Resuming model: {cfg['model_name']}")
            return cfg

    # Interactive wizard
    return _wizard(args)


def _wizard(args) -> dict:
    log_section("Configuration Wizard")
    print()

    model = input("  Wake word model name (letters/digits/_): ").strip().replace(" ", "_")
    if not model:
        model = "my_wake_word"

    he = input(f"  Hebrew text for '{model}': ").strip()
    if not he: he = model

    en = input(f"  English text for '{model}': ").strip()
    if not en: en = model

    # ── Recording mode ─────────────────────────────────────────────────────────
    n_existing = len(list(MY_RECORDINGS_DIR.glob("*.wav"))) if MY_RECORDINGS_DIR.exists() else 0
    print()
    print("  Voice samples:")
    print("    1  Record now   — guided mic session (~5 min, 50 takes)")
    if n_existing:
        print(f"    2  Use existing — {n_existing} recordings already in my_recordings/  [recommended]")
    else:
        print("    2  Skip         — I'll add recordings to my_recordings/ later")
    print("    3  TTS only     — no personal recordings  (lower quality, faster)")
    print()
    rec_choice = input("  Mode [1/2/3, default 2]: ").strip() or "2"

    if rec_choice == "1":
        from forge import recorder
        recorder.run(model_name=model, target=50)
        n_existing = len(list(MY_RECORDINGS_DIR.glob("*.wav"))) if MY_RECORDINGS_DIR.exists() else 0
        if n_existing:
            log_ok(f"  {n_existing} recordings ready")
        else:
            log_warn("  No recordings saved — will use TTS-only mode")
    elif rec_choice == "3":
        log_info("  TTS-only mode selected")
    else:
        if n_existing:
            log_ok(f"  Using {n_existing} existing recordings from my_recordings/")
        else:
            log_info("  No recordings found — will use TTS-only mode")

    # ── Output target ───────────────────────────────────────────────────────────
    print()
    print("  Output target:")
    print("    1  both  — ONNX (openWakeWord/HA) + TFLite (microWakeWord/ESP32)  [default]")
    print("    2  oww   — ONNX only  (openWakeWord / wyoming-openwakeword)")
    print("    3  mww   — TFLite only  (microWakeWord / ESP32)")
    target_choice = input("  Target [1/2/3, default 1]: ").strip() or "1"
    target_map = {"1": "both", "2": "oww", "3": "mww"}
    output_target = target_map.get(target_choice, "both")

    def _int(prompt, default):
        try: return int(input(f"  {prompt} [{default}]: ").strip() or default)
        except ValueError: return default

    print()
    n_samples = _int("Number of augmented samples", DEFAULT_N_SAMPLES)
    n_steps   = _int("Training steps",              DEFAULT_N_STEPS)
    penalty   = _int("False-activation penalty",    DEFAULT_PENALTY)

    cfg = {
        "model_name":    model,
        "he_text":       he,
        "en_text":       en,
        "n_samples":     n_samples,
        "n_steps":       n_steps,
        "penalty":       penalty,
        "output_target": output_target,
    }
    # Persist for resume
    save_state(**cfg)
    return cfg


def _resume(cfg: dict) -> dict:
    """Fill any missing cfg fields from state file."""
    saved = {
        "model_name": get_state("model_name"),
        "he_text":    get_state("he_text"),
        "en_text":    get_state("en_text"),
        "n_samples":  get_state("n_samples", DEFAULT_N_SAMPLES),
        "n_steps":    get_state("n_steps",   DEFAULT_N_STEPS),
        "penalty":    get_state("penalty",   DEFAULT_PENALTY),
    }
    for k, v in saved.items():
        if v and not cfg.get(k):
            cfg[k] = v
    return cfg


# ══════════════════════════════════════════════════════════
#  Step selection logic
# ══════════════════════════════════════════════════════════
def _steps_to_run(args, cfg: dict) -> list:
    name = cfg.get("model_name", "")

    # Explicit single step
    if args.step:
        return [args.step]

    # Explicit range
    if args.from_step:
        start = args.from_step
        end   = args.to_step or 7
        return list(range(start, end + 1))

    # Resume: find first incomplete step
    if args.resume:
        for s in range(1, 8):
            step_key = {
                1: ("setup",    ""),
                2: ("verify",   name),
                3: ("generate", name),
                4: ("features", name),
                5: ("train",    name),
                6: ("test",     name),
                7: ("cleanup",  name),
            }[s]
            if not is_done(*step_key):
                log_info(f"Resuming from step {s}: {STEP_NAMES[s]}")
                return list(range(s, 8))
        log_ok("All steps already complete")
        return []

    # Default: all steps
    return list(range(1, 8))


# ══════════════════════════════════════════════════════════
#  Argument parser
# ══════════════════════════════════════════════════════════
def _parse_args():
    p = argparse.ArgumentParser(
        prog="python run.py",
        description="WakeWordForge — wake word training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Identity
    p.add_argument("--model",   help="Model name (letters/digits/_)")
    p.add_argument("--he",      help="Hebrew text for TTS. Use | to separate multiple phrases: 'phrase one|phrase two'")
    p.add_argument("--en",      help="English text for TTS. Use | to separate multiple phrases: 'phrase one|phrase two'")
    p.add_argument("--ipa",     help="IPA phonetic text (e.g. \"maʁˈʔa\"). Uses espeak-ng to synthesise sounds that English TTS cannot produce.")

    # Training parameters
    p.add_argument("--samples",  type=int, default=DEFAULT_N_SAMPLES,
                   help=f"Augmented samples target (default {DEFAULT_N_SAMPLES:,})")
    p.add_argument("--steps",    type=int, default=DEFAULT_N_STEPS,
                   help=f"Training steps (default {DEFAULT_N_STEPS:,})")
    p.add_argument("--penalty",  type=int, default=DEFAULT_PENALTY,
                   help=f"False-activation penalty (default {DEFAULT_PENALTY:,})")
    p.add_argument("--arch",     choices=["open", "micro"], default="open",
                   help="Model architecture: open=192-unit DNN (default), micro=64-unit DNN (mobile/edge)")
    p.add_argument("--target",   choices=["oww", "mww", "both"], default="both",
                   help="Output format: oww=openWakeWord ONNX only, mww=microWakeWord TFLite only, both=ONNX+TFLite (default)")

    # Step control
    p.add_argument("--step",       type=int, choices=range(1, 8),
                   help="Run a single step (1-7)")
    p.add_argument("--from-step",  type=int, dest="from_step",
                   help="Run steps from N to end (or --to-step)")
    p.add_argument("--to-step",    type=int, dest="to_step",
                   help="Stop at step N (use with --from-step)")
    p.add_argument("--only",       help="For step 1: space-separated subset "
                                        "(venv deps oww piper audioset rirs fma acav val)")
    p.add_argument("--skip-downloads", action="store_true", dest="skip_downloads",
                   help="Step 1: skip AudioSet/FMA/MIT-RIR downloads (use if downloads fail)")

    # Behaviour
    p.add_argument("--resume",  action="store_true",
                   help="Continue from last completed step")
    p.add_argument("--force",   action="store_true",
                   help="Re-run step even if already complete")
    p.add_argument("--auto",    action="store_true",
                   help="Non-interactive (skip confirmations)")
    p.add_argument("--record",  action="store_true",
                   help="Launch microphone recording wizard before training")

    return p.parse_args()


if __name__ == "__main__":
    main()
