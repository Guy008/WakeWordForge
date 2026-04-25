"""
Model configuration loader.
Reads models/{name}.yaml and provides typed access to all settings.
Falls back to CLI args if no YAML found.
"""
from pathlib import Path
import yaml
from .common import log_info, log_ok, log_warn, WORKSPACE

MODELS_DIR = Path(__file__).parent.parent / "models"


def load(model_name: str, cli_en: str = "", cli_he: str = "",
         cli_samples: int = 0, cli_steps: int = 0) -> dict:
    """
    Load model config from models/{name}.yaml.
    CLI args override YAML values when provided.
    Returns a unified config dict used by all steps.
    """
    yaml_path = MODELS_DIR / f"{model_name}.yaml"

    cfg = _defaults(model_name)

    if yaml_path.exists():
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        cfg = _merge(cfg, data)
        log_ok(f"  Loaded model config: {yaml_path.name}")
    else:
        log_info(f"  No YAML config found — using CLI args")
        log_info(f"  Tip: create models/{model_name}.yaml for full control")

    # CLI overrides
    if cli_en:    cfg["en_primary"] = cli_en
    if cli_he:    cfg["he_primary"] = cli_he
    if cli_samples: cfg["samples"] = cli_samples
    if cli_steps:   cfg["steps"]   = cli_steps

    # Build flat positive/negative lists
    cfg["positive_en"] = cfg.get("positive", {}).get("en", [cli_en] if cli_en else [])
    cfg["positive_he"] = cfg.get("positive", {}).get("he", [cli_he] if cli_he else [])
    cfg["negative_en"] = cfg.get("negative", {}).get("en", [])
    cfg["negative_he"] = cfg.get("negative", {}).get("he", [])

    # Primary texts for step2 preview
    cfg["en_text"] = cfg["positive_en"][0] if cfg["positive_en"] else cli_en
    cfg["he_text"] = cfg["positive_he"][0] if cfg["positive_he"] else cli_he

    # STT keywords — auto-generate from positive phrases if not set
    stt = cfg.get("stt", {})
    if not stt.get("keywords"):
        keywords = set()
        for phrase in cfg["positive_en"] + cfg["positive_he"]:
            for w in phrase.lower().split():
                if len(w) > 3:
                    keywords.add(w)
        cfg["stt_keywords"] = sorted(keywords)
    else:
        cfg["stt_keywords"] = stt["keywords"]

    cfg["stt_enabled"] = stt.get("enabled", True)
    cfg["stt_model"]   = stt.get("model", "small.en")
    cfg["stt_device"]  = stt.get("device", "cuda")

    return cfg


def _defaults(model_name: str) -> dict:
    return {
        "model_name": model_name,
        "samples": 50000,
        "steps": 100000,
        "penalty": 5000,
        "positive": {"en": [], "he": []},
        "negative": {"en": [], "he": []},
        "stt": {"enabled": True, "model": "small.en", "device": "cuda", "keywords": []},
        "tts": {
            "edge_tts": {"enabled": True},
            "piper_positive": {"enabled": True, "clips_per_text": 50},
            "piper_negative": {"enabled": True},
            "google_tts": {"enabled": True},
            "wyoming": {"enabled": False},
            "speecht5": {"enabled": False},
            "phonikud": {"enabled": False},
        },
    }


def _merge(base: dict, override: dict) -> dict:
    """Deep merge override into base."""
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _merge(result[k], v)
        else:
            result[k] = v
    return result


def all_positive_texts(cfg: dict) -> tuple:
    """Returns (he_texts, en_texts) — all phonetic variants."""
    he = cfg.get("positive_he", [cfg.get("he_text", "")])
    en = cfg.get("positive_en", [cfg.get("en_text", "")])
    return [t for t in he if t], [t for t in en if t]


def all_negative_texts(cfg: dict) -> list:
    """Returns flat list of all negative phrases."""
    return cfg.get("negative_en", []) + cfg.get("negative_he", [])


def stt_accepts(transcript: str, cfg: dict) -> bool:
    """
    Returns True if transcript is an acceptable wake word utterance.
    Uses both keyword matching AND negative phrase rejection.
    """
    t = transcript.lower().strip()
    if not t:
        return False

    # Reject if matches a known negative phrase
    for neg in all_negative_texts(cfg):
        n = neg.lower().strip()
        if n and (n in t or t in n):
            return False  # definitely wrong

    # Accept if matches any positive phrase (50%+ word overlap)
    for pos in cfg.get("positive_en", []) + cfg.get("positive_he", []):
        p = pos.lower().strip()
        if not p: continue
        if p in t or t in p: return True
        p_words = set(p.split())
        t_words = set(t.split())
        if p_words and len(p_words & t_words) / len(p_words) >= 0.5:
            return True

    # Accept — keyword match (relaxed — Whisper may spell differently from input)
    keywords = cfg.get("stt_keywords", [])
    if keywords:
        found = sum(1 for kw in keywords if kw in t)
        if found >= 1: return True
        # Partial phonetic match (first 4 chars of each keyword)
        short = [kw[:4] for kw in keywords if len(kw) >= 4]
        if short and sum(1 for kw in short if kw in t) >= 2:
            return True

    return False


def expand_word_parts(cfg: dict) -> tuple:
    """
    Generate all prefix×suffix combinations from word_parts config.
    Returns (positive_combos, negative_combos) as flat lists of strings.

    Example:
      prefix_en: ["hey jent", "agent"]
      suffix_en: ["smeet", "smitt"]
      → ["hey jent smeet", "hey jent smitt", "agent smeet", "agent smitt"]
    """
    parts = cfg.get("word_parts", {})
    if not parts:
        return [], []

    def combine(prefixes, suffixes):
        result = []
        for p in (prefixes or []):
            for s in (suffixes or []):
                combo = f"{p} {s}".strip()
                if combo:
                    result.append(combo)
        return result

    pos = parts.get("positive", {})
    neg = parts.get("negative", {})

    pos_combos = (
        combine(pos.get("prefix_en", []), pos.get("suffix_en", [])) +
        combine(pos.get("prefix_he", []), pos.get("suffix_he", []))
    )
    neg_combos = (
        combine(neg.get("prefix_en", []), neg.get("suffix_en", [])) +
        combine(neg.get("prefix_he", []), neg.get("suffix_he", []))
    )

    # Remove duplicates while preserving order
    seen = set()
    pos_unique = [x for x in pos_combos if not (x in seen or seen.add(x))]
    seen = set()
    neg_unique = [x for x in neg_combos if not (x in seen or seen.add(x))]

    return pos_unique, neg_unique
