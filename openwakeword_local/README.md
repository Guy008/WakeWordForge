# WakeWordForge

**A wake word factory.** Give it a word, get a model. Works for everyone.

You put your wake word in one end — the tool records samples, generates training data,
trains a model, and hands you a ready-to-use file for Home Assistant or ESP32.

Built on top of [openWakeWord](https://github.com/dscripka/openWakeWord).

---

## What you get

| File | Use |
|------|-----|
| `model_name.onnx` | [wyoming-openwakeword](https://github.com/rhasspy/wyoming-openwakeword) on Home Assistant server |
| `model_name.tflite` | [microWakeWord](https://github.com/kahrendt/microWakeWord) on ESP32-S3 / ESPHome |

You can request one or both:  `--target oww` / `--target mww` / `--target both` (default).

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/YOUR_USER/WakeWordForge
cd WakeWordForge

# 2. Install (Linux / macOS)
bash install.sh

# 3a. Web UI — easiest option (opens in browser automatically)
python3 web_ui.py

# 3b. Or train interactively in the terminal
python3 run.py
```

### Web UI

```bash
python3 web_ui.py
# Opens http://localhost:7860
```

Fill in the form, click **Start Training**, and watch the live terminal output.
When training finishes, download your model files directly from the browser.

The Web UI auto-detects the project venv and installs Flask if needed — no manual setup required.

The wizard (terminal mode) will ask:
1. Wake word name + text
2. Record samples now / use existing recordings / TTS-only
3. Output target (onnx / tflite / both)
4. Training parameters (defaults are fine for most users)

---

## Installation

### Linux / macOS

```bash
bash install.sh
```

Requires: `python3` (3.10+), `git`, `ffmpeg`.
The script installs missing system packages automatically where possible.

### Windows

```bat
install.bat
```

Requires: [Python 3.10+](https://www.python.org/downloads/), [git](https://git-scm.com/download/win), [ffmpeg](https://ffmpeg.org/download.html).

### Manual

```bash
python3 run.py --step 1
```

Step 1 creates `workspace/venv/`, installs all Python dependencies, and downloads
training data (~2-4 GB total — one time only).

---

## System requirements

| | Minimum | Recommended |
|---|---|---|
| OS | Linux / macOS / Windows | Ubuntu 22.04 |
| Python | 3.10 | 3.11 |
| RAM | 8 GB | 16 GB |
| GPU | none (CPU works) | NVIDIA CUDA 12+ |
| Disk | 15 GB free | 25 GB |
| ffmpeg | required | required |

---

## Usage

### Interactive wizard (recommended for first-time users)

```bash
python3 run.py
```

The wizard guides you through every step and asks all necessary questions.

### One-line non-interactive training

```bash
python3 run.py \
  --model hey_gadi \
  --he "היי גדי" \
  --en "hey gadi" \
  --target both
```

### With your own recordings (best quality)

```bash
# Put your WAV recordings in my_recordings/
# (or use --record to record now)
python3 run.py \
  --model hey_gadi \
  --he "היי גדי" \
  --en "hey gadi" \
  --record \
  --target both
```

### Run a single step

```bash
python3 run.py --step 2 --model hey_gadi --he "היי גדי" --en "hey gadi"
```

### Resume after crash

```bash
python3 run.py --resume
```

---

## Recording your own voice (strongly recommended)

Personal recordings beat TTS every time. Even 20 takes dramatically improve accuracy.

**Option 1 — Built-in recording wizard:**
```bash
python3 run.py --record --model my_word --he "..." --en "..."
```
The wizard counts down, records each take, shows a live level meter, and lets you redo.

**Option 2 — Add recordings manually:**
1. Record 50 WAV or MP3 files of your wake word (any sample rate, mono or stereo)
2. Drop them into `my_recordings/`
3. Run training — WakeWordForge converts them automatically

**Recording tips:**
- Vary your distance to the mic (close, normal, slightly far)
- Record some takes quietly and some at normal volume
- Record in the room where the device will be used
- Try different speeds (a bit faster, normal, a bit slower)
- 20-50 takes is enough; more is better

**Without recordings (TTS-only mode):**
WakeWordForge falls back to generating voice samples from 12 edge-tts voices.
Models trained this way work but are less accurate than recording-based ones.

---

## CLI flags reference

| Flag | Default | Description |
|------|---------|-------------|
| `--model NAME` | — | Model name (letters, digits, `_`) |
| `--he TEXT` | — | Hebrew text (or phonetic text for Hebrew TTS) |
| `--en TEXT` | — | English text. Use `\|` to separate multiple variants |
| `--target` | `both` | Output: `oww` (ONNX only), `mww` (TFLite only), `both` |
| `--arch` | `open` | Architecture: `open` (192-unit, HA server), `micro` (64-unit, ESP32) |
| `--samples N` | 100,000 | Augmented training samples |
| `--steps N` | 100,000 | Training steps |
| `--penalty N` | 5,000 | False-activation penalty (higher = fewer false triggers) |
| `--record` | false | Launch mic recording wizard before training |
| `--step N` | — | Run only step N (1-7) |
| `--from-step N` | — | Start from step N (runs N through 7) |
| `--resume` | false | Continue from last completed step |
| `--force` | false | Re-run step even if already completed |
| `--auto` | false | Skip interactive confirmations |
| `--skip-downloads` | false | Step 1: skip large file downloads |

---

## Pipeline steps

| # | Name | What it does | Typical time |
|---|------|-------------|-------------|
| 1 | Setup | Create venv, install deps, download training data | 20-60 min (once) |
| 2 | Verify | Play TTS preview — check pronunciation before training | seconds |
| 3 | Generate | Import recordings + TTS + intensive augmentation | 30-90 min |
| 4 | Features | Build training directory structure | seconds |
| 5 | Train | Feature extraction + DNN training + ONNX/TFLite export | 2-6 hours |
| 6 | Test | Live microphone test | manual |
| 7 | Cleanup | Free disk space (keeps model files) | seconds |

---

## Choosing a good wake word

**What makes a wake word work well:**
- 3-5 syllables — single syllables cause many false triggers
- Unique sounds — strong consonants like K, SH, T, G work better than soft ones
- Not a common word — "shalom" appears in every conversation; "shalo-mag" does not
- Phonetically clear — the model should recognize it even with noise

**Good examples:**
```
"hey gadi"        (3 syllables, clear consonants)
"beit ohr"        (3 syllables)
"hey computer"    (4 syllables)
"activate forge"  (5 syllables)
```

**Poor examples:**
```
"ok"              (too short — many false triggers)
"shalom"          (too common in normal speech)
"home"            (1 syllable — false triggers constantly)
```

---

## Architecture: OPEN vs MICRO

| | OPEN (default) | MICRO |
|--|--|--|
| File size | ~1.3 MB (ONNX) | ~0.3 MB (TFLite) |
| Accuracy | High | Slightly lower |
| Latency | ~2 ms | ~0.5 ms |
| Use on | Home Assistant, PC, RPi 4 | ESP32-S3, RPi Zero |

```bash
# OPEN (default) — for HA / server
python3 run.py --model my_word --en "..." --arch open

# MICRO — for ESP32 / edge devices
python3 run.py --model my_word --en "..." --arch micro
```

---

## Deploying to Home Assistant

### wyoming-openwakeword (ONNX)

1. Copy `workspace/ha_deploy/MODEL_NAME/MODEL_NAME.onnx` to your
   wyoming-openwakeword `custom_models/` directory.
2. Restart the wyoming-openwakeword add-on.
3. In HA: Settings → Voice Assistants → select your wake word.

```bash
# Files are also copied automatically:
ls workspace/ha_deploy/MODEL_NAME/
# MODEL_NAME.onnx       ← wyoming-openwakeword (HA server)
# MODEL_NAME.tflite     ← raw TFLite (for reference / manual use)
# MODEL_NAME.json       ← microWakeWord manifest (ESPHome, self-contained)
# DEPLOY.txt            ← exact deployment instructions
```

### microWakeWord (TFLite / ESPHome)

ESPHome requires a **JSON manifest** file — not the `.tflite` directly.
WakeWordForge generates it automatically. The JSON embeds the model as base64
so you only need to copy one file.

1. Copy `workspace/ha_deploy/MODEL_NAME/MODEL_NAME.json` to your ESPHome config:
   `/config/custom_components/micro_wake_word/models/`
2. Reference it in your device YAML:

```yaml
micro_wake_word:
  model: /config/custom_components/micro_wake_word/models/MODEL_NAME.json
```

> The `.tflite` file is also provided for tools that accept raw TFLite input,
> but ESPHome needs the `.json` manifest.

---

## Resuming interrupted training

```bash
# Automatic resume from last completed step
python3 run.py --resume

# Force-restart a specific step
python3 run.py --step 5 --model my_word --en "..." --force
```

Step 5 (training) saves checkpoint NPY feature files. If interrupted:
- All 4 NPY files present → restarts only the DNN training (~10 min)
- NPY files missing → full restart (feature extraction + training)

---

## Project structure

```
WakeWordForge/
├── run.py                      # entry point (TUI / CLI)
├── web_ui.py                   # Web UI — open http://localhost:7860
├── install.sh                  # Linux/macOS installer
├── install.bat                 # Windows installer
├── my_recordings/              # put your WAV recordings here
│   └── speaker01_take01.wav    # naming: any name, any format
├── forge/
│   ├── common.py               # constants, logging, state
│   ├── hardware.py             # GPU/CPU/RAM detection
│   ├── recorder.py             # microphone recording wizard
│   ├── step1_setup.py          # install deps + download data
│   ├── step2_verify.py         # TTS pronunciation preview
│   ├── step3_generate.py       # samples + augmentation
│   ├── step3b_stt_filter.py    # Whisper quality filter
│   ├── step4_features.py       # training directory setup
│   ├── step5_train.py          # DNN training + export
│   ├── step6_test.py           # live mic test
│   └── step7_cleanup.py        # free disk space
└── workspace/                  # created at runtime
    ├── venv/                   # Python environment
    ├── models/MODEL_NAME/      # WAV base + augmented
    ├── ha_deploy/MODEL_NAME/   # ready-to-deploy files
    └── ...
```

---

## FAQ

**Q: Can I train without a GPU?**
Yes. CPU training works but is 5-10x slower. A model that takes 2 hours on RTX 2060
may take 12-16 hours on CPU. Use `--samples 20000 --steps 50000` to reduce time.

**Q: Can I train in Hebrew?**
Yes. Pass `--he "מילה שלי"` and `--en "mila sheli"` (phonetic transliteration).
The Hebrew text is used with the he-IL edge-tts voices. The English text controls
pronunciation quality — use a phonetic transliteration for best results.

**Q: How many recordings do I need?**
20 recordings produce a usable model. 50 recordings produce a good model.
100+ recordings produce an excellent model. TTS-only (0 recordings) works
but accuracy is noticeably lower.

**Q: My model has too many false triggers. What to do?**
Increase `--penalty` (try 20,000-50,000) and retrain step 5 only:
```bash
python3 run.py --step 5 --model my_word --en "..." --penalty 30000 --force
```

**Q: My model misses detections. What to do?**
Lower `--penalty` (try 1,000-3,000) and retrain, or add more recordings and
retrain from step 3.

**Q: Can I train multiple wake words?**
Yes. Use a different `--model` name for each. Steps 1 (setup) and data downloads
are shared — only steps 3-5 are model-specific.

---

## License

MIT

---

---

# עברית — WakeWordForge

**מפעל מילות השכמה.** אתה מכניס מילה, מקבל מודל.

הכלי מטפל בכל השאר: מקליט דוגמאות קוליות, יוצר נתוני אימון, מאמן מודל,
ומחזיר קבצים מוכנים ל-Home Assistant או ESP32.

---

## התחלה מהירה

```bash
# 1. הורד
git clone https://github.com/YOUR_USER/WakeWordForge
cd WakeWordForge

# 2. התקן
bash install.sh

# 3א. ממשק ווב — הכי קל (נפתח בדפדפן אוטומטית)
python3 web_ui.py

# 3ב. או wizard בטרמינל
python3 run.py
```

### ממשק ווב

```bash
python3 web_ui.py
# נפתח ב-http://localhost:7860
```

מלא את הטופס, לחץ **Start Training**, וצפה בפלט החי.
כשהאימון מסתיים — הורד את קבצי המודל ישירות מהדפדפן.

---

## בחירת מילת השכמה טובה

**מה גורם למילה לעבוד טוב:**
- **3-5 הברות** — מילה אחת = הרבה הפעלות שגויות
- **צלילים ייחודיים** — "ק", "ש", "ג", "ט" עדיפים על צלילים רכים
- **לא מופיעה בשיחה רגילה** — "שלום" גרוע מאוד; "שלו-מג" עובד נהדר
- **ברורה פונטית** — המודל צריך לזהות גם עם רעש

**דוגמאות טובות:** `היי גדי`, `בוקר אור`, `hey computer`, `activate forge`

**דוגמאות גרועות:** `אוקיי`, `שלום`, `בית`

---

## הקלטות אישיות

הקלטות שלך עדיפות על TTS תמיד. גם 20 הקלטות משפרות דרמטית.

**אפשרות 1 — wizard מובנה:**
```bash
python3 run.py --record --model my_word --he "..." --en "..."
```

**אפשרות 2 — הוסף ידנית:**
1. הקלט 50 קבצי WAV/MP3 של מילת ההשכמה
2. שים אותם בתיקיית `my_recordings/`
3. הרץ אימון — WakeWordForge ממיר הכל אוטומטית

**טיפים:**
- שנה מרחק מהמיקרופון בין הקלטות
- הקלט בעצמות ובעוצמות שונות
- הקלט בחדר שבו תשתמש במכשיר
- 20-50 הקלטות מספיקות; יותר = טוב יותר

---

## פריסה ל-Home Assistant

### wyoming-openwakeword (ONNX)
העתק את `MODEL_NAME.onnx` לתיקיית `custom_models/` של wyoming-openwakeword.

### microWakeWord (TFLite / ESPHome)
WakeWordForge מייצר קובץ **JSON manifest** שמכיל את המודל מוטמע בתוכו.
ESPHome דורש את הקובץ הזה — לא את ה-TFLite ישירות.

1. העתק את `MODEL_NAME.json` לתיקיה:
   `/config/custom_components/micro_wake_word/models/`
2. ציין אותו ב-ESPHome:
```yaml
micro_wake_word:
  model: /config/custom_components/micro_wake_word/models/MODEL_NAME.json
```

הקבצים נמצאים ב: `workspace/ha_deploy/MODEL_NAME/`

---

## דרישות מערכת

- Python 3.10+
- ffmpeg
- git
- GPU NVIDIA מומלץ (בלי GPU עובד אבל איטי פי 5-10)
- 8GB RAM מינימום, 16GB מומלץ
- 15GB דיסק פנוי
