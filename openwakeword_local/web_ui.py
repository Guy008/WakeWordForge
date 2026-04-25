#!/usr/bin/env python3
"""
WakeWordForge Web UI
Single-file web application — no build tools needed.

Run:   python3 web_ui.py
Open:  http://localhost:7860
"""
from __future__ import annotations
import sys, os, re, json, queue, threading, subprocess
from pathlib import Path

HERE        = Path(__file__).parent.resolve()
VENV_PYTHON = HERE / "workspace" / "venv" / "bin" / "python3"
PORT        = 7860

# ── Auto-reexec inside workspace venv (has all training deps) ─────────────────
if VENV_PYTHON.exists() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON)] + sys.argv)

# ── Ensure Flask is installed ─────────────────────────────────────────────────
try:
    from flask import Flask, Response, request, send_file, jsonify, abort
except ImportError:
    print("[web_ui] Flask not found — installing…")
    subprocess.run([sys.executable, "-m", "pip", "install", "flask", "-q"], check=True)
    from flask import Flask, Response, request, send_file, jsonify, abort  # type: ignore

# ── App globals ───────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

_lock        = threading.Lock()
_buf_lock    = threading.Lock()
_log_buffer: list[str]                    = []
_subscribers: list[queue.Queue[str|None]] = []
_proc:  subprocess.Popen | None           = None
_status = "idle"   # idle | running | done | error

# ── Log broadcast helpers ─────────────────────────────────────────────────────
def _emit_line(line: str):
    with _buf_lock:
        _log_buffer.append(line)
        for sq in _subscribers:
            sq.put(line)

def _finish(rc: int):
    global _status
    with _lock:
        _status = "done" if rc == 0 else "error"
    with _buf_lock:
        for sq in _subscribers:
            sq.put(None)          # sentinel → each SSE stream closes

def _stream_proc(proc: subprocess.Popen):
    try:
        for raw in iter(proc.stdout.readline, ""):
            _emit_line(raw.rstrip("\n"))
        proc.wait()
    finally:
        _finish(proc.returncode or 1)

# ── API ───────────────────────────────────────────────────────────────────────
@app.post("/api/start")
def api_start():
    global _proc, _status, _log_buffer, _subscribers
    with _lock:
        if _status == "running":
            return jsonify(error="Already running"), 409

        data = request.get_json(force=True, silent=True) or {}
        if not data.get("model"):
            return jsonify(error="model name is required"), 400

        cmd = [sys.executable, str(HERE / "run.py"), "--auto"]
        cmd += ["--model",  data["model"]]
        if data.get("he"):      cmd += ["--he",       data["he"]]
        if data.get("en"):      cmd += ["--en",       data["en"]]
        cmd += ["--target", data.get("target", "both")]
        cmd += ["--arch",   data.get("arch",   "open")]
        if data.get("samples"): cmd += ["--samples",  str(int(data["samples"]))]
        if data.get("steps"):   cmd += ["--steps",    str(int(data["steps"]))]
        if data.get("penalty"): cmd += ["--penalty",  str(int(data["penalty"]))]
        if data.get("from_step"):
            cmd += ["--from-step", str(int(data["from_step"]))]
        if data.get("record"):  cmd += ["--record"]

        with _buf_lock:
            _log_buffer.clear()
            _subscribers.clear()

        _status = "running"
        _proc = subprocess.Popen(
            cmd, cwd=str(HERE),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        threading.Thread(target=_stream_proc, args=(_proc,), daemon=True).start()

    return jsonify(ok=True, cmd=" ".join(cmd))


@app.post("/api/stop")
def api_stop():
    global _status
    with _lock:
        if _proc and _proc.poll() is None:
            _proc.terminate()
            _status = "idle"
    return jsonify(ok=True)


@app.get("/api/status")
def api_status():
    return jsonify(status=_status)


@app.get("/api/logs")
def api_logs():
    """SSE — streams every log line to the browser. Replays buffer on reconnect."""
    sq: queue.Queue[str | None] = queue.Queue()
    with _buf_lock:
        buffered     = list(_log_buffer)
        already_done = _status != "running" and len(_log_buffer) > 0
        _subscribers.append(sq)

    def generate():
        yield "retry: 3000\n\n"
        # Replay what we already have
        for line in buffered:
            yield f"data: {json.dumps({'type':'log','text':line}, ensure_ascii=False)}\n\n"
        if already_done:
            yield f"data: {json.dumps({'type':'done','status':_status}, ensure_ascii=False)}\n\n"
            with _buf_lock:
                try: _subscribers.remove(sq)
                except ValueError: pass
            return
        # Stream new lines
        while True:
            try:
                item = sq.get(timeout=25)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue
            if item is None:
                yield f"data: {json.dumps({'type':'done','status':_status}, ensure_ascii=False)}\n\n"
                break
            yield f"data: {json.dumps({'type':'log','text':item}, ensure_ascii=False)}\n\n"
        with _buf_lock:
            try: _subscribers.remove(sq)
            except ValueError: pass

    return Response(
        generate(), mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/models")
def api_models():
    deploy_dir = HERE / "workspace" / "ha_deploy"
    if not deploy_dir.exists():
        return jsonify(models=[])
    models = []
    for d in sorted(deploy_dir.iterdir()):
        if not d.is_dir():
            continue
        files = [
            {"name": f.name, "size": f.stat().st_size}
            for f in sorted(d.iterdir())
            if f.suffix in (".onnx", ".tflite", ".json", ".txt") and f.is_file()
        ]
        if files:
            models.append({"name": d.name, "files": files})
    return jsonify(models=models)


@app.get("/api/download/<model>/<filename>")
def api_download(model: str, filename: str):
    if not re.fullmatch(r"[A-Za-z0-9_\-\.]+", model) or \
       not re.fullmatch(r"[A-Za-z0-9_\-\.]+", filename):
        abort(400)
    p = HERE / "workspace" / "ha_deploy" / model / filename
    if not p.exists():
        abort(404)
    return send_file(str(p), as_attachment=True)


# ── HTML SPA ──────────────────────────────────────────────────────────────────
_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WakeWordForge</title>
<style>
:root {
  --bg:      #0d1117;
  --bg2:     #161b22;
  --bg3:     #21262d;
  --border:  #30363d;
  --text:    #c9d1d9;
  --muted:   #8b949e;
  --blue:    #58a6ff;
  --green:   #3fb950;
  --yellow:  #d29922;
  --red:     #f85149;
  --cyan:    #39c5cf;
  --purple:  #d2a8ff;
  --radius:  8px;
  --font-ui: -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
  --font-mono: "SFMono-Regular",Consolas,"Liberation Mono",Menlo,monospace;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:var(--font-ui);font-size:14px;height:100vh;display:flex;flex-direction:column;overflow:hidden}

/* ── Header ── */
header{
  display:flex;align-items:center;gap:12px;padding:10px 20px;
  background:var(--bg2);border-bottom:1px solid var(--border);flex-shrink:0;
}
.logo{font-size:18px;font-weight:700;color:var(--blue);letter-spacing:-0.5px}
.logo span{color:var(--cyan)}
.status-badge{
  padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600;
  text-transform:uppercase;letter-spacing:.5px;
}
.status-idle  {background:#21262d;color:var(--muted);border:1px solid var(--border)}
.status-running{background:#1f3a1f;color:var(--green);border:1px solid #2ea043;animation:pulse 1.5s infinite}
.status-done  {background:#1a2f1a;color:var(--green);border:1px solid var(--green)}
.status-error {background:#2d1a1a;color:var(--red);border:1px solid var(--red)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.6}}
.header-right{margin-left:auto;display:flex;gap:8px;align-items:center}
.lang-btn{
  background:none;border:1px solid var(--border);color:var(--muted);
  padding:3px 10px;border-radius:5px;cursor:pointer;font-size:12px;transition:.15s
}
.lang-btn:hover,.lang-btn.active{border-color:var(--blue);color:var(--blue)}

/* ── Main layout ── */
main{
  display:grid;grid-template-columns:380px 1fr;flex:1;overflow:hidden;gap:0;
}

/* ── Panels ── */
.panel{overflow-y:auto;padding:20px}
.panel-config{border-right:1px solid var(--border);background:var(--bg)}
.panel-terminal{background:var(--bg);display:flex;flex-direction:column;overflow:hidden;padding:0}

/* ── Form ── */
.section-title{
  font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.8px;
  color:var(--muted);margin-bottom:12px;margin-top:20px;
}
.section-title:first-child{margin-top:0}
.field{margin-bottom:14px}
.field label{
  display:block;font-size:12px;font-weight:600;color:var(--muted);
  margin-bottom:5px;text-transform:uppercase;letter-spacing:.4px
}
.field input[type=text],
.field input[type=number],
.field select{
  width:100%;background:var(--bg3);border:1px solid var(--border);
  color:var(--text);padding:8px 10px;border-radius:var(--radius);
  font-size:13px;outline:none;transition:.15s;font-family:var(--font-ui);
}
.field input[type=text]:focus,
.field input[type=number]:focus,
.field select:focus{border-color:var(--blue);box-shadow:0 0 0 2px #1f3a5f}
.field .hint{font-size:11px;color:var(--muted);margin-top:4px}
.field-row{display:grid;grid-template-columns:1fr 1fr;gap:10px}

/* Radio cards */
.radio-group{display:flex;gap:8px;flex-wrap:wrap}
.radio-card input{display:none}
.radio-card label{
  display:block;padding:7px 14px;border:1px solid var(--border);border-radius:var(--radius);
  cursor:pointer;font-size:12px;font-weight:600;color:var(--muted);transition:.15s;
  text-align:center;
}
.radio-card input:checked+label{
  border-color:var(--blue);color:var(--blue);background:#0d1f3c;
}
.radio-card label:hover{border-color:var(--muted);color:var(--text)}

/* Toggle */
.toggle-row{display:flex;align-items:center;gap:10px;margin-bottom:14px}
.toggle-row label{font-size:13px;color:var(--text);cursor:pointer;user-select:none}
input[type=checkbox].toggle{
  width:34px;height:18px;appearance:none;background:var(--bg3);border:1px solid var(--border);
  border-radius:9px;cursor:pointer;position:relative;transition:.2s;flex-shrink:0;
}
input[type=checkbox].toggle:checked{background:#1f3a5f;border-color:var(--blue)}
input[type=checkbox].toggle::after{
  content:"";position:absolute;width:12px;height:12px;background:var(--muted);
  border-radius:50%;top:2px;left:2px;transition:.2s;
}
input[type=checkbox].toggle:checked::after{left:18px;background:var(--blue)}

/* Advanced section */
.advanced-toggle{
  background:none;border:none;color:var(--blue);cursor:pointer;
  font-size:12px;padding:0;margin-bottom:12px;display:flex;align-items:center;gap:4px;
}
.advanced-toggle .arrow{transition:.2s;display:inline-block}
.advanced-toggle.open .arrow{transform:rotate(90deg)}
.advanced-body{display:none}
.advanced-body.open{display:block}

/* Buttons */
.btn{
  width:100%;padding:10px;border-radius:var(--radius);border:none;cursor:pointer;
  font-size:14px;font-weight:600;transition:.15s;font-family:var(--font-ui);
}
.btn-start{background:#238636;color:#fff;margin-bottom:8px}
.btn-start:hover:not(:disabled){background:#2ea043}
.btn-start:disabled{background:#1a2f1a;color:var(--muted);cursor:not-allowed}
.btn-stop{background:var(--bg3);color:var(--red);border:1px solid var(--border)}
.btn-stop:hover:not(:disabled){border-color:var(--red);background:#2d1a1a}
.btn-stop:disabled{opacity:.4;cursor:not-allowed}
.btn-row{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:4px}
.btn-row .btn{margin:0}

/* ── Terminal panel ── */
.term-header{
  padding:10px 16px;background:var(--bg2);border-bottom:1px solid var(--border);
  flex-shrink:0;display:flex;align-items:center;gap:12px;
}
.term-title{font-size:12px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.6px}

/* Step progress */
.steps{display:flex;gap:6px;align-items:center;margin-left:auto}
.step-dot{
  width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;
  font-size:11px;font-weight:700;background:var(--bg3);border:1px solid var(--border);
  color:var(--muted);flex-shrink:0;transition:.3s;cursor:default;
  position:relative;
}
.step-dot[title]:hover::after{
  content:attr(title);position:absolute;bottom:calc(100% + 6px);left:50%;transform:translateX(-50%);
  background:#161b22;border:1px solid var(--border);color:var(--text);
  padding:3px 8px;border-radius:4px;white-space:nowrap;font-size:11px;font-weight:400;
  pointer-events:none;z-index:10;
}
.step-dot.done{background:#1a2f1a;border-color:#2ea043;color:var(--green)}
.step-dot.active{background:#0d1f3c;border-color:var(--blue);color:var(--blue);animation:pulse 1s infinite}
.step-sep{width:12px;height:1px;background:var(--border);flex-shrink:0}

/* Terminal output */
.term-body{
  flex:1;overflow-y:auto;padding:12px 16px;font-family:var(--font-mono);
  font-size:12.5px;line-height:1.6;background:var(--bg);
}
.term-line{white-space:pre-wrap;word-break:break-all}
.term-line.line-ok    {color:var(--green)}
.term-line.line-warn  {color:var(--yellow)}
.term-line.line-err   {color:var(--red)}
.term-line.line-step  {color:var(--blue)}
.term-line.line-title {color:var(--cyan);font-weight:700}
.term-line.line-dim   {color:var(--muted)}
.ansi-reset {}
.ansi-bold  {font-weight:700}
.ansi-dim   {opacity:.6}
.ansi-red   {color:var(--red)}
.ansi-green {color:var(--green)}
.ansi-yellow{color:var(--yellow)}
.ansi-blue  {color:var(--blue)}
.ansi-cyan  {color:var(--cyan)}
.ansi-white {color:#ffffff}
.ansi-bred  {color:#ff6e6e}
.ansi-bgreen{color:#7ee787}
.ansi-byellow{color:#f2cc60}
.ansi-bblue {color:#79b8ff}
.ansi-bcyan {color:#56d4dd}
.ansi-bwhite{color:#ffffff}

/* Downloads */
.downloads{
  border-top:1px solid var(--border);padding:14px 16px;
  background:var(--bg2);flex-shrink:0;max-height:220px;overflow-y:auto;
}
.downloads-title{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.6px;color:var(--muted);margin-bottom:10px}
.model-group{margin-bottom:10px}
.model-name{font-size:12px;font-weight:600;color:var(--blue);margin-bottom:6px;font-family:var(--font-mono)}
.file-list{display:flex;flex-wrap:wrap;gap:6px}
.dl-btn{
  display:inline-flex;align-items:center;gap:5px;
  background:var(--bg3);border:1px solid var(--border);color:var(--text);
  padding:5px 10px;border-radius:5px;text-decoration:none;font-size:11px;
  font-family:var(--font-mono);transition:.15s;cursor:pointer;
}
.dl-btn:hover{border-color:var(--blue);color:var(--blue)}
.dl-btn .ext-onnx {color:var(--green)}
.dl-btn .ext-tflite{color:var(--yellow)}
.dl-btn .ext-json {color:var(--cyan)}
.dl-btn .ext-txt  {color:var(--muted)}
.no-models{color:var(--muted);font-size:12px}

/* Scrollbar */
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:var(--muted)}

/* Mobile */
@media(max-width:768px){
  main{grid-template-columns:1fr;grid-template-rows:auto 1fr}
  .panel-config{border-right:none;border-bottom:1px solid var(--border);overflow-y:auto;max-height:55vh}
  .panel-terminal{min-height:40vh}
}
</style>
</head>

<body>

<!-- ══ HEADER ════════════════════════════════════════════════ -->
<header>
  <div class="logo">⚡&nbsp;Wake<span>Word</span>Forge</div>
  <div id="status-badge" class="status-badge status-idle">Ready</div>
  <div class="header-right">
    <button class="lang-btn active" id="btn-en" onclick="setLang('en')">EN</button>
    <button class="lang-btn"        id="btn-he" onclick="setLang('he')">עב</button>
  </div>
</header>

<!-- ══ MAIN ══════════════════════════════════════════════════ -->
<main>

<!-- ── LEFT: CONFIG ───────────────────────────────────────── -->
<section class="panel panel-config">

  <p class="section-title" data-en="Wake Word" data-he="מילת ההשכמה">Wake Word</p>

  <div class="field">
    <label data-en="Model name (letters, digits, _)" data-he="שם המודל (אותיות, ספרות, _)">Model name</label>
    <input id="f-model" type="text" placeholder="hey_gadi" pattern="[a-z0-9_]+"
           autocomplete="off" spellcheck="false">
    <div class="hint" data-en="Lowercase letters, digits and _ only" data-he="אותיות קטנות, ספרות וקו תחתון בלבד">Lowercase letters, digits and _ only</div>
  </div>

  <div class="field">
    <label data-en="Hebrew text (for TTS voices)" data-he="טקסט בעברית (לקולות TTS)">Hebrew text</label>
    <input id="f-he" type="text" placeholder='היי גדי' dir="auto" autocomplete="off">
    <div class="hint" data-en="Used with he-IL edge-tts voices. Leave empty to skip Hebrew TTS." data-he="משמש עם קולות edge-tts בעברית. השאר ריק לדילוג.">Used with he-IL edge-tts voices</div>
  </div>

  <div class="field">
    <label data-en="English text (phonetic)" data-he="טקסט באנגלית (פונטי)">English text</label>
    <input id="f-en" type="text" placeholder="hey gadi" autocomplete="off" spellcheck="false">
    <div class="hint" data-en="Use | to separate variants: &quot;hey gadi|hey, gadi&quot;" data-he="הפרד וריאנטים עם |: &quot;hey gadi|hey, gadi&quot;">Use | to separate variants</div>
  </div>

  <p class="section-title" data-en="Output Format" data-he="פורמט פלט">Output Format</p>

  <div class="field">
    <label data-en="Target" data-he="יעד">Target</label>
    <div class="radio-group">
      <div class="radio-card">
        <input type="radio" name="target" id="t-both"  value="both"  checked>
        <label for="t-both"  title="ONNX + TFLite">Both</label>
      </div>
      <div class="radio-card">
        <input type="radio" name="target" id="t-oww"   value="oww">
        <label for="t-oww"   title="openWakeWord ONNX for Home Assistant">ONNX only</label>
      </div>
      <div class="radio-card">
        <input type="radio" name="target" id="t-mww"   value="mww">
        <label for="t-mww"   title="microWakeWord TFLite + JSON for ESP32">TFLite only</label>
      </div>
    </div>
    <div class="hint" data-en="Both = ONNX (Home Assistant) + TFLite/JSON (ESP32)" data-he="שניהם = ONNX (Home Assistant) + TFLite/JSON (ESP32)">Both = ONNX (HA) + TFLite/JSON (ESP32)</div>
  </div>

  <div class="field">
    <label data-en="Architecture" data-he="ארכיטקטורה">Architecture</label>
    <div class="radio-group">
      <div class="radio-card">
        <input type="radio" name="arch" id="a-open"  value="open"  checked>
        <label for="a-open"  title="192-unit DNN — high accuracy for HA server / RPi">OPEN</label>
      </div>
      <div class="radio-card">
        <input type="radio" name="arch" id="a-micro" value="micro">
        <label for="a-micro" title="64-unit DNN — fast for ESP32-S3 / RPi Zero">MICRO</label>
      </div>
    </div>
    <div class="hint" data-en="OPEN: 192-unit (HA/server). MICRO: 64-unit (ESP32/edge)." data-he="OPEN: 192 יחידות (HA). MICRO: 64 יחידות (ESP32).">OPEN: HA/server. MICRO: ESP32/edge.</div>
  </div>

  <p class="section-title" data-en="Options" data-he="אפשרויות">Options</p>

  <div class="toggle-row">
    <input type="checkbox" id="f-record" class="toggle">
    <label for="f-record" data-en="Record my voice (guided mic session before training)" data-he="הקלט את קולי (אשף הקלטות לפני האימון)">Record my voice before training</label>
  </div>

  <div class="field">
    <label data-en="Start from step" data-he="התחל משלב">Start from step</label>
    <select id="f-from-step">
      <option value="">1 — Setup (full pipeline)</option>
      <option value="2">2 — Verify</option>
      <option value="3">3 — Generate samples</option>
      <option value="4">4 — Build features</option>
      <option value="5">5 — Train model</option>
      <option value="6">6 — Test</option>
      <option value="7">7 — Cleanup</option>
    </select>
  </div>

  <button class="advanced-toggle" id="adv-toggle" onclick="toggleAdvanced()">
    <span class="arrow">▶</span>
    <span data-en="Advanced parameters" data-he="פרמטרים מתקדמים">Advanced parameters</span>
  </button>
  <div class="advanced-body" id="adv-body">
    <div class="field-row">
      <div class="field">
        <label data-en="Samples" data-he="דוגמאות">Samples</label>
        <input id="f-samples" type="number" value="100000" min="10000" step="10000">
        <div class="hint" data-en="Augmented training samples" data-he="דוגמאות אימון מוגברות">Augmented samples</div>
      </div>
      <div class="field">
        <label data-en="Steps" data-he="שלבי אימון">Steps</label>
        <input id="f-steps" type="number" value="100000" min="10000" step="10000">
        <div class="hint" data-en="Training iterations" data-he="איטרציות אימון">Training iterations</div>
      </div>
    </div>
    <div class="field">
      <label data-en="False-activation penalty" data-he="קנס הפעלה שגויה">Penalty</label>
      <input id="f-penalty" type="number" value="5000" min="1000" step="1000">
      <div class="hint" data-en="Higher = fewer false triggers (try 20000–50000 if too many false alarms)" data-he="גבוה יותר = פחות טריגרים שגויים">Higher = fewer false triggers</div>
    </div>
  </div>

  <div class="btn-row" style="margin-top:20px">
    <button class="btn btn-start" id="btn-start" onclick="startTraining()">
      <span data-en="▶  Start Training" data-he="▶  התחל אימון">▶  Start Training</span>
    </button>
    <button class="btn btn-stop" id="btn-stop" disabled onclick="stopTraining()">
      <span data-en="■  Stop" data-he="■  עצור">■  Stop</span>
    </button>
  </div>

</section><!-- /panel-config -->

<!-- ── RIGHT: TERMINAL ────────────────────────────────────── -->
<section class="panel-terminal">

  <div class="term-header">
    <span class="term-title" data-en="Live output" data-he="פלט בזמן אמת">Live output</span>
    <div class="steps" id="steps-bar">
      <!-- JS renders the 7 step dots -->
    </div>
  </div>

  <div class="term-body" id="terminal">
    <div class="term-line line-dim" data-en="Waiting for training to start…" data-he="ממתין להתחלת אימון…">Waiting for training to start…</div>
  </div>

  <div class="downloads" id="downloads-panel">
    <div class="downloads-title" data-en="Downloads" data-he="הורדות">Downloads</div>
    <div id="downloads-list"><span class="no-models" data-en="No trained models yet." data-he="אין מודלים מאומנים עדיין.">No trained models yet.</span></div>
  </div>

</section>

</main><!-- /main -->

<script>
// ── Constants ──────────────────────────────────────────────
const STEP_NAMES = ["Setup","Verify","Generate","Features","Train","Test","Cleanup"];

// ── State ──────────────────────────────────────────────────
let currentStep  = 0;
let autoScroll   = true;
let evtSource    = null;
let currentLang  = "en";

// ── Build step progress bar ────────────────────────────────
(function buildSteps() {
  const bar = document.getElementById("steps-bar");
  bar.innerHTML = "";
  STEP_NAMES.forEach((name, i) => {
    if (i > 0) {
      const sep = document.createElement("div");
      sep.className = "step-sep";
      bar.appendChild(sep);
    }
    const dot = document.createElement("div");
    dot.className = "step-dot";
    dot.id = "step-" + (i + 1);
    dot.textContent = i + 1;
    dot.title = name;
    bar.appendChild(dot);
  });
})();

// ── Language toggle ────────────────────────────────────────
function setLang(lang) {
  currentLang = lang;
  document.getElementById("btn-en").classList.toggle("active", lang === "en");
  document.getElementById("btn-he").classList.toggle("active", lang === "he");
  document.querySelectorAll("[data-en][data-he]").forEach(el => {
    el.innerHTML = el.dataset[lang] || el.dataset.en;
  });
}

// ── Advanced section ───────────────────────────────────────
function toggleAdvanced() {
  const btn  = document.getElementById("adv-toggle");
  const body = document.getElementById("adv-body");
  btn.classList.toggle("open");
  body.classList.toggle("open");
}

// ── Step progress update ───────────────────────────────────
function setStep(n) {
  for (let i = 1; i <= 7; i++) {
    const dot = document.getElementById("step-" + i);
    if (!dot) continue;
    dot.classList.remove("done", "active");
    if (i < n)  dot.classList.add("done");
    if (i === n) dot.classList.add("active");
  }
  currentStep = n;
}

function resetSteps() {
  for (let i = 1; i <= 7; i++) {
    const dot = document.getElementById("step-" + i);
    if (dot) dot.classList.remove("done","active");
  }
  currentStep = 0;
}

// ── ANSI → HTML ────────────────────────────────────────────
const ANSI_MAP = {
  "0":  ["ansi-reset"],
  "1":  ["ansi-bold"],
  "2":  ["ansi-dim"],
  "31": ["ansi-red"],    "91": ["ansi-bred"],
  "32": ["ansi-green"],  "92": ["ansi-bgreen"],
  "33": ["ansi-yellow"], "93": ["ansi-byellow"],
  "34": ["ansi-blue"],   "94": ["ansi-bblue"],
  "36": ["ansi-cyan"],   "96": ["ansi-bcyan"],
  "37": ["ansi-white"],  "97": ["ansi-bwhite"],
};

function ansiToHtml(text) {
  // Split on ANSI escape sequences
  const parts = text.split(/(\x1b\[[0-9;]*m)/g);
  let html = "";
  let classes = [];
  for (const part of parts) {
    if (part.startsWith("\x1b[")) {
      const codes = part.slice(2, -1).split(";");
      for (const code of codes) {
        if (code === "0" || code === "") { classes = []; continue; }
        const cls = ANSI_MAP[code];
        if (cls) classes.push(...cls);
      }
    } else if (part) {
      const safe = part.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
      if (classes.length) {
        html += `<span class="${[...new Set(classes)].join(" ")}">${safe}</span>`;
      } else {
        html += safe;
      }
    }
  }
  return html;
}

// ── Classify line type for extra color hints ───────────────
function lineClass(text) {
  const t = text.replace(/\x1b\[[0-9;]*m/g, "");
  if (/\bOK\b/.test(t))                         return "line-ok";
  if (/^\s*!!\s/.test(t))                       return "line-warn";
  if (/^\s*XX\s/.test(t))                       return "line-err";
  if (/^\s*>>\s/.test(t))                       return "line-step";
  if (/={4,}/.test(t) || /-{4,}/.test(t))       return "line-title";
  return "";
}

// ── Detect step from log line ──────────────────────────────
function detectStep(text) {
  const clean = text.replace(/\x1b\[[0-9;]*m/g, "");
  const m = clean.match(/---\s*Step\s+(\d+)\s*:/i) ||
            clean.match(/Step\s+(\d+)\s*\/\s*7/i)  ||
            clean.match(/\[(\d+)\/7\]/);
  if (m) setStep(parseInt(m[1]));
}

// ── Append line to terminal ────────────────────────────────
function appendLine(text) {
  const term = document.getElementById("terminal");
  const div  = document.createElement("div");
  div.className = "term-line " + lineClass(text);
  div.innerHTML = ansiToHtml(text);
  term.appendChild(div);
  if (autoScroll) term.scrollTop = term.scrollHeight;
  detectStep(text);
}

// Pause auto-scroll when user scrolls up
document.getElementById("terminal").addEventListener("scroll", function() {
  const el = this;
  autoScroll = el.scrollTop + el.clientHeight >= el.scrollHeight - 30;
});

// ── Status badge update ────────────────────────────────────
function setStatus(s) {
  const badge = document.getElementById("status-badge");
  badge.className = "status-badge status-" + s;
  const labels = {
    idle:    {en:"Ready",    he:"מוכן"},
    running: {en:"Training…",he:"מאמן…"},
    done:    {en:"Done",     he:"הסתיים"},
    error:   {en:"Error",    he:"שגיאה"},
  };
  badge.textContent = (labels[s] || labels.idle)[currentLang];
}

// ── Button state ───────────────────────────────────────────
function setRunning(running) {
  document.getElementById("btn-start").disabled = running;
  document.getElementById("btn-stop").disabled  = !running;
}

// ── SSE connection ─────────────────────────────────────────
function connectLogs() {
  if (evtSource) { evtSource.close(); evtSource = null; }
  evtSource = new EventSource("/api/logs");
  evtSource.onmessage = function(e) {
    if (!e.data || e.data === "{}") return;
    try {
      const msg = JSON.parse(e.data);
      if (msg.type === "log") {
        appendLine(msg.text);
      } else if (msg.type === "done") {
        setStatus(msg.status || "done");
        setRunning(false);
        if (msg.status === "done") markAllStepsDone();
        refreshDownloads();
        evtSource.close(); evtSource = null;
      }
    } catch {}
  };
  evtSource.onerror = function() {
    if (evtSource) { evtSource.close(); evtSource = null; }
  };
}

function markAllStepsDone() {
  for (let i = 1; i <= 7; i++) {
    const dot = document.getElementById("step-" + i);
    if (dot) { dot.classList.remove("active"); dot.classList.add("done"); }
  }
}

// ── Start training ─────────────────────────────────────────
async function startTraining() {
  const model = document.getElementById("f-model").value.trim();
  if (!model) {
    alert(currentLang === "he" ? "יש להזין שם מודל" : "Model name is required");
    document.getElementById("f-model").focus();
    return;
  }
  const payload = {
    model:     model,
    he:        document.getElementById("f-he").value.trim(),
    en:        document.getElementById("f-en").value.trim(),
    target:    document.querySelector("input[name=target]:checked").value,
    arch:      document.querySelector("input[name=arch]:checked").value,
    samples:   parseInt(document.getElementById("f-samples").value) || 100000,
    steps:     parseInt(document.getElementById("f-steps").value)   || 100000,
    penalty:   parseInt(document.getElementById("f-penalty").value) || 5000,
    record:    document.getElementById("f-record").checked,
    from_step: parseInt(document.getElementById("f-from-step").value) || null,
  };
  if (!payload.from_step) delete payload.from_step;

  // Clear terminal
  document.getElementById("terminal").innerHTML = "";
  autoScroll = true;
  resetSteps();

  const r = await fetch("/api/start", {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify(payload),
  });
  const data = await r.json();
  if (!r.ok) {
    alert(data.error || "Failed to start");
    return;
  }
  setStatus("running");
  setRunning(true);
  appendLine("$ " + data.cmd);
  appendLine("");
  connectLogs();
}

// ── Stop training ──────────────────────────────────────────
async function stopTraining() {
  await fetch("/api/stop", {method:"POST"});
  setStatus("idle");
  setRunning(false);
}

// ── Downloads ──────────────────────────────────────────────
async function refreshDownloads() {
  const r = await fetch("/api/models");
  const { models } = await r.json();
  const list = document.getElementById("downloads-list");
  if (!models || models.length === 0) {
    list.innerHTML = `<span class="no-models" data-en="No trained models yet." data-he="אין מודלים מאומנים עדיין.">${currentLang==="he"?"אין מודלים מאומנים עדיין.":"No trained models yet."}</span>`;
    return;
  }
  list.innerHTML = models.map(m => `
    <div class="model-group">
      <div class="model-name">${escHtml(m.name)}</div>
      <div class="file-list">
        ${m.files.map(f => {
          const ext = f.name.split(".").pop();
          const sz  = (f.size / 1024).toFixed(0) + " KB";
          return `<a class="dl-btn" href="/api/download/${encodeURIComponent(m.name)}/${encodeURIComponent(f.name)}" download>
            <span class="ext-${ext}">${escHtml(f.name)}</span>
            <span style="color:var(--muted)">${sz}</span>
          </a>`;
        }).join("")}
      </div>
    </div>
  `).join("");
}

function escHtml(s) {
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

// ── Init ───────────────────────────────────────────────────
(async function init() {
  // Check current status
  const r = await fetch("/api/status");
  const { status } = await r.json();
  setStatus(status);
  if (status === "running") {
    setRunning(true);
    connectLogs();
  }
  refreshDownloads();
})();
</script>
</body>
</html>"""

@app.get("/")
def index():
    return _HTML, 200, {"Content-Type": "text/html; charset=utf-8"}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import webbrowser, socket
    # Try to bind to PORT, fall back gracefully
    try:
        s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", PORT)); s.close()
    except OSError:
        print(f"[web_ui] Port {PORT} is in use. Change PORT at top of web_ui.py.")
        sys.exit(1)

    url = f"http://localhost:{PORT}"
    print(f"\n  WakeWordForge Web UI")
    print(f"  {url}")
    print(f"  Press Ctrl+C to stop.\n")
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
