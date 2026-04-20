"""
Call Analyzer Web App
=====================
Flask web server to replace the scheduled automation.

Flow:
 1. User opens the web UI.
 2. Chooses number of calls + pick mode (latest / random).
 3. Clicks "Analyze".
 4. Backend fetches calls from MongoDB, runs analysis.py + clip_stt_issues.py.
 5. Frontend polls /api/status for progress.
 6. When done, frontend renders the raw analysis_results.csv as a table
    and offers CSV / JSON downloads.

Run:
    pip install -r requirements.txt
    python app.py
Then open: http://127.0.0.1:5000/
"""

import csv
import json
import os
import random
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
from datetime import datetime

from bson import json_util
from flask import Flask, jsonify, request, send_file, render_template, abort
from pymongo import MongoClient, DESCENDING

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_FILE = "selenium_voice_calling.chat_history.json"
ANALYSIS_JSON = "analysis_results.json"
ANALYSIS_CSV = "analysis_results.csv"
STT_CSV = "stt_clips_report.csv"

MONGO_USER = urllib.parse.quote_plus("avishkar_app")
MONGO_PASS = urllib.parse.quote_plus("Avishkar2026!App")
MONGO_URI = (
    f"mongodb://{MONGO_USER}:{MONGO_PASS}"
    "@35.226.60.171:27017/avishkar_db?authSource=avishkar_db"
)

# ---------------------------------------------------------------------------
# Global job state (single job at a time)
# ---------------------------------------------------------------------------
JOB_LOCK = threading.Lock()
JOB_STATE = {
    "status": "idle",          # idle | fetching | analyzing | clipping | done | error
    "message": "",
    "logs": [],
    "started_at": None,
    "finished_at": None,
    "params": {},
    "fetched_count": 0,
    "analyzed_count": 0,
    "target_count": 0,
    "error": None,
}


def _log(msg: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    JOB_STATE["logs"].append(line)
    # Keep log buffer bounded
    if len(JOB_STATE["logs"]) > 500:
        JOB_STATE["logs"] = JOB_STATE["logs"][-500:]


def _reset_state(params: dict) -> None:
    JOB_STATE.update({
        "status": "starting",
        "message": "Starting...",
        "logs": [],
        "started_at": datetime.now().isoformat(),
        "finished_at": None,
        "params": params,
        "fetched_count": 0,
        "analyzed_count": 0,
        "target_count": params.get("count", 0),
        "error": None,
    })


# ---------------------------------------------------------------------------
# MongoDB fetch
# ---------------------------------------------------------------------------
def fetch_calls_sync(count: int, mode: str) -> int:
    """Fetch calls from Mongo and write INPUT_FILE. Returns number saved."""
    JOB_STATE["status"] = "fetching"
    JOB_STATE["message"] = f"Fetching {count} calls ({mode}) from MongoDB..."
    _log(JOB_STATE["message"])

    client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=20000,
        connectTimeoutMS=20000,
        socketTimeoutMS=20000,
    )
    db = client.avishkar_db
    coll = db.chat_history

    today = datetime.now()
    start_dt = datetime(today.year, today.month, today.day, 0, 0, 0)
    end_dt = datetime(today.year, today.month, today.day, 23, 59, 59)
    query = {"start_time": {"$gte": start_dt, "$lte": end_dt}}

    if mode == "latest":
        cursor = coll.find(query).sort("start_time", DESCENDING).limit(count)
        data = list(cursor)
    else:
        # random: pull all, sample client-side (same as old logic)
        data = list(coll.find(query))
        if len(data) > count:
            data = random.sample(data, count)

    _log(f"Pulled {len(data)} records from MongoDB.")

    if not data:
        client.close()
        return 0

    with open(INPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, default=json_util.default, indent=2)

    client.close()
    JOB_STATE["fetched_count"] = len(data)
    JOB_STATE["target_count"] = len(data)
    _log(f"Saved {len(data)} calls to {INPUT_FILE}.")
    return len(data)


# ---------------------------------------------------------------------------
# Analysis runners
# ---------------------------------------------------------------------------
def _count_csv_rows(path: str) -> int:
    if not os.path.exists(path):
        return 0
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            return max(0, sum(1 for _ in reader) - 1)  # minus header
    except Exception:
        return 0


def _remove_if_exists(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def run_analysis_subprocess(target_count: int) -> None:
    JOB_STATE["status"] = "analyzing"
    JOB_STATE["message"] = "Running analysis.py (Gemini) ..."
    _log(JOB_STATE["message"])

    # Clean prior outputs so analysis.py does a fresh run
    _remove_if_exists(ANALYSIS_JSON)
    _remove_if_exists(ANALYSIS_CSV)

    proc = subprocess.Popen(
        [sys.executable, "-u", "analysis.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Poll output + CSV row count while running
    while True:
        line = proc.stdout.readline() if proc.stdout else ""
        if line:
            _log(line.rstrip())
        JOB_STATE["analyzed_count"] = _count_csv_rows(ANALYSIS_CSV)
        if proc.poll() is not None and not line:
            break
    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"analysis.py exited with code {proc.returncode}")

    JOB_STATE["analyzed_count"] = _count_csv_rows(ANALYSIS_CSV)
    _log(f"analysis.py finished. {JOB_STATE['analyzed_count']} rows in CSV.")


def run_clip_subprocess() -> None:
    JOB_STATE["status"] = "clipping"
    JOB_STATE["message"] = "Running clip_stt_issues.py ..."
    _log(JOB_STATE["message"])

    proc = subprocess.Popen(
        [sys.executable, "-u", "clip_stt_issues.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    while True:
        line = proc.stdout.readline() if proc.stdout else ""
        if line:
            _log(line.rstrip())
        if proc.poll() is not None and not line:
            break
    proc.wait()

    if proc.returncode != 0:
        # Clipping is best-effort, don't hard-fail the whole job
        _log(f"WARNING: clip_stt_issues.py exited with code {proc.returncode}")
    else:
        _log("clip_stt_issues.py finished.")


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------
def _job_worker(count: int, mode: str, run_clipping: bool) -> None:
    try:
        fetched = fetch_calls_sync(count, mode)
        if fetched == 0:
            JOB_STATE["status"] = "done"
            JOB_STATE["message"] = "No calls found for today."
            JOB_STATE["finished_at"] = datetime.now().isoformat()
            _log(JOB_STATE["message"])
            return

        run_analysis_subprocess(fetched)

        if run_clipping:
            run_clip_subprocess()

        JOB_STATE["status"] = "done"
        JOB_STATE["message"] = "Analysis complete."
        JOB_STATE["finished_at"] = datetime.now().isoformat()
        _log("Job complete.")
    except Exception as e:
        JOB_STATE["status"] = "error"
        JOB_STATE["error"] = str(e)
        JOB_STATE["message"] = f"Error: {e}"
        JOB_STATE["finished_at"] = datetime.now().isoformat()
        _log(f"ERROR: {e}")
        _log(traceback.format_exc())


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(silent=True) or {}
    try:
        count = int(data.get("count", 20))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "count must be an integer"}), 400
    mode = str(data.get("mode", "random")).lower()
    if mode not in ("latest", "random"):
        return jsonify({"ok": False, "error": "mode must be 'latest' or 'random'"}), 400
    if count <= 0 or count > 500:
        return jsonify({"ok": False, "error": "count must be 1..500"}), 400

    run_clipping = bool(data.get("run_clipping", True))

    with JOB_LOCK:
        if JOB_STATE["status"] in ("fetching", "analyzing", "clipping", "starting"):
            return jsonify({
                "ok": False,
                "error": "A job is already running.",
                "status": JOB_STATE["status"],
            }), 409

        _reset_state({"count": count, "mode": mode, "run_clipping": run_clipping})

    t = threading.Thread(
        target=_job_worker,
        args=(count, mode, run_clipping),
        daemon=True,
    )
    t.start()
    return jsonify({"ok": True, "message": "Job started."})


@app.route("/api/status")
def api_status():
    # Live-update analyzed_count while a run is in progress
    if JOB_STATE["status"] == "analyzing":
        JOB_STATE["analyzed_count"] = _count_csv_rows(ANALYSIS_CSV)
    return jsonify(JOB_STATE)


def _read_csv_as_json(path: str) -> dict:
    if not os.path.exists(path):
        return {"columns": [], "rows": [], "exists": False}
    rows = []
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            for r in reader:
                rows.append(r)
        return {"columns": columns, "rows": rows, "exists": True}
    except Exception as e:
        return {"columns": [], "rows": [], "exists": True, "error": str(e)}


@app.route("/api/results")
def api_results():
    """Return parsed CSV as JSON. ?source=analysis (default) | stt"""
    source = request.args.get("source", "analysis").lower()
    path = STT_CSV if source == "stt" else ANALYSIS_CSV
    return jsonify(_read_csv_as_json(path))


ALLOWED_DOWNLOADS = {
    "analysis_csv": ANALYSIS_CSV,
    "analysis_json": ANALYSIS_JSON,
    "stt_csv": STT_CSV,
    "input_json": INPUT_FILE,
}


@app.route("/api/download/<key>")
def api_download(key):
    path = ALLOWED_DOWNLOADS.get(key)
    if not path:
        abort(404)
    if not os.path.exists(path):
        abort(404, description=f"{path} not found. Run analysis first.")
    return send_file(path, as_attachment=True, download_name=os.path.basename(path))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print(f"Call Analyzer web app running on http://127.0.0.1:{port}/")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
