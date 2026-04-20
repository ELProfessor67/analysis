"""
STT Issue Audio Clipper
========================
Reads analysis_results.json, finds all STT issues (customer_said_in_audio),
uses Gemini to locate the exact timestamps in the audio, then clips those
segments with ffmpeg. Saves clips + a CSV report.

Usage:
    set GEMINI_API_KEY=your_api_key_here
    python clip_stt_issues.py

Output: stt_clips/ folder + stt_clips_report.csv
"""

import csv
import json
import os
import subprocess
import sys
import time
import threading
import traceback
import concurrent.futures

import httpx
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------------------------
# Pydantic model for Gemini structured output
# ---------------------------------------------------------------------------
class TimestampResult(BaseModel):
    start_seconds: float
    end_seconds: float
    found: bool
    confidence: str

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ANALYSIS_FILE = "analysis_results.json"
CLIPS_DIR = "stt_clips"
AUDIO_CACHE_DIR = "stt_clips_audio_cache"
OUTPUT_CSV = "stt_clips_report.csv"
MODEL = "gemini-3-flash-preview"
PARALLEL_WORKERS = 5

# Prompt sent to Gemini to locate the timestamp
TIMESTAMP_PROMPT = """Role:
You are a forensic audio analyst and linguistic expert specializing in Hindi and Indian English.
Your task is to locate a specific target phrase in the provided audio file and return its exact timestamps.

Context:
- This is a Hindi/Hinglish customer support phone call between a customer and an AI bot.
- The audio may contain noise, overlap, and dialect nuances.
- Focus ONLY on the CUSTOMER's speech, not the bot/agent.

ASR Error Details:
- customer_said_in_audio (ground truth — listen for this): "{phrase}"
- transcript_showed (what the AI transcriber incorrectly heard — use only as acoustic context): "{transcript}"
- turn_number (approximate conversational turn): {turn_number}

Task:
1. Listen to the audio carefully.
2. Locate where the CUSTOMER says the phrase: "{phrase}" (or something very close to it).
3. Identify the EXACT start and end timestamps (in seconds) of that phrase.
4. Add a 0.5 second buffer before start and after end for cleaner clips.

Return ONLY a JSON object with these fields:
- "start_seconds": float — when the customer STARTS saying the phrase (e.g. 12.5)
- "end_seconds": float — when the customer FINISHES saying the phrase (e.g. 15.2)
- "found": boolean — true if you found the phrase, false if not
- "confidence": string — "high", "medium", or "low"

IMPORTANT: Return ONLY the raw JSON object. No markdown, no code fences, no explanation.
"""

# ---------------------------------------------------------------------------
# Thread-safe print
# ---------------------------------------------------------------------------
_print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_analysis(filepath: str) -> list[dict]:
    """Load analysis results from JSON."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def download_audio(url: str, cache_dir: str) -> str:
    """Download audio file, using cache to avoid re-downloading."""
    filename = url.split("/")[-1]
    filepath = os.path.join(cache_dir, filename)

    if os.path.exists(filepath):
        safe_print(f"    [CACHE] Using cached audio: {filename}")
        return filepath

    safe_print(f"    [DOWNLOAD] Downloading: {filename}")
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(resp.content)

    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    safe_print(f"    [DOWNLOAD] Done: {size_mb:.2f} MB")
    return filepath


# parse_gemini_json removed — using Pydantic response_schema instead


MAX_RETRIES = 3


def get_timestamps_from_gemini(
    client: genai.Client,
    phrase: str,
    transcript: str,
    turn_number: str,
    audio_url: str,
) -> dict:
    """Use Gemini to find the timestamp of a phrase in the audio."""
    prompt = TIMESTAMP_PROMPT.format(phrase=phrase, transcript=transcript, turn_number=turn_number)



    raw = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_uri(
                                file_uri=audio_url,
                                mime_type="audio/wav",
                            ),
                            types.Part.from_text(text=prompt),
                        ],
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=1024,
                    response_mime_type="application/json",
                    response_schema=TimestampResult,
                ),
            )

            safe_print(f"    [GEMINI] Raw response: {response.text[:200] if response.text else 'None'}")

            # Handle None / empty response (safety blocks, empty candidates)
            if response.text is None:
                block_reason = ""
                if hasattr(response, "prompt_feedback") and response.prompt_feedback:
                    block_reason = str(response.prompt_feedback)
                raise ValueError(
                    f"Gemini returned empty response (possibly blocked). {block_reason}"
                )

            raw = response.text.strip()
            if not raw:
                raise ValueError("Gemini returned an empty string.")

            # Parse with Pydantic — guaranteed to match schema
            result = TimestampResult.model_validate_json(raw)
            return result.model_dump()

        except Exception as e:
            backoff = 2 ** attempt  # 2s, 4s, 8s
            if attempt < MAX_RETRIES:
                safe_print(
                    f"    [GEMINI] Attempt {attempt}/{MAX_RETRIES} failed: "
                    f"{type(e).__name__}: {e}. Retrying in {backoff}s..."
                )
                time.sleep(backoff)
            else:
                safe_print(
                    f"    [GEMINI] Failed after {MAX_RETRIES} attempts. "
                    f"Last error: {type(e).__name__}: {e}"
                )
                if raw:
                    safe_print(f"    [GEMINI] Last raw response: {raw[:300]}")
                raise


def clip_audio_ffmpeg(
    input_path: str,
    output_path: str,
    start_seconds: float,
    end_seconds: float,
):
    """Use ffmpeg to clip a segment from the audio file."""
    cmd = [
        "ffmpeg",
        "-y",                     # overwrite output
        "-i", input_path,         # input file
        "-ss", str(start_seconds),  # start time
        "-to", str(end_seconds),    # end time
        "-c", "copy",             # copy codec (fast, no re-encode)
        output_path,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode != 0:
        safe_print(f"    [FFMPEG] WARNING: ffmpeg returned non-zero: {result.stderr[:200]}")
        # Retry with re-encoding if copy fails
        cmd_reencode = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-ss", str(start_seconds),
            "-to", str(end_seconds),
            output_path,
        ]
        subprocess.run(cmd_reencode, capture_output=True, text=True, timeout=60)


# ---------------------------------------------------------------------------
# Thread-safe incremental CSV writer
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "call_index",
    "recording_url",
    "clip_filename",
    "customer_said_in_audio",
    "transcript_showed",
    "turn_number",
    "type",
    "impact",
    "clip_start",
    "clip_end",
    "confidence",
    "error",
]


class CsvWriter:
    """Thread-safe CSV writer that appends rows immediately to disk."""

    def __init__(self, filepath: str, columns: list[str]):
        self.filepath = filepath
        self.columns = columns
        self._lock = threading.Lock()
        self._file = None
        self._writer = None

        # If file exists, keep it (append mode); otherwise create with header
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            self._file = open(filepath, "a", encoding="utf-8", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=columns)
        else:
            self._file = open(filepath, "w", encoding="utf-8", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=columns)
            self._writer.writeheader()
            self._file.flush()

    def write_row(self, row: dict):
        with self._lock:
            self._writer.writerow(row)
            self._file.flush()

    def close(self):
        if self._file:
            self._file.close()


def load_done_set(filepath: str) -> set:
    """Load existing CSV and return set of (call_index, issue_index) already done."""
    done = set()
    if not os.path.exists(filepath):
        return done
    try:
        with open(filepath, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ci = row.get("call_index", "")
                clip = row.get("clip_filename", "")
                # Extract issue index from clip_filename like "call_5_issue_2.wav"
                if clip:
                    done.add((str(ci), clip))
                else:
                    # For errors/skips, use customer_said_in_audio as key
                    phrase = row.get("customer_said_in_audio", "")
                    done.add((str(ci), phrase))
        safe_print(f"[RESUME] Found {len(done)} already-processed entries in CSV.")
    except Exception as e:
        safe_print(f"[RESUME] Could not read existing CSV: {e}")
    return done


# ---------------------------------------------------------------------------
# Per-call worker (runs in a thread)
# ---------------------------------------------------------------------------

def process_call(
    entry: dict,
    gemini_client: genai.Client,
    csv_writer: CsvWriter,
    done_set: set,
) -> tuple[int, int]:
    """
    Process a single call entry: download audio, find timestamps via Gemini,
    clip with ffmpeg. Writes CSV rows immediately. Returns (clip_count, error_count).
    """
    call_index = entry.get("call_index", "?")
    recording_url = entry.get("recording_url", "")
    issues = entry["analysis"]["5_stt_issues"].get("issues", [])

    safe_print(f"\n{'='*60}")
    safe_print(f"[Worker] Call #{call_index} — {len(issues)} STT issue(s)")
    safe_print(f"URL: {recording_url}")
    safe_print(f"{'='*60}")

    clip_count = 0
    error_count = 0

    if not recording_url:
        safe_print(f"  [Call #{call_index}] SKIP: No recording URL.")
        return clip_count, error_count

    # Download audio for ffmpeg clipping
    try:
        audio_path = download_audio(recording_url, AUDIO_CACHE_DIR)
    except Exception as e:
        safe_print(f"  [Call #{call_index}] ERROR downloading audio: {e}")
        for i, issue in enumerate(issues):
            csv_writer.write_row({
                "call_index": call_index,
                "recording_url": recording_url,
                "clip_filename": "",
                "customer_said_in_audio": issue.get("customer_said_in_audio", ""),
                "transcript_showed": issue.get("transcript_showed", ""),
                "turn_number": issue.get("turn_number", ""),
                "type": issue.get("type", ""),
                "impact": issue.get("impact", ""),
                "clip_start": "",
                "clip_end": "",
                "confidence": "",
                "error": f"Download failed: {e}",
            })
            error_count += 1
        return clip_count, error_count

    # Process each STT issue for this call
    for i, issue in enumerate(issues):
        phrase = issue.get("customer_said_in_audio", "")
        transcript = issue.get("transcript_showed", "")
        turn_number = issue.get("turn_number", "")
        issue_type = issue.get("type", "")
        impact = issue.get("impact", "")

        clip_filename = f"call_{call_index}_issue_{i+1}.wav"

        # Check if already done (resume support)
        if (str(call_index), clip_filename) in done_set or (str(call_index), phrase) in done_set:
            safe_print(f"  [Call #{call_index}] Issue {i+1}/{len(issues)}: ALREADY DONE, skipping.")
            continue

        safe_print(f"\n  [Call #{call_index}] Issue {i+1}/{len(issues)}: \"{phrase}\"")
        safe_print(f"    Transcript showed: \"{transcript}\"")
        safe_print(f"    Turn: {turn_number}, Type: {issue_type}, Impact: {impact}")

        # Ask Gemini for timestamps
        try:
            ts = get_timestamps_from_gemini(
                gemini_client,
                phrase,
                transcript,
                turn_number,
                recording_url,
            )
            safe_print(f"    [Call #{call_index}] [GEMINI] Result: {json.dumps(ts, ensure_ascii=False)}")
        except Exception as e:
            safe_print(f"    [Call #{call_index}] [GEMINI] ERROR: {e}")
            traceback.print_exc()
            csv_writer.write_row({
                "call_index": call_index,
                "recording_url": recording_url,
                "clip_filename": "",
                "customer_said_in_audio": phrase,
                "transcript_showed": transcript,
                "turn_number": turn_number,
                "type": issue_type,
                "impact": impact,
                "clip_start": "",
                "clip_end": "",
                "confidence": "",
                "error": f"Gemini error: {e}",
            })
            error_count += 1
            time.sleep(2)
            continue

        # Check if Gemini found it
        if not ts.get("found", False):
            safe_print(f"    [Call #{call_index}] [SKIP] Gemini could not locate phrase in audio.")
            csv_writer.write_row({
                "call_index": call_index,
                "recording_url": recording_url,
                "clip_filename": "",
                "customer_said_in_audio": phrase,
                "transcript_showed": transcript,
                "turn_number": turn_number,
                "type": issue_type,
                "impact": impact,
                "clip_start": ts.get("start_seconds", ""),
                "clip_end": ts.get("end_seconds", ""),
                "confidence": ts.get("confidence", ""),
                "error": "Phrase not found in audio",
            })
            error_count += 1
            continue

        start_sec = float(ts["start_seconds"])
        end_sec = float(ts["end_seconds"])
        confidence = ts.get("confidence", "unknown")

        # Clip with ffmpeg
        clip_path = os.path.join(CLIPS_DIR, clip_filename)

        safe_print(f"    [Call #{call_index}] [FFMPEG] Clipping {start_sec:.1f}s — {end_sec:.1f}s → {clip_filename}")
        try:
            clip_audio_ffmpeg(audio_path, clip_path, start_sec, end_sec)
            clip_count += 1
            safe_print(f"    [Call #{call_index}] [FFMPEG] ✓ Saved: {clip_path}")
        except Exception as e:
            safe_print(f"    [Call #{call_index}] [FFMPEG] ERROR: {e}")
            clip_filename = ""
            error_count += 1

        csv_writer.write_row({
            "call_index": call_index,
            "recording_url": recording_url,
            "clip_filename": clip_filename,
            "customer_said_in_audio": phrase,
            "transcript_showed": transcript,
            "turn_number": turn_number,
            "type": issue_type,
            "impact": impact,
            "clip_start": start_sec,
            "clip_end": end_sec,
            "confidence": confidence,
            "error": "",
        })

    return clip_count, error_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Setup directories
    os.makedirs(CLIPS_DIR, exist_ok=True)
    os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

    # Load analysis data
    print(f"Loading analysis from {ANALYSIS_FILE}...")
    data = load_analysis(ANALYSIS_FILE)
    print(f"Loaded {len(data)} call entries.")

    # Filter entries with STT issues
    stt_entries = [
        d for d in data
        if d.get("analysis", {}).get("5_stt_issues", {}).get("answer") == "yes"
    ]
    total_issues = sum(
        len(d["analysis"]["5_stt_issues"].get("issues", []))
        for d in stt_entries
    )
    print(f"Found {len(stt_entries)} calls with STT issues ({total_issues} total issues).")

    # Load already-done entries for resume
    done_set = load_done_set(OUTPUT_CSV)
    if done_set:
        print(f"Resuming — {len(done_set)} entries already in CSV will be skipped.")

    print(f"Processing with {PARALLEL_WORKERS} parallel workers...\n")

    # Init Gemini client
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY environment variable.")
        sys.exit(1)

    gemini_client = genai.Client(api_key=api_key)

    # Open CSV writer (appends to existing file)
    csv_writer = CsvWriter(OUTPUT_CSV, CSV_COLUMNS)

    # Process calls in parallel
    total_clips = 0
    total_errors = 0

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
            future_to_call = {
                executor.submit(process_call, entry, gemini_client, csv_writer, done_set): entry.get("call_index", "?")
                for entry in stt_entries
            }

            for future in concurrent.futures.as_completed(future_to_call):
                call_index = future_to_call[future]
                try:
                    clip_count, error_count = future.result()
                    total_clips += clip_count
                    total_errors += error_count
                    safe_print(f"\n  ✓ Call #{call_index} done — {clip_count} clips, {error_count} errors")
                except Exception as e:
                    safe_print(f"\n  ✗ Call #{call_index} FAILED: {e}")
                    traceback.print_exc()
                    total_errors += 1
    finally:
        csv_writer.close()

    print(f"\n{'='*60}")
    print(f"DONE!")
    print(f"  Clips saved:  {total_clips}")
    print(f"  Errors:       {total_errors}")
    print(f"  CSV report:   {OUTPUT_CSV}")
    print(f"  Clips folder: {CLIPS_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

