#!/usr/bin/env python3
"""
TempleOS Archive Video Transcriber  (cross-platform edition)
=============================================================
Downloads every video from the Terry A. Davis / TempleOS Archive on archive.org
and transcribes each one using Whisper.

Backend selection (automatic, in priority order)
-------------------------------------------------
  1. mlx-whisper          - Apple Silicon Mac (M1/M2/M3/M4)
                            pip install mlx-whisper
  2. faster-whisper       - Any platform with NVIDIA CUDA GPU, or CPU fallback
                            pip install faster-whisper
                            (on Linux/Windows this is the primary path)
  3. openai-whisper       - Pure-CPU fallback if neither of the above is found
                            pip install openai-whisper

The active backend is printed at startup and recorded in every transcript.

Audio strategy (no full video download)
----------------------------------------
Instead of downloading entire video files, this script streams audio directly
from archive.org using ffmpeg's HTTP input capability.  ffmpeg reads only the
bytes it needs from the remote stream to decode audio - no video data is ever
written to disk.

Fallback tiers, in order
-------------------------
  1. Archive.org pre-generated audio derivative (.ogg / .mp3 / .wav), when present
  2. ffmpeg streaming extraction from the video URL
  3. Recovery (only after a permanent decode failure, see below)

Recovery of corrupt sources
---------------------------
Some archive uploads are truncated mid-recording (the MP4 moov index, written
last, never made it).  Normal streaming can't decode them, so before giving up
the script tries, in order:
  1. Full download + local decode   - fixes failures that were only about HTTP seeking
  2. untrunc                          - rebuilds the missing moov from a healthy,
                                        brand-matched archive file used as reference
                                        (recovers ~all of the surviving audio)
  3. yt-dlp                           - re-fetches the original from YouTube when the
                                        filename still carries a video id
Recovered transcripts are flagged ("recovered_via" / "partial_recovery") since
only the bytes that survived truncation can be transcribed.  Whatever still can't
be salvaged is written to output/logs/unrecoverable.csv with archive + YouTube
links for hand investigation.

Features
--------
  • Fully resumable - state machine persisted to disk (debounced writes).
  • Hash verification at every stage (SHA-256 for audio/transcript).
  • Disk-safe: WAV deleted right after transcription; no video ever touches disk.
    Extraction runs ahead of transcription, so a bounded handful of WAVs
    (≈ workers + queue depth) can exist at once - never the whole archive.
  • Parallel audio extractions feed a single GPU transcription worker that uses
    faster-whisper batched inference for a large speedup.
  • Distinguishes transient failures (network resets, 5xx, timeouts - retried
    with backoff) from permanent ones.  Permanent failures fail fast, attempt
    recovery, and otherwise settle as "corrupt" (skipped on later runs; force a
    retry with --retry-corrupt) or "no_audio" (the source has no audio track at
    all, so it is never counted as a missing transcript).

Requirements
------------
    pip install -r requirements.txt      # internetarchive, faster-whisper, rich, tqdm
    # faster-whisper (the default) covers NVIDIA CUDA *and* CPU on every platform.
    # Apple Silicon users may instead:  pip install mlx-whisper
    # Pure-CPU last resort:             pip install openai-whisper

    brew install ffmpeg          # macOS
    sudo pacman -S ffmpeg        # Arch / CachyOS
    sudo apt install ffmpeg      # Debian/Ubuntu
    # Windows: https://ffmpeg.org/download.html  (add to PATH)

    ia configure                 # archive.org S3 keys (optional, higher rate limits)

Tuning (CLI flags, or the matching WHISPER_* env vars)
------------------------------------------------------
    --model        / WHISPER_MODEL        default large-v3 (try large-v3-turbo, faster)
    --beam-size    / WHISPER_BEAM_SIZE    default 5  (1 = greedy: faster, slightly worse)
    --batch-size   / WHISPER_BATCH_SIZE   default 8  (faster-whisper batches; 1 = off)
    --workers                             parallel audio extractions (default 5)
    --verify-hashes                       deep SHA-256 recheck of done transcripts
    --retry-corrupt                       re-attempt sources previously marked
                                          unrecoverable (truncated / no audio)

Output layout
-------------
    output/
        audio/            <- temporary WAVs; pruned after each transcription
        transcripts/
            <flat-stem>.json
            <flat-stem>.txt
        logs/
            progress.json
            errors.log
"""
import argparse
import atexit
import csv
import hashlib
import json
import logging
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# ── Configuration ──────────────────────────────────────────────────────────────
ARCHIVE_IDENTIFIER = "TerryADavis_TempleOS_Archive"
VIDEO_EXTENSIONS   = {
    ".mp4", ".mkv", ".avi", ".mov", ".webm",
    ".flv", ".wmv", ".ogv", ".m4v", ".mpg", ".mpeg",
}
AUDIO_DERIVATIVE_EXTENSIONS = [".ogg", ".mp3", ".wav"]

BASE_DIR       = Path("output")
AUDIO_DIR      = BASE_DIR / "audio"
RECOVERY_DIR   = BASE_DIR / "recovery"      # temp full-file downloads for recovery (pruned)
TRANSCRIPT_DIR = BASE_DIR / "transcripts"
LOG_DIR        = BASE_DIR / "logs"
PROGRESS_FILE  = LOG_DIR / "progress.json"
ERROR_LOG_FILE = LOG_DIR / "errors.log"
UNRECOVERABLE_REPORT = LOG_DIR / "unrecoverable.csv"
IGNORE_FILE    = Path(".ignore")

MAX_CONCURRENT_EXTRACTIONS = 5
STREAM_RETRIES             = 6
STREAM_TIMEOUT             = 600     # per-attempt floor (seconds)
STREAM_TIMEOUT_PER_MB      = 3       # + this many seconds per MB of source …
STREAM_TIMEOUT_MAX         = 5400    # … capped here (90 min) for huge streams
STREAM_THROTTLE_SECS       = 1.0
STREAM_BACKOFF_BASE        = 5
STREAM_BACKOFF_MAX         = 120

# Transcription tuning - override via env vars or the CLI flags parsed in main().
WHISPER_MODEL_OVERRIDE = os.environ.get("WHISPER_MODEL")             # e.g. large-v3-turbo
BEAM_SIZE              = int(os.environ.get("WHISPER_BEAM_SIZE",  "5"))   # 1 = greedy/fast
BATCH_SIZE             = int(os.environ.get("WHISPER_BATCH_SIZE", "8"))   # 1 disables batching

# Rewrite the on-disk progress file at most once per this many seconds.
PROGRESS_FLUSH_INTERVAL = 2.0

S_PENDING     = "pending"
S_EXTRACTED   = "extracted"
S_TRANSCRIBED = "transcribed"
S_DONE        = "done"
S_ERROR       = "error"       # transient failure - reset to pending and retried each run
S_CORRUPT     = "corrupt"     # decode failed; recovery also failed - skipped (see --retry-corrupt)
S_NO_AUDIO    = "no_audio"    # source has no audio track - nothing to transcribe; never a 1:1 gap

# Terminal states for a source we will not turn into a transcript on normal reruns.
_UNRECOVERABLE_STATES = (S_CORRUPT, S_NO_AUDIO)

# Set from --retry-corrupt in main(): re-attempt sources previously deemed corrupt.
RETRY_CORRUPT = False

# ── Recovery configuration ───────────────────────────────────────────────────
# When normal streaming fails on a decodable-looking source, try harder before
# giving up: download the whole file and decode locally, reconstruct a truncated
# MP4's missing moov index with `untrunc`, or re-fetch the original from YouTube
# with `yt-dlp`.  All three are best-effort and degrade gracefully when the tool
# or the data isn't available.
ENABLE_RECOVERY          = True
RECOVERY_MAX_DOWNLOAD_MB = int(os.environ.get("RECOVERY_MAX_DOWNLOAD_MB", "4096"))
# How to invoke untrunc.  A bare binary on PATH wins; otherwise a prebuilt docker
# image (built from anthwlock/untrunc) is used.  Resolved in _detect_recovery_tools().
UNTRUNC_DOCKER_IMAGE     = os.environ.get("UNTRUNC_DOCKER_IMAGE", "untrunc-local")
_HAVE_UNTRUNC: str | None = None       # None | "binary" | "docker"
_HAVE_YTDLP:   bool       = False

# Recovery needs to pick a healthy reference file (for untrunc) from the archive,
# so the pipeline shares the video list / store with the recovery helpers here.
_RECOVERY_VIDEOS: list      = []
_RECOVERY_STORE             = None
_RECOVERY_REF_CACHE: dict   = {}       # (ext, brand) -> downloaded reference Path | None
_RECOVERY_BRAND_CACHE: dict = {}       # source_url -> ftyp brand (probe cache)
_RECOVERY_REF_LOCK          = threading.Lock()
_RECOVERY_REF_PROBE_CAP     = 80       # max remote brand probes when seeking a reference

# ── Backend detection ──────────────────────────────────────────────────────────

def _detect_backend() -> str:
    """
    Return the name of the Whisper backend to use.

    Priority:
      mlx_whisper    - Apple Silicon only
      faster_whisper - NVIDIA CUDA or CPU (any OS)
      openai_whisper - pure CPU fallback
    """
    # MLX only works on Apple Silicon.
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            import mlx_whisper  # noqa: F401
            return "mlx_whisper"
        except ImportError:
            pass

    try:
        import faster_whisper  # noqa: F401
        return "faster_whisper"
    except ImportError:
        pass

    try:
        import whisper  # openai-whisper  # noqa: F401
        return "openai_whisper"
    except ImportError:
        pass

    return "none"


WHISPER_BACKEND = _detect_backend()

# Model names vary by backend.
# faster-whisper uses plain size strings; mlx uses HF repo IDs.
_MODEL_NAME: dict[str, str] = {
    "mlx_whisper":    "mlx-community/whisper-large-v3-turbo",
    "faster_whisper": "large-v3",
    "openai_whisper": "large-v3",
}
WHISPER_MODEL = WHISPER_MODEL_OVERRIDE or _MODEL_NAME.get(WHISPER_BACKEND, "large-v3")

# ── Directory bootstrap ────────────────────────────────────────────────────────
for _d in (AUDIO_DIR, RECOVERY_DIR, TRANSCRIPT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
_file_handler = logging.FileHandler(ERROR_LOG_FILE, encoding="utf-8")
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
log = logging.getLogger("transcriber")
log.setLevel(logging.DEBUG)
log.addHandler(_file_handler)
log.propagate = False

# ── Rich + tqdm ───────────────────────────────────────────────────────────────
try:
    from rich.console import Console
except ImportError:
    print("rich is required.  Run:  pip install rich", file=sys.stderr)
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm is required.  Run:  pip install tqdm", file=sys.stderr)
    sys.exit(1)

console     = Console(stderr=True, highlight=False)
_print_lock = threading.Lock()


def _print(msg: str, style: str = "") -> None:
    with _print_lock:
        if style:
            console.print(msg, style=style)
        else:
            console.print(msg)


def _status(icon: str, colour: str, label: str, detail: str = "") -> None:
    parts = f"[{colour}]{icon}[/{colour}]  {label}"
    if detail:
        parts += f"  [dim]{detail}[/dim]"
    _print(parts)


# ── Throttle ───────────────────────────────────────────────────────────────────
_throttle_lock    = threading.Lock()
_last_stream_time = 0.0


def _throttle_stream() -> None:
    global _last_stream_time
    while True:
        with _throttle_lock:
            now  = time.monotonic()
            wait = STREAM_THROTTLE_SECS - (now - _last_stream_time)
            if wait <= 0:
                _last_stream_time = time.monotonic()
                return
        time.sleep(wait)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_elapsed(s: float) -> str:
    h = int(s // 3600); m = int((s % 3600) // 60); sec = s % 60
    if h:   return f"{h}h {m:02d}m {sec:04.1f}s"
    if m:   return f"{m}m {sec:04.1f}s"
    return f"{sec:04.1f}s"


def _fmt_duration(s: float) -> str:
    h = int(s // 3600); m = int((s % 3600) // 60); sec = int(s % 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def _audio_path(name: str) -> Path:
    return AUDIO_DIR / (Path(name).stem + ".wav")


def _stem(name: str) -> str:
    """
    Convert an archive file path into a flat filename stem suitable for use
    as a transcript filename.

    Path separators are replaced with ' -- ' and the result is clamped to
    241 UTF-8 bytes so that adding a '.json' or '.txt' extension never
    exceeds the 255-byte limit imposed by Linux/ext4 (and most other OSes).

    When truncation is required a short SHA-256 suffix (~8 hex chars) of the
    *full* stem is appended so that two different long names never collide.
    """
    normalised  = name.replace("\\", "/")
    without_ext = normalised.rsplit(".", 1)[0] if "." in Path(normalised).name else normalised
    stem        = without_ext.replace("/", " -- ")

    # Linux/ext4 hard limit is 255 bytes per filename component.
    # Reserve 14 bytes: 5 for ".json" + 1 for "~" separator + 8 for hash.
    max_bytes = 241
    encoded   = stem.encode("utf-8")
    if len(encoded) <= max_bytes:
        return stem

    short_hash = hashlib.sha256(stem.encode("utf-8")).hexdigest()[:8]
    # Decode back after truncation, ignoring any split multi-byte sequence.
    truncated  = encoded[:max_bytes].decode("utf-8", errors="ignore").rstrip()
    return f"{truncated}~{short_hash}"


def _delete_audio(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError as exc:
        log.warning("Could not delete %s: %s", path, exc)


# ── Progress store ─────────────────────────────────────────────────────────────

class ProgressStore:
    """
    On-disk state machine.  Writes are debounced: a transition marks the store
    dirty and the JSON is rewritten at most once per PROGRESS_FLUSH_INTERVAL
    seconds (plus a forced flush at process exit).  This removes the O(n²) write
    amplification of rewriting the whole file on every single transition.  Losing
    the last couple of seconds of state on a crash is harmless - those files are
    simply re-processed on the next run.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._data: dict = {}
        self._lock = threading.Lock()
        self._dirty = False
        self._last_flush = 0.0
        self._load()
        atexit.register(self.flush, True)

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as f:
                    self._data = json.load(f)
                log.info("Loaded progress for %d files.", len(self._data))
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("Could not read progress file (%s); starting fresh.", exc)

    def _write(self) -> None:
        # Caller holds the lock.
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)
        tmp.replace(self._path)
        self._dirty = False
        self._last_flush = time.monotonic()

    def _maybe_flush(self) -> None:
        # Caller holds the lock.  Write only once the debounce window elapses.
        if time.monotonic() - self._last_flush >= PROGRESS_FLUSH_INTERVAL:
            self._write()

    def flush(self, force: bool = False) -> None:
        with self._lock:
            if self._dirty or force:
                try:
                    self._write()
                except OSError as exc:
                    log.warning("Could not flush progress: %s", exc)

    def get(self, key: str) -> dict:
        with self._lock:
            return dict(self._data.get(key, {"status": S_PENDING}))

    def set(self, key: str, value: dict) -> None:
        with self._lock:
            self._data[key] = value
            self._dirty = True
            self._maybe_flush()

    def update(self, key: str, **kwargs) -> None:
        with self._lock:
            entry = self._data.get(key, {})
            entry.update(kwargs)
            self._data[key] = entry
            self._dirty = True
            self._maybe_flush()

    def all_entries(self) -> dict:
        with self._lock:
            return dict(self._data)


# ── Hashing ────────────────────────────────────────────────────────────────────

def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


# ── Ignore list ────────────────────────────────────────────────────────────────

def _load_ignores() -> list[str]:
    if not IGNORE_FILE.exists():
        return []
    patterns: list[str] = []
    with open(IGNORE_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line)
    return patterns


def _is_ignored(name: str, patterns: list[str]) -> bool:
    return any(name.startswith(p) for p in patterns)


# ── Archive helpers ─────────────────────────────────────────────────────────────

def _archive_url(path: str) -> str:
    """
    Build an archive.org download URL with the path percent-encoded.

    Without this, filenames containing '#', '+', '?' or spaces corrupt the
    request - most importantly a literal '#' (e.g. "Psalmody #3") is read as a
    URL fragment, so ffmpeg/curl truncate the URL there and the server returns
    404.  Encoding makes every filename resolve (verified: raw 404/500 → 200).
    """
    return (f"https://archive.org/download/{ARCHIVE_IDENTIFIER}/"
            + urllib.parse.quote(path, safe="/"))


def get_video_files() -> list:
    try:
        import internetarchive as ia
    except ImportError:
        console.print("[red]internetarchive not installed.[/red]  "
                      "Run: pip install 'internetarchive[speedups]'")
        sys.exit(1)
    ignore = _load_ignores()
    _status("🔍", "cyan", "Fetching file list from archive.org …")
    item      = ia.get_item(ARCHIVE_IDENTIFIER)
    all_files = {f["name"]: f for f in item.files}

    files = []
    for f in item.files:
        name = f.get("name", "")
        if _is_ignored(name, ignore):
            continue
        ext = Path(name).suffix.lower()
        if ext not in VIDEO_EXTENSIONS:
            continue

        derivative_url = None
        for audio_ext in AUDIO_DERIVATIVE_EXTENSIONS:
            # Keep the directory: archive.org file keys are full paths
            # (e.g. "videos/2016/….ogg"), so a bare stem never matches.
            candidate = str(Path(name).with_suffix(audio_ext))
            if candidate in all_files:
                derivative_url = _archive_url(candidate)
                log.info("Audio derivative found for %s: %s", name, candidate)
                break

        files.append({
            "name":           name,
            "size":           int(f.get("size", 0) or 0),
            "format":         f.get("format", ""),
            "source_url":     _archive_url(name),
            "derivative_url": derivative_url,
        })

    log.info("Found %d video files (%d with audio derivatives).",
             len(files), sum(1 for f in files if f["derivative_url"]))
    return files


# ── Audio extraction ───────────────────────────────────────────────────────────

def _stream_timeout_for(size_bytes: int) -> int:
    """
    Per-attempt ffmpeg timeout that scales with source size.

    archive.org throughput is highly variable (~0.3–1.5 MB/s under load), so a
    flat cap kills large files mid-extract.  Allow STREAM_TIMEOUT_PER_MB seconds
    per MB, floored at STREAM_TIMEOUT and capped at STREAM_TIMEOUT_MAX.
    """
    size_mb = max(0, size_bytes) / 1_048_576
    return int(min(STREAM_TIMEOUT_MAX,
                   max(STREAM_TIMEOUT, size_mb * STREAM_TIMEOUT_PER_MB)))


# ffmpeg failure fingerprints.  A *permanent* failure is a deterministic decode
# problem - the bytes archive.org served are not a decodable A/V stream, so every
# retry of the identical request fails identically (truncated upload with no moov
# index, a 0-byte file, or a video with no audio track).  A *transient* failure is
# a network/IO hiccup (reset, timeout, 5xx, premature EOF) that a retry may clear.
_TRANSIENT_FINGERPRINTS = (
    "connection reset", "reset by peer", "connection timed out",
    "connection refused", "network is unreachable", "no route to host",
    "temporary failure in name resolution", "error in the pull function",
    "server returned 5", "server returned 4", "503 ", "502 ", "500 ",
    "input/output error", "i/o error", "broken pipe", "operation timed out",
    "timed out", "end of file", "will reconnect", "http error",
)
_PERMANENT_FINGERPRINTS = (
    "moov atom not found",            # MP4 index absent - truncated/incomplete upload
    "does not contain any stream",    # after -vn: source has no audio track at all
    "could not find codec parameters",
    "invalid data found when processing input",
)


def _classify_ffmpeg_failure(stderr: str, timed_out: bool) -> tuple[str, str]:
    """
    Return ``(kind, reason)`` where *kind* is ``"transient"`` or ``"permanent"``.

    Network markers are checked *first* so a transient read error that also
    happens to print "Invalid data" is still retried; only a clean, deterministic
    container failure with no network noise is treated as permanent.  (ffmpeg is
    run with ``-hide_banner`` so its build-config line - which contains "gnutls" -
    can't masquerade as a TLS network error here.)
    """
    if timed_out:
        return "transient", "timed out"
    s = (stderr or "").lower()
    for fp in _TRANSIENT_FINGERPRINTS:
        if fp in s:
            return "transient", fp.strip()
    for fp in _PERMANENT_FINGERPRINTS:
        if fp in s:
            return "permanent", fp.strip()
    return "transient", "unknown ffmpeg error"


def _run_ffmpeg_stream(input_url: str, audio_path: Path,
                       timeout: int = STREAM_TIMEOUT) -> tuple[bool, str, str]:
    """
    Run ffmpeg, reading *input_url* as a remote HTTP/HTTPS source and writing
    a 16 kHz mono PCM WAV.  No video data touches the disk.

    Returns ``(ok, kind, reason)``: ``(True, "", "")`` on success, otherwise
    ``kind`` is ``"transient"`` or ``"permanent"`` (see ``_classify_ffmpeg_failure``)
    and ``reason`` is a short human-readable cause.
    """
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    is_remote = input_url.startswith(("http://", "https://"))
    cmd = ["ffmpeg", "-y", "-nostdin", "-hide_banner"]
    if is_remote:
        # Long streams off archive.org routinely hit transient TLS resets,
        # premature EOF, and HTTP 5xx.  Bare -reconnect only retries on a clean
        # EOF, so also reconnect on network/http errors with a longer give-up
        # window.  (These are http-protocol options - invalid for local inputs,
        # so they're only added when the source is actually an HTTP(S) URL.)
        cmd += [
            "-reconnect",                  "1",
            "-reconnect_streamed",         "1",
            "-reconnect_on_network_error", "1",
            "-reconnect_on_http_error",    "4xx,5xx",
            "-reconnect_delay_max",        "60",
        ]
    # Local paths may contain ':' (timestamps) which ffmpeg would read as a
    # protocol - the file: prefix forces literal-filename handling.
    ffmpeg_input = input_url if is_remote else f"file:{input_url}"
    cmd += [
        "-i",  ffmpeg_input,
        "-vn",
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(audio_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            kind, reason = _classify_ffmpeg_failure(result.stderr, timed_out=False)
            log.warning("[FFMPEG] exit %d (%s: %s) for %s\n%s",
                        result.returncode, kind, reason, input_url, result.stderr[-600:])
            return False, kind, reason
        if audio_path.exists() and audio_path.stat().st_size > 0:
            return True, "", ""
        # Clean exit but no audio produced - treat as transient (retry) since
        # a genuine "no audio track" surfaces as a non-zero exit handled above.
        return False, "transient", "empty output"
    except subprocess.TimeoutExpired:
        log.warning("[FFMPEG] timed out after %ds for %s", timeout, input_url)
        return False, "transient", "timed out"
    except Exception as exc:
        log.warning("[FFMPEG] unexpected error for %s: %s", input_url, exc)
        return False, "transient", str(exc)


# ── Recovery: try harder before giving up ────────────────────────────────────

def _detect_recovery_tools() -> None:
    """Resolve which recovery tools are usable.  Called once at startup."""
    global _HAVE_UNTRUNC, _HAVE_YTDLP
    if shutil.which("untrunc"):
        _HAVE_UNTRUNC = "binary"
    elif shutil.which("docker"):
        try:
            r = subprocess.run(["docker", "image", "inspect", UNTRUNC_DOCKER_IMAGE],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=30)
            if r.returncode == 0:
                _HAVE_UNTRUNC = "docker"
        except Exception:
            pass
    _HAVE_YTDLP = shutil.which("yt-dlp") is not None


def _youtube_id(name: str) -> str | None:
    """Extract an 11-char YouTube id from an archive filename, if one is present."""
    m = re.search(r'\(([A-Za-z0-9_-]{11})\)', name)          # "… (xxxxxxxxxxx).mp4"
    if m:
        return m.group(1)
    m = re.search(r'(?:^|/| )([A-Za-z0-9_-]{11}) - ', name)  # "… - xxxxxxxxxxx - …"
    return m.group(1) if m else None


def _download_full(url: str, dest: Path, max_mb: int) -> bool:
    """Stream-download an entire remote file to *dest*, capped at *max_mb*."""
    cap = max_mb * 1_048_576
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "templeos-transcriber"})
        with urllib.request.urlopen(req, timeout=STREAM_TIMEOUT) as r, open(dest, "wb") as f:
            total = 0
            while chunk := r.read(1 << 20):
                total += len(chunk)
                if total > cap:
                    log.warning("[RECOVERY] %s exceeds %d MB cap - aborting download",
                                url, max_mb)
                    dest.unlink(missing_ok=True)
                    return False
                f.write(chunk)
        return dest.exists() and dest.stat().st_size > 0
    except Exception as exc:                                  # noqa: BLE001
        log.warning("[RECOVERY] full download failed for %s: %s", url, exc)
        dest.unlink(missing_ok=True)
        return False


def _mp4_brand(path: Path) -> str | None:
    """Read the ftyp major brand (e.g. 'MSNV', 'isom') from a local MP4."""
    try:
        with open(path, "rb") as f:
            head = f.read(12)
        if head[4:8] == b"ftyp":
            return head[8:12].decode("latin1", "ignore")
    except Exception:                                         # noqa: BLE001
        pass
    return None


def _mp4_brand_remote(url: str) -> str | None:
    """Read the ftyp major brand of a remote MP4 with a 12-byte range request."""
    try:
        req = urllib.request.Request(url, headers={"Range": "bytes=0-11"})
        b = urllib.request.urlopen(req, timeout=20).read()
        if b[4:8] == b"ftyp":
            return b[8:12].decode("latin1", "ignore")
    except Exception:                                         # noqa: BLE001
        pass
    return None


def _reference_is_usable(path: Path) -> bool:
    """
    A usable untrunc reference must itself decode: have a real moov index *and* an
    audio track.  A truncated file - e.g. a source we only partially *recovered*
    (still truncated on archive.org) - has neither and would make untrunc fail or
    emit garbage, so such files must never be used as references.
    """
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
             "-of", "csv=p=0", f"file:{path}"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=120)
        return "audio" in (r.stdout or "")
    except Exception:                                         # noqa: BLE001
        return False


def _get_reference(name_ext: str, brand: str | None, min_size: int = 0) -> Path | None:
    """
    Download (once, cached per brand) a *brand-matched, genuinely healthy* file to
    seed untrunc.  untrunc copies the reference's codec/timescale *and only rebuilds
    up to the reference's own duration*, so the reference must be at least as long
    as the broken file or recovery is truncated to a few seconds (verified).  A
    mismatched brand instead yields empty/garbage audio (also verified).

    Candidates are `done` files that were *not themselves recovered* (a recovered
    source is still truncated on the server and useless as a reference), preferring
    a brand match, exact extension case, and size ≥ *min_size*.  Each candidate is
    download-then-verified to actually decode before being accepted.
    """
    ext = name_ext.lower()
    key = (ext, brand or "")
    with _RECOVERY_REF_LOCK:
        if key in _RECOVERY_REF_CACHE:
            p = _RECOVERY_REF_CACHE[key]
            return p if (p and p.exists()) else None

        store = _RECOVERY_STORE

        def is_candidate(fi) -> bool:
            if Path(fi["name"]).suffix.lower() != ext:
                return False
            if store is None:
                return True
            e = store.get(fi["name"])
            # Healthy = transcribed from a clean extraction, not a partial recovery.
            return e.get("status") == S_DONE and not e.get("recovered_via")

        cands = [fi for fi in _RECOVERY_VIDEOS if is_candidate(fi)]

        def sort_key(fi):
            exact      = Path(fi["name"]).suffix != name_ext     # exact-case first
            size       = fi.get("size", 0)
            big_enough = size >= min_size
            # long-enough refs first (smallest of them); else fall back largest-first
            return (exact, 0, size) if big_enough else (exact, 1, -size)

        cands.sort(key=sort_key)

        ref_path: Path | None = None
        probed = 0
        dest = RECOVERY_DIR / f"_reference_{brand or 'any'}{ext}"
        for fi in cands:
            if brand:                                # require a matching ftyp brand
                url = fi["source_url"]
                if url not in _RECOVERY_BRAND_CACHE:
                    if probed >= _RECOVERY_REF_PROBE_CAP:
                        break
                    probed += 1
                    _RECOVERY_BRAND_CACHE[url] = _mp4_brand_remote(url)
                if _RECOVERY_BRAND_CACHE[url] != brand:
                    continue
            _status("⬇", "cyan",
                    f"Recovery: fetching untrunc reference [{brand or 'any'}] "
                    f"({fi['size']/1_048_576:.1f} MB): {Path(fi['name']).name}")
            if not _download_full(fi["source_url"], dest, RECOVERY_MAX_DOWNLOAD_MB):
                continue
            if not _reference_is_usable(dest):        # itself truncated/broken - skip
                log.info("[RECOVERY] candidate reference unusable, trying next: %s",
                         fi["name"])
                dest.unlink(missing_ok=True)
                continue
            ref_path = dest
            break
        _RECOVERY_REF_CACHE[key] = ref_path
        return ref_path


def _run_untrunc(reference: Path, broken: Path) -> Path | None:
    """Reconstruct *broken*'s missing index from *reference* with untrunc."""
    # untrunc strips the extension and writes "<stem>_fixed.mp4" next to the input.
    for stale in broken.parent.glob(broken.stem + "_fixed*"):
        stale.unlink(missing_ok=True)
    try:
        if _HAVE_UNTRUNC == "binary":
            cmd = ["untrunc", str(reference), str(broken)]
        else:
            d = str(RECOVERY_DIR.resolve())
            cmd = ["docker", "run", "--rm",
                   "--user", f"{os.getuid()}:{os.getgid()}",
                   "-v", f"{d}:/data", UNTRUNC_DOCKER_IMAGE,
                   f"/data/{reference.name}", f"/data/{broken.name}"]
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                           text=True, timeout=1800)
        fixed = next(iter(broken.parent.glob(broken.stem + "_fixed*")), None)
        if fixed and fixed.exists() and fixed.stat().st_size > 0:
            return fixed
        log.warning("[RECOVERY] untrunc produced no output for %s:\n%s",
                    broken.name, (r.stderr or "")[-500:])
        return None
    except Exception as exc:                                  # noqa: BLE001
        log.warning("[RECOVERY] untrunc failed for %s: %s", broken.name, exc)
        return None


def _run_ytdlp(yt_id: str, audio_path: Path) -> bool:
    """Fetch bestaudio for a YouTube id and normalise it to our 16 kHz mono WAV."""
    stem = RECOVERY_DIR / f"_yt_{yt_id}"
    for stale in RECOVERY_DIR.glob(f"_yt_{yt_id}.*"):
        stale.unlink(missing_ok=True)
    try:
        cmd = ["yt-dlp", "-q", "--no-playlist", "-f", "bestaudio/best",
               "-o", f"{stem}.%(ext)s", f"https://www.youtube.com/watch?v={yt_id}"]
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                           text=True, timeout=1800)
        produced = next(iter(RECOVERY_DIR.glob(f"_yt_{yt_id}.*")), None)
        if not produced or produced.stat().st_size == 0:
            log.info("[RECOVERY] yt-dlp got nothing for %s: %s",
                     yt_id, (r.stderr or "")[-300:])
            return False
        ok, _, _ = _run_ffmpeg_stream(str(produced), audio_path)
        produced.unlink(missing_ok=True)
        return ok
    except Exception as exc:                                  # noqa: BLE001
        log.warning("[RECOVERY] yt-dlp failed for %s: %s", yt_id, exc)
        return False


def _attempt_recovery(file_info: dict, audio_path: Path, reason: str) -> tuple[bool, str]:
    """
    Best-effort recovery after normal streaming gave a permanent decode failure.

    Ladder (cheapest first, skipping steps that can't apply):
      1. full download + local decode   - fixes HTTP-seek-only failures
      2. untrunc                          - rebuilds a truncated MP4's moov index
      3. yt-dlp                           - re-fetches the original from YouTube

    Returns ``(ok, method)``.
    """
    name     = file_info["name"]
    short    = Path(name).name
    ext      = Path(name).suffix.lower()
    yt       = _youtube_id(name)
    size_mb  = file_info.get("size", 0) / 1_048_576
    no_audio = (reason == "does not contain any stream")

    # Local recovery needs the file's bytes and only helps when there is audio
    # data to recover - a genuine "no audio track" can't be fixed locally.
    if not no_audio and size_mb <= RECOVERY_MAX_DOWNLOAD_MB:
        local = RECOVERY_DIR / (_stem(name) + ext)
        local.unlink(missing_ok=True)
        _status("🛟", "cyan",
                f"Recovery: downloading full source ({size_mb:.1f} MB): {short}")
        if _download_full(file_info["source_url"], local, RECOVERY_MAX_DOWNLOAD_MB):
            ok, _, _ = _run_ffmpeg_stream(str(local), audio_path)      # 1) local decode
            if ok:
                local.unlink(missing_ok=True)
                return True, "full-download"
            if _HAVE_UNTRUNC and ext in (".mp4", ".m4v", ".mov"):       # 2) untrunc
                ref = _get_reference(Path(name).suffix, _mp4_brand(local),
                                     min_size=file_info.get("size", 0))
                if ref:
                    _status("🛟", "cyan", f"Recovery: untrunc reconstruction: {short}")
                    fixed = _run_untrunc(ref, local)
                    if fixed:
                        ok, _, _ = _run_ffmpeg_stream(str(fixed), audio_path)
                        fixed.unlink(missing_ok=True)
                        if ok:
                            local.unlink(missing_ok=True)
                            return True, "untrunc"
            local.unlink(missing_ok=True)

    if yt and _HAVE_YTDLP:                                              # 3) yt-dlp
        _status("🛟", "cyan", f"Recovery: yt-dlp re-download (youtube {yt}): {short}")
        if _run_ytdlp(yt, audio_path):
            return True, "yt-dlp"

    return False, ""


def _extract_audio_with_retry(file_info: dict,
                              audio_path: Path) -> tuple[bool, bool, str, str]:
    """
    Try to extract audio, retrying transient failures with exponential backoff.

    Returns ``(ok, permanent, reason, recovered_via)``.  *permanent* is True only
    when the authoritative video stream produced a deterministic decode failure,
    meaning the source is undecodable and must not be re-attempted on future runs.
    *recovered_via* names the recovery method when normal streaming failed but a
    fallback succeeded (empty for an ordinary first-try success).
    """
    name      = file_info["name"]
    short     = Path(name).name
    video_url = file_info["source_url"]
    deriv_url = file_info.get("derivative_url")

    if audio_path.exists() and audio_path.stat().st_size == 0:
        audio_path.unlink()

    timeout     = _stream_timeout_for(file_info.get("size", 0))
    last_reason = "audio extraction failed after all retries"

    for attempt in range(1, STREAM_RETRIES + 1):
        _throttle_stream()

        if deriv_url and attempt % 2 == 1:
            url  = deriv_url
            tier = "derivative"
        else:
            url  = video_url
            tier = "video-stream"

        _status("🔊", "yellow",
                f"Extracting audio [{tier}] ({attempt}/{STREAM_RETRIES}): {short}")
        ok, kind, reason = _run_ffmpeg_stream(url, audio_path, timeout=timeout)
        if ok:
            log.info("[AUDIO] success tier=%s attempt=%d file=%s", tier, attempt, name)
            return True, False, "", ""

        last_reason = reason or last_reason

        # A deterministic decode failure on the authoritative video stream means
        # the bytes themselves are undecodable - retrying the identical request
        # only wastes minutes of backoff.  Try harder recovery methods, then,
        # failing those, give up now and skip it on reruns.
        if kind == "permanent" and tier == "video-stream":
            if ENABLE_RECOVERY:
                recovered, method = _attempt_recovery(file_info, audio_path, reason)
                if recovered:
                    log.info("[AUDIO] recovered via %s file=%s", method, name)
                    _status("🛟", "green", f"Recovered via {method}: {short}")
                    return True, False, "", method
            log.error("[AUDIO] unrecoverable source (%s) file=%s", reason, name)
            _status("🚫", "magenta",
                    f"Unrecoverable source ({reason}); skipping: {short}")
            return False, True, reason, ""

        if attempt < STREAM_RETRIES:
            backoff = min(STREAM_BACKOFF_BASE * (2 ** (attempt - 1)), STREAM_BACKOFF_MAX)
            log.warning("Audio extraction attempt %d/%d failed (%s) for %s - sleeping %ds",
                        attempt, STREAM_RETRIES, reason, name, backoff)
            _status("⏳", "yellow",
                    f"Attempt {attempt}/{STREAM_RETRIES} failed ({reason}), "
                    f"retrying in {backoff}s…", short)
            time.sleep(backoff)
        else:
            log.error("All %d audio extraction attempts failed for %s (last: %s)",
                      STREAM_RETRIES, name, last_reason)
    return False, False, last_reason, ""


# ── Transcription - backend dispatch ──────────────────────────────────────────

def _determine_compute_device() -> tuple[str, str]:
    """
    Return (device, compute_type) appropriate for faster-whisper.

    Tries CUDA first (NVIDIA), then CPU.
    Uses float16 on GPU for speed/accuracy; int8 on CPU for speed.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
    except ImportError:
        pass

    # Try ctranslate2 directly (faster-whisper's backend)
    try:
        import ctranslate2
        devices = ctranslate2.get_supported_compute_types("cuda")
        if devices:
            return "cuda", "float16"
    except Exception:
        pass

    return "cpu", "int8"


def _transcribe_mlx(audio_path: Path) -> dict | None:
    """Transcribe using mlx-whisper (Apple Silicon)."""
    import traceback
    import mlx_whisper
    try:
        return mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=WHISPER_MODEL,
            word_timestamps=True,
            language=None,
            verbose=False,
        )
    except Exception as exc:
        tb = traceback.format_exc()
        log.error("mlx-whisper error for %s: %s\n%s", audio_path.name, exc, tb)
        _status("\u274c", "red", f"mlx-whisper exception on {audio_path.name}: {exc}")
        return None


# ── faster-whisper: eagerly-loaded singleton with OOM recovery ─────────────────

class _FasterWhisperModelHolder:
    """
    Holds the faster-whisper model singleton (loaded once at startup) plus, when
    the installed faster-whisper supports it, a BatchedInferencePipeline that
    transcribes VAD-chunked audio in batches for a large GPU speedup.

    CUDA out-of-memory recovery ladder (each step is sticky for the rest of the
    run): (1) halve the batch size, down to 1 = batching disabled; then
    (2) reload the model on CPU int8.  If nothing helps the job errors out and
    the pipeline moves on.
    """

    def __init__(self) -> None:
        self._model   = None
        self._batched = None                  # BatchedInferencePipeline | None
        self._lock    = threading.Lock()      # serialises transcriptions
        self._device:       str = "cpu"
        self._compute_type: str = "int8"
        self._batch_size:   int = max(1, BATCH_SIZE)

    def load(self, device: str, compute_type: str) -> None:
        """Eagerly load the model.  Call once before the pipeline starts."""
        with self._lock:
            self._device       = device
            self._compute_type = compute_type
            self._batch_size   = max(1, BATCH_SIZE)   # pick up CLI/env override
            self._model        = self._create_model(device, compute_type)
            self._batched      = self._maybe_wrap_batched(self._model)

    def _create_model(self, device: str, compute_type: str):
        from faster_whisper import WhisperModel
        log.info("Loading faster-whisper model: %s  device=%s  compute_type=%s",
                 WHISPER_MODEL, device, compute_type)
        _status("🖥️", "cyan",
                f"Loading faster-whisper model ({WHISPER_MODEL})  "
                f"device={device}  compute={compute_type}")
        model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute_type)
        log.info("faster-whisper model loaded.")
        return model

    def _maybe_wrap_batched(self, model):
        """Return a BatchedInferencePipeline, or None to use sequential decoding."""
        if self._batch_size <= 1:
            return None
        try:
            from faster_whisper import BatchedInferencePipeline
        except ImportError:
            log.info("Installed faster-whisper has no BatchedInferencePipeline; "
                     "using sequential transcription.")
            return None
        try:
            pipe = BatchedInferencePipeline(model=model)
            log.info("Batched inference enabled (batch_size=%d).", self._batch_size)
            _status("⚡", "cyan",
                    f"Batched inference enabled (batch_size={self._batch_size})")
            return pipe
        except Exception as exc:
            log.warning("Could not enable batched inference (%s); using sequential.", exc)
            return None

    def _reload_cpu_fallback(self) -> bool:
        """
        After a CUDA OOM, drop to CPU int8.  Returns True on success.
        Only called while _lock is held.
        """
        if self._device == "cpu":
            return False
        log.warning("CUDA OOM: falling back to CPU int8 for the remainder of the run.")
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            self._model        = self._create_model("cpu", "int8")
            self._device       = "cpu"
            self._compute_type = "int8"
            self._batched      = self._maybe_wrap_batched(self._model)
            return True
        except Exception as exc:
            log.error("CPU fallback model load failed: %s", exc)
            return False

    def _run_once(self, audio_path: Path) -> dict:
        """One transcription pass with the current model and batch settings."""
        if self._batched is not None:
            segments_gen, info = self._batched.transcribe(
                str(audio_path),
                batch_size=self._batch_size,
                word_timestamps=True,
                language=None,
                beam_size=BEAM_SIZE,
            )
        else:
            segments_gen, info = self._model.transcribe(
                str(audio_path),
                word_timestamps=True,
                language=None,
                beam_size=BEAM_SIZE,
                vad_filter=True,
            )

        text_parts:   list[str]  = []
        segments_out: list[dict] = []
        for s in segments_gen:                       # consume the lazy generator once
            text_parts.append(s.text.strip())
            segments_out.append({
                "id":             s.id,
                "start":          s.start,
                "end":            s.end,
                "text":           s.text,
                "avg_logprob":    s.avg_logprob,
                "no_speech_prob": s.no_speech_prob,
                "words": [
                    {"word": w.word, "start": w.start,
                     "end": w.end, "probability": w.probability}
                    for w in (s.words or [])
                ],
            })
        return {
            "text":     " ".join(text_parts),
            "segments": segments_out,
            "language": info.language,
        }

    def transcribe(self, audio_path: Path) -> dict | None:
        """
        Transcribe *audio_path* with the loaded model.

        Thread-safe (one transcription at a time).  On CUDA OOM it first shrinks
        the batch size, then falls back to CPU int8, retrying after each step.
        """
        with self._lock:
            for _ in range(8):                       # bounded recovery attempts
                try:
                    return self._run_once(audio_path)
                except Exception as exc:             # noqa: BLE001
                    msg    = str(exc).lower()
                    is_oom = "out of memory" in msg or ("cuda" in msg and "memory" in msg)

                    if is_oom and self._batched is not None and self._batch_size > 1:
                        new_bs = max(1, self._batch_size // 2)
                        log.warning("CUDA OOM on %s - batch_size %d→%d, retrying.",
                                    audio_path.name, self._batch_size, new_bs)
                        _status("⚠️", "yellow",
                                f"CUDA OOM - reducing batch_size to {new_bs} and retrying…")
                        self._batch_size = new_bs
                        if new_bs <= 1:
                            self._batched = None     # batching fully disabled now
                        try:
                            import torch
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        continue

                    if is_oom and self._device != "cpu":
                        _status("⚠️", "yellow",
                                f"CUDA OOM on {audio_path.name} - reloading on CPU…")
                        if self._reload_cpu_fallback():
                            continue

                    log.error("faster-whisper error for %s: %s",
                              audio_path.name, exc, exc_info=True)
                    _status("❌", "red",
                            f"faster-whisper exception on {audio_path.name}: {exc}")
                    return None

            return None


_faster_whisper_holder = _FasterWhisperModelHolder()


def _transcribe_faster(audio_path: Path) -> dict | None:
    """Transcribe using faster-whisper via the singleton holder."""
    return _faster_whisper_holder.transcribe(audio_path)


def _transcribe_openai(audio_path: Path) -> dict | None:
    """Transcribe using openai-whisper (pure CPU fallback)."""
    import traceback
    import whisper
    try:
        model = whisper.load_model(WHISPER_MODEL)
        return model.transcribe(
            str(audio_path),
            word_timestamps=True,
            language=None,
            verbose=False,
        )
    except Exception as exc:
        tb = traceback.format_exc()
        log.error("openai-whisper error for %s: %s\n%s", audio_path.name, exc, tb)
        _status("\u274c", "red", f"openai-whisper exception on {audio_path.name}: {exc}")
        return None


def _transcribe(audio_path: Path, file_info: dict,
                recovered_via: str = "") -> dict | None:
    """
    Dispatch to the correct Whisper backend, then normalise the output into
    our canonical transcript dict format.  *recovered_via*, when set, flags that
    the audio was rebuilt from a broken source (likely partial) and records how.
    """
    if WHISPER_BACKEND == "mlx_whisper":
        raw = _transcribe_mlx(audio_path)
    elif WHISPER_BACKEND == "faster_whisper":
        raw = _transcribe_faster(audio_path)
    elif WHISPER_BACKEND == "openai_whisper":
        raw = _transcribe_openai(audio_path)
    else:
        log.error("No Whisper backend available.")
        return None

    if raw is None:
        return None

    segments_out = []
    for seg in raw.get("segments", []):
        segments_out.append({
            "id":             seg.get("id"),
            "start":          round(float(seg.get("start", 0.0)), 3),
            "end":            round(float(seg.get("end",   0.0)), 3),
            "text":           (seg.get("text") or "").strip().replace(chr(8212), "-"),
            "words": [
                {
                    "word":        (w.get("word") or "").strip().replace(chr(8212), "-"),
                    "start":       round(float(w.get("start", 0.0)), 3),
                    "end":         round(float(w.get("end",   0.0)), 3),
                    "probability": round(float(w.get("probability", 0.0)), 4),
                }
                for w in seg.get("words", [])
            ],
            "avg_logprob":    round(float(seg.get("avg_logprob",    0.0)), 4),
            "no_speech_prob": round(float(seg.get("no_speech_prob", 0.0)), 4),
        })

    duration = segments_out[-1]["end"] if segments_out else 0.0
    return {
        "source": {
            "archive_identifier": ARCHIVE_IDENTIFIER,
            "filename":           file_info["name"],
            "source_url":         file_info["source_url"],
            "derivative_url":     file_info.get("derivative_url"),
            "file_format":        file_info.get("format", ""),
            "file_size_bytes":    file_info.get("size", 0),
        },
        "transcription": {
            "model":              WHISPER_MODEL,
            "backend":            WHISPER_BACKEND,
            "language_detected":  raw.get("language", "unknown"),
            "transcribed_at_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds":   duration,
            "platform":           platform.system(),
            "machine":            platform.machine(),
            # When set, the source was corrupt and this audio was reconstructed
            # (e.g. via untrunc) - likely only a partial recovery of the original.
            "recovered_via":      recovered_via or None,
            "partial_recovery":   bool(recovered_via),
        },
        "full_text": raw.get("text", "").strip().replace(chr(8212), "-"),
        "segments":  segments_out,
    }


# ── Save transcript ────────────────────────────────────────────────────────────

def _fmt_ts(seconds: float) -> str:
    h = int(seconds // 3600); m = int((seconds % 3600) // 60); s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _save_transcript(transcript: dict, stem_name: str) -> Path:
    json_path = TRANSCRIPT_DIR / f"{stem_name}.json"
    txt_path  = TRANSCRIPT_DIR / f"{stem_name}.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    src  = transcript["source"]
    meta = transcript["transcription"]
    lines = [
        "=" * 72,
        f"FILE        : {src['filename']}",
        f"SOURCE URL  : {src['source_url']}",
        f"DERIVATIVE  : {src.get('derivative_url') or '(none - streamed from video)'}",
        f"FORMAT      : {src['file_format']}",
        f"SIZE        : {src['file_size_bytes']:,} bytes",
        f"LANGUAGE    : {meta['language_detected']}",
        f"MODEL       : {meta['model']}",
        f"BACKEND     : {meta['backend']}",
        f"PLATFORM    : {meta['platform']} / {meta['machine']}",
        f"DURATION    : {_fmt_ts(meta['duration_seconds'])}",
        f"TRANSCRIBED : {meta['transcribed_at_utc']}",
        *( [f"RECOVERED   : partial - source was corrupt, rebuilt via "
            f"{meta['recovered_via']}"] if meta.get("recovered_via") else [] ),
        "=" * 72, "",
        "── FULL TRANSCRIPT " + "─" * 53,
        transcript["full_text"], "",
        "── TIMESTAMPED SEGMENTS " + "─" * 48,
    ]
    for seg in transcript["segments"]:
        lines.append(
            f"[{_fmt_ts(seg['start'])} -> {_fmt_ts(seg['end'])}]  {seg['text']}"
        )
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return json_path


# ── Pipeline stages ────────────────────────────────────────────────────────────

_SENTINEL = object()


def extraction_stage(file_info: dict, store: ProgressStore) -> dict:
    name   = file_info["name"]
    short  = Path(name).name
    apath  = _audio_path(name)
    entry  = store.get(name)
    status = entry.get("status", S_PENDING)
    t_ex   = 0.0

    # No audio track: nothing to transcribe, ever - always skip (cheap, no network).
    if status == S_NO_AUDIO:
        return {"file_info": file_info, "status": S_NO_AUDIO, "elapsed_extract": 0.0}

    # Undecodable last time: skip without touching the network unless the user
    # explicitly asked to re-attempt corrupt sources (--retry-corrupt).
    if status == S_CORRUPT and not RETRY_CORRUPT:
        return {"file_info": file_info, "status": S_CORRUPT, "elapsed_extract": 0.0}

    # Transient error (or a forced corrupt retry): clear it and try again.
    if status in (S_ERROR, S_CORRUPT):
        store.set(name, {"status": S_PENDING})
        entry  = store.get(name)
        status = S_PENDING

    if status in (S_TRANSCRIBED, S_DONE):
        return {"file_info": file_info, "status": status, "elapsed_extract": 0.0}

    if status == S_EXTRACTED and not apath.exists():
        _status("⚠", "yellow", f"WAV deleted externally, re-extracting: {short}")
        store.set(name, {"status": S_PENDING})
        entry  = store.get(name)
        status = S_PENDING

    if status == S_EXTRACTED and apath.exists():
        cur_sha   = _sha256(apath)
        saved_sha = entry.get("audio_sha256", "")
        if saved_sha and cur_sha != saved_sha:
            _status("⚠", "yellow", f"WAV hash mismatch, re-extracting: {short}")
            _delete_audio(apath)
            store.set(name, {"status": S_PENDING})
            status = S_PENDING
        else:
            return {"file_info": file_info, "status": S_EXTRACTED,
                    "elapsed_extract": 0.0}

    deriv   = file_info.get("derivative_url")
    size_mb = file_info["size"] / 1_048_576
    label   = (f"Streaming audio ({size_mb:.1f} MB source)"
               if not deriv else "Downloading audio derivative")
    _status("🔊", "yellow", f"{label}: {short}")

    t0   = time.monotonic()
    ok, permanent, reason, recovered_via = _extract_audio_with_retry(file_info, apath)
    t_ex = time.monotonic() - t0

    if not ok or not apath.exists():
        if not permanent:
            new_status = S_ERROR
        elif reason == "does not contain any stream":
            new_status = S_NO_AUDIO          # no audio track - not a 1:1 gap
        else:
            new_status = S_CORRUPT           # truncated / undecodable
        msg = reason or "audio extraction failed after all retries"
        log.error("[EX] %s (%s)  file=%s", msg, new_status, name)
        if new_status == S_NO_AUDIO:
            _status("🔇", "blue", f"No audio track (nothing to transcribe): {short}")
        elif new_status == S_CORRUPT:
            _status("🚫", "magenta", f"Unrecoverable (will skip on reruns): {short}")
        else:
            _status("❌", "red", f"Extraction failed (will retry next run): {short}")
        store.set(name, {"status": new_status, "error_stage": "extraction",
                         "error_msg": msg,
                         "failed_at": datetime.now(timezone.utc).isoformat()})
        return {"file_info": file_info, "status": new_status, "elapsed_extract": t_ex}

    sha   = _sha256(apath)
    entry = {"status": S_EXTRACTED, "audio_sha256": sha,
             "extracted_at": datetime.now(timezone.utc).isoformat()}
    if recovered_via:
        entry["recovered_via"] = recovered_via      # partial: rebuilt from a broken source
    store.set(name, entry)
    ready = (f"Audio ready (recovered via {recovered_via}): {short}"
             if recovered_via else f"Audio ready: {short}")
    _status("🔊", "yellow", ready, _fmt_elapsed(t_ex))
    return {"file_info": file_info, "status": S_EXTRACTED, "elapsed_extract": t_ex}


def process_stage(ex_result: dict, store: ProgressStore) -> dict:
    file_info  = ex_result["file_info"]
    name       = file_info["name"]
    short      = Path(name).name
    apath      = _audio_path(name)
    stem_name  = _stem(name)
    json_path  = TRANSCRIPT_DIR / f"{stem_name}.json"
    txt_path   = TRANSCRIPT_DIR / f"{stem_name}.txt"

    entry      = store.get(name)
    status     = ex_result["status"]
    t_ex       = ex_result["elapsed_extract"]
    t_tr       = 0.0
    vid_dur    = 0.0
    proc_start = time.monotonic()

    def _elapsed_total() -> float:
        return t_ex + (time.monotonic() - proc_start)

    def _result(s: str) -> dict:
        return {"status": s, "video_duration": vid_dur,
                "elapsed_extract": t_ex, "elapsed_transcribe": t_tr,
                "elapsed_total": _elapsed_total()}

    def _fail(stage: str, msg: str) -> dict:
        log.error("[PR] stage=%s  %s  file=%s", stage, msg, name)
        _status("❌", "red", f"{stage}: {msg[:60]}")
        store.set(name, {"status": S_ERROR, "error_stage": stage,
                         "error_msg": msg,
                         "failed_at": datetime.now(timezone.utc).isoformat()})
        return _result(S_ERROR)

    if status in (S_ERROR, S_CORRUPT, S_NO_AUDIO, S_DONE):
        return _result(status)

    if status == S_EXTRACTED:
        if not apath.exists():
            _status("⚠", "yellow",
                    f"WAV deleted before transcription, re-extracting: {short}")
            re_ex = extraction_stage(file_info, store)
            t_ex += re_ex["elapsed_extract"]
            if re_ex["status"] != S_EXTRACTED:
                return _fail("re-extraction",
                             "re-extraction after external deletion failed")

        if json_path.exists():
            cur_sha   = _sha256(json_path)
            saved_sha = entry.get("transcript_sha256", "")
            if saved_sha and cur_sha == saved_sha:
                _delete_audio(apath)
                store.update(name, status=S_DONE,
                             completed_at=datetime.now(timezone.utc).isoformat())
                return _result(S_DONE)
            json_path.unlink()
            txt_path.unlink(missing_ok=True)

        _status("✍", "green", f"Transcribing: {short}")
        t0         = time.monotonic()
        transcript = _transcribe(apath, file_info,
                                 recovered_via=entry.get("recovered_via", ""))
        t_tr       = time.monotonic() - t0
        if transcript is None:
            return _fail("transcription", "Whisper returned no result")

        vid_dur    = transcript["transcription"].get("duration_seconds", 0.0)
        saved_json = _save_transcript(transcript, stem_name)
        sha        = _sha256(saved_json)
        store.update(name, status=S_TRANSCRIBED, transcript_sha256=sha,
                     transcript_json=str(saved_json),
                     transcript_txt=str(txt_path),
                     transcribed_at=datetime.now(timezone.utc).isoformat())
        status = S_TRANSCRIBED

    if status == S_TRANSCRIBED:
        _delete_audio(apath)
        store.update(name, status=S_DONE,
                     completed_at=datetime.now(timezone.utc).isoformat())
        return _result(S_DONE)

    return _result(entry.get("status", S_ERROR))


# ── Transcript verification ────────────────────────────────────────────────────

_REQUIRED_TRANSCRIPT_KEYS    = {"source", "transcription", "full_text", "segments"}
_REQUIRED_SOURCE_KEYS        = {"archive_identifier", "filename", "source_url"}
_REQUIRED_TRANSCRIPTION_KEYS = {"model", "language_detected", "duration_seconds"}


def _check_transcript_json(path: Path) -> str | None:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        return f"unreadable: {exc}"
    missing = _REQUIRED_TRANSCRIPT_KEYS - data.keys()
    if missing:
        return f"missing top-level keys: {sorted(missing)}"
    if not isinstance(data["segments"], list):
        return "segments is not a list"
    if _REQUIRED_SOURCE_KEYS - data.get("source", {}).keys():
        return f"missing source keys: {sorted(_REQUIRED_SOURCE_KEYS - data['source'].keys())}"
    if _REQUIRED_TRANSCRIPTION_KEYS - data.get("transcription", {}).keys():
        return "missing transcription keys"
    if not isinstance(data.get("full_text"), str):
        return "full_text is not a string"
    return None


def verify_transcripts(video_list: list, store: ProgressStore,
                       deep: bool = False) -> int:
    """
    Sanity-check completed transcripts and reset any that are broken.

    Fast path (default): confirm the .json and .txt exist and, for legacy
    entries lacking a stored hash, that the JSON is structurally valid - an
    O(files) stat pass.  Pass *deep* (``--verify-hashes``) to additionally
    re-compute every SHA-256, which is O(total bytes) and grows with the
    archive, so it is opt-in rather than run on every startup.
    """
    snapshot     = store.all_entries()
    done_entries = [
        (fi, snapshot.get(fi["name"], {}))
        for fi in video_list
        if snapshot.get(fi["name"], {}).get("status") == S_DONE
    ]
    if not done_entries:
        console.print("[dim]  No completed transcripts to verify.[/dim]")
        return 0

    console.print(
        f"\n[bold cyan]Verifying {len(done_entries)} completed transcript(s)"
        f"{' (deep / SHA-256)' if deep else ''}…[/bold cyan]"
    )

    ok_count = reset_count = no_hash_count = 0
    issues: list[tuple[str, str]] = []

    with tqdm(total=len(done_entries), unit="file", desc="Verifying",
              dynamic_ncols=True, colour="yellow") as pbar:
        for fi, entry in done_entries:
            name      = fi["name"]
            short     = Path(name).name
            stem_name = _stem(name)
            json_path = TRANSCRIPT_DIR / f"{stem_name}.json"
            txt_path  = TRANSCRIPT_DIR / f"{stem_name}.txt"
            saved_sha = entry.get("transcript_sha256", "")
            reason    = None

            pbar.set_postfix_str(short[:40], refresh=False)

            if not json_path.exists():
                reason = "transcript JSON missing from disk"
            elif not txt_path.exists():
                reason = ".txt file missing"
            elif not saved_sha:
                no_hash_count += 1
                reason = _check_transcript_json(json_path)
                if reason:
                    reason = f"no stored hash + {reason}"
            elif deep:
                actual_sha = _sha256(json_path)
                if actual_sha != saved_sha:
                    reason = (f"SHA-256 mismatch "
                              f"(stored={saved_sha[:12]}… got={actual_sha[:12]}…)")
                else:
                    reason = _check_transcript_json(json_path)

            if reason:
                reset_count += 1
                issues.append((short, reason))
                for p in (json_path, txt_path):
                    if p.exists():
                        try:
                            p.unlink()
                        except OSError:
                            pass
                store.set(name, {"status": S_PENDING,
                                 "verify_reset_reason": reason,
                                 "verify_reset_at": datetime.now(timezone.utc).isoformat()})
            else:
                ok_count += 1

            pbar.update(1)

    console.print()
    if reset_count == 0:
        console.print(
            f"[bold green]  ✅ All {ok_count} transcript(s) verified OK.[/bold green]"
        )
    else:
        console.print(
            f"[bold yellow]  ⚠  {reset_count} transcript(s) failed and will "
            f"be re-transcribed:[/bold yellow]"
        )
        for short, reason in issues:
            console.print(f"     [red]•[/red] [dim]{short}[/dim]  -  {reason}")
    if no_hash_count:
        console.print(f"  [dim]{no_hash_count} entry/entries had no stored hash.[/dim]")
    console.print()
    return reset_count


# ── Unrecoverable report ────────────────────────────────────────────────────────

def _write_unrecoverable_report(video_list: list, store: ProgressStore) -> tuple[int, int]:
    """
    Write output/logs/unrecoverable.csv listing every source we could not turn
    into a transcript, with archive.org (and YouTube, when derivable) links so
    they can be investigated by hand.  Returns (n_corrupt, n_no_audio).
    """
    entries = store.all_entries()
    rows: list[dict] = []
    n_corrupt = n_no_audio = 0
    for fi in video_list:
        st = entries.get(fi["name"], {}).get("status")
        if st not in _UNRECOVERABLE_STATES:
            continue
        if st == S_CORRUPT:
            n_corrupt += 1
        else:
            n_no_audio += 1
        yt = _youtube_id(fi["name"])
        rows.append({
            "status":      st,
            "reason":      entries.get(fi["name"], {}).get("error_msg", ""),
            "size_bytes":  fi.get("size", 0),
            "filename":    fi["name"],
            "archive_url": fi["source_url"],
            "youtube_url": f"https://www.youtube.com/watch?v={yt}" if yt else "",
        })

    rows.sort(key=lambda r: (r["status"], r["filename"]))
    try:
        with open(UNRECOVERABLE_REPORT, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["status", "reason", "size_bytes",
                                              "filename", "archive_url", "youtube_url"])
            w.writeheader()
            w.writerows(rows)
    except OSError as exc:
        log.warning("Could not write unrecoverable report: %s", exc)
    return n_corrupt, n_no_audio


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    global WHISPER_MODEL, BEAM_SIZE, BATCH_SIZE, MAX_CONCURRENT_EXTRACTIONS, RETRY_CORRUPT
    global ENABLE_RECOVERY, RECOVERY_MAX_DOWNLOAD_MB, _RECOVERY_VIDEOS, _RECOVERY_STORE

    parser = argparse.ArgumentParser(
        description="Stream audio from the TempleOS archive and transcribe it.")
    parser.add_argument("--model", default=None,
                        help="Whisper model override (e.g. large-v3-turbo).")
    parser.add_argument("--beam-size", type=int, default=BEAM_SIZE,
                        help="Beam size; 1 = greedy (faster, slightly worse).")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="faster-whisper batch size; 1 disables batching.")
    parser.add_argument("--workers", type=int, default=MAX_CONCURRENT_EXTRACTIONS,
                        help="Parallel audio extractions (default 5).")
    parser.add_argument("--verify-hashes", action="store_true",
                        help="Re-compute SHA-256 of every done transcript at startup.")
    parser.add_argument("--retry-corrupt", action="store_true",
                        help="Re-attempt sources previously marked unrecoverable "
                             "(truncated / no-moov).")
    parser.add_argument("--no-recovery", action="store_true",
                        help="Disable recovery attempts (full-download / untrunc / yt-dlp).")
    parser.add_argument("--recovery-max-mb", type=int, default=RECOVERY_MAX_DOWNLOAD_MB,
                        help="Skip recovery downloads larger than this many MB "
                             f"(default {RECOVERY_MAX_DOWNLOAD_MB}).")
    args = parser.parse_args()

    if args.model:
        WHISPER_MODEL = args.model
    BEAM_SIZE                  = max(1, args.beam_size)
    BATCH_SIZE                 = max(1, args.batch_size)
    MAX_CONCURRENT_EXTRACTIONS = max(1, args.workers)
    RETRY_CORRUPT              = bool(args.retry_corrupt)
    ENABLE_RECOVERY            = not args.no_recovery
    RECOVERY_MAX_DOWNLOAD_MB   = max(1, args.recovery_max_mb)

    # ── Preflight checks ──────────────────────────────────────────────────────
    if WHISPER_BACKEND == "none":
        console.print(
            "[red bold]No Whisper backend found.[/red bold]\n\n"
            "Install one of:\n"
            "  [cyan]pip install faster-whisper[/cyan]"
            "   ← recommended for Linux / Windows (NVIDIA CUDA or CPU)\n"
            "  [cyan]pip install mlx-whisper[/cyan]"
            "      ← Apple Silicon only\n"
            "  [cyan]pip install openai-whisper[/cyan]"
            "   ← CPU-only fallback (slow)\n"
        )
        sys.exit(1)

    for pkg, hint in [
        ("internetarchive", "pip install 'internetarchive[speedups]'"),
    ]:
        try:
            __import__(pkg)
        except ImportError:
            console.print(f"[red]Missing:[/red] {pkg}  →  {hint}")
            sys.exit(1)

    if not shutil.which("ffmpeg"):
        console.print(
            "[red]ffmpeg not found.[/red]\n"
            "  macOS  :  [cyan]brew install ffmpeg[/cyan]\n"
            "  Arch   :  [cyan]sudo pacman -S ffmpeg[/cyan]\n"
            "  Debian :  [cyan]sudo apt install ffmpeg[/cyan]\n"
            "  Windows:  https://ffmpeg.org/download.html  (add ffmpeg to PATH)"
        )
        sys.exit(1)

    # ── Show active backend ────────────────────────────────────────────────────
    _backend_colour = {
        "mlx_whisper":    "magenta",
        "faster_whisper": "green",
        "openai_whisper": "yellow",
    }.get(WHISPER_BACKEND, "white")

    console.print(
        f"[bold]Whisper backend :[/bold] "
        f"[{_backend_colour}]{WHISPER_BACKEND}[/{_backend_colour}]  "
        f"[dim](model: {WHISPER_MODEL})[/dim]"
    )

    # ── Eager model load (faster-whisper only) ─────────────────────────────────
    # Load BEFORE the pipeline starts so:
    #   • "Loading model" appears exactly once, in the right place.
    #   • The transcription worker never blocks waiting for a model load.
    #   • CUDA OOM during load is caught here with a clear error message.
    if WHISPER_BACKEND == "faster_whisper":
        dev, ct = _determine_compute_device()
        console.print(
            f"[bold]Compute device  :[/bold] "
            f"[{'green' if dev == 'cuda' else 'yellow'}]{dev}[/]  "
            f"[dim]compute_type={ct}[/dim]"
        )
        try:
            _faster_whisper_holder.load(dev, ct)
        except Exception as exc:
            # GPU load failed - try CPU int8 before giving up.
            log.warning("Model load on %s failed (%s); retrying on CPU int8.", dev, exc)
            _status("⚠️", "yellow",
                    f"Model load failed on {dev} ({exc}); retrying on CPU int8…")
            try:
                _faster_whisper_holder.load("cpu", "int8")
            except Exception as exc2:
                console.print(f"[red bold]Could not load faster-whisper model: {exc2}[/red bold]")
                sys.exit(1)

    console.print(
        f"[bold]Platform        :[/bold] "
        f"[dim]{platform.system()} {platform.machine()} "
        f"Python {sys.version.split()[0]}[/dim]\n"
    )

    # ── Recovery tooling ───────────────────────────────────────────────────────
    if ENABLE_RECOVERY:
        _detect_recovery_tools()
        bits = []
        bits.append(f"[green]full-download[/green]")
        bits.append(f"[green]untrunc:{_HAVE_UNTRUNC}[/green]" if _HAVE_UNTRUNC
                    else "[dim]untrunc:off[/dim]")
        bits.append("[green]yt-dlp[/green]" if _HAVE_YTDLP else "[dim]yt-dlp:off[/dim]")
        console.print(f"[bold]Recovery        :[/bold] " + "  ".join(bits))
        if not _HAVE_UNTRUNC:
            console.print("  [dim]untrunc disabled - install it or build the "
                          "'untrunc-local' docker image to recover truncated MP4s.[/dim]")

    store      = ProgressStore(PROGRESS_FILE)
    video_list = get_video_files()
    if not video_list:
        console.print("[yellow]No video files found.  Exiting.[/yellow]")
        return

    # Share the list/store with the recovery helpers (untrunc reference picking).
    _RECOVERY_VIDEOS = video_list
    _RECOVERY_STORE  = store

    verify_transcripts(video_list, store, deep=args.verify_hashes)

    all_entries   = store.all_entries()
    state_counts: dict = {}
    for fi in video_list:
        s = all_entries.get(fi["name"], {}).get("status", S_PENDING)
        state_counts[s] = state_counts.get(s, 0) + 1
    log.info("State summary: %s", state_counts)

    total        = len(video_list)
    already_done = state_counts.get(S_DONE, 0)
    n_corrupt    = state_counts.get(S_CORRUPT, 0)
    n_no_audio   = state_counts.get(S_NO_AUDIO, 0)
    n_derivs     = sum(1 for f in video_list if f.get("derivative_url"))
    # 1:1 target = every video that *can* have a transcript (i.e. has audio).
    transcribable = total - n_no_audio

    # No-audio sources are never processed; corrupt ones only with --retry-corrupt.
    skip_states = {S_DONE, S_NO_AUDIO} if RETRY_CORRUPT else {S_DONE, S_NO_AUDIO, S_CORRUPT}
    work_list = [
        fi for fi in video_list
        if all_entries.get(fi["name"], {}).get("status", S_PENDING) not in skip_states
    ]
    work_total = len(work_list)

    console.print(
        f"\n[bold]TempleOS Archive Transcriber[/bold]  "
        f"[dim](cross-platform edition)[/dim]\n"
        f"  [dim]{total} videos  •  {transcribable} with audio  •  "
        f"{already_done} transcribed  •  {work_total} to process  •  "
        f"{n_corrupt} unrecoverable{' (retrying)' if RETRY_CORRUPT and n_corrupt else ''}  •  "
        f"{n_no_audio} no-audio (skipped)  •  "
        f"{state_counts.get(S_ERROR, 0)} errors[/dim]\n"
        f"  [dim]No video files will be written to disk.[/dim]\n"
    )

    if work_total == 0:
        _write_unrecoverable_report(video_list, store)
        gap = transcribable - already_done
        if gap <= 0:
            console.print(
                f"[bold green]1:1 complete - all {transcribable} videos with audio "
                f"are transcribed.[/bold green]"
                + (f"  [dim]({n_no_audio} have no audio; "
                   f"{n_corrupt} unrecoverable - see "
                   f"{UNRECOVERABLE_REPORT}.)[/dim]" if (n_no_audio or n_corrupt) else "")
            )
        else:
            console.print(
                f"[bold yellow]{gap} video(s) with audio still lack a transcript "
                f"(unrecoverable).[/bold yellow]  [dim]See {UNRECOVERABLE_REPORT}; "
                f"re-attempt with[/dim] [cyan]--retry-corrupt[/cyan][dim].[/dim]"
            )
        return

    # ── Three-stage pipeline ───────────────────────────────────────────────────
    #
    #   [extractor pool]  →  ex_queue  →  [transcription worker]
    #                                              │
    #                                        results_queue
    #                                              │
    #                                    [main thread - progress bar]

    ex_queue:      queue.Queue = queue.Queue(maxsize=MAX_CONCURRENT_EXTRACTIONS + 2)
    results_queue: queue.Queue = queue.Queue()

    def _extractor_pool() -> None:
        def _task(file_info):
            log.info("extracting  %s", file_info["name"])
            res = extraction_stage(file_info, store)
            res["name"] = file_info["name"]
            ex_queue.put(res)

        try:
            with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_EXTRACTIONS) as pool:
                list(pool.map(_task, work_list))
        except Exception as exc:                       # noqa: BLE001
            log.error("extractor pool crashed: %s", exc, exc_info=True)
        finally:
            ex_queue.put(_SENTINEL)                     # always unblock the worker

    extractor_thread = threading.Thread(target=_extractor_pool, daemon=True,
                                        name="extractor-pool")
    extractor_thread.start()

    def _transcription_worker() -> None:
        try:
            while True:
                res = ex_queue.get()
                if res is _SENTINEL:
                    break
                try:
                    info         = process_stage(res, store)
                    info["name"] = res["name"]
                except Exception as exc:               # noqa: BLE001
                    log.error("transcription worker error on %s: %s",
                              res.get("name", "?"), exc, exc_info=True)
                    info = {"status": S_ERROR, "name": res.get("name", "?")}
                results_queue.put(info)
        finally:
            results_queue.put(_SENTINEL)               # always unblock the main loop

    transcription_thread = threading.Thread(target=_transcription_worker,
                                            daemon=True, name="transcriber")
    transcription_thread.start()

    run_done = run_errors = run_corrupt = run_no_audio = 0

    # Sources skipped this run (already-resolved corrupt / no-audio) won't flow
    # through the pipeline, so seed them into the bar or it can't reach 100%.
    bar_initial = already_done + n_no_audio + (0 if RETRY_CORRUPT else n_corrupt)

    with tqdm(total=total, initial=bar_initial, unit="file",
              desc="Overall", dynamic_ncols=True, colour="cyan") as pbar:
        while True:
            info = results_queue.get()
            if info is _SENTINEL:
                break

            name   = info["name"]
            short  = Path(name).name
            status = info["status"]
            seen   = already_done + run_done + run_errors + run_corrupt + run_no_audio

            pbar.set_postfix_str(short[:40], refresh=False)
            pbar.update(1)

            if status == S_DONE:
                run_done += 1
                counter  = already_done + run_done
                parts    = [f"✅ [{counter}/{total}] {short}"]
                if info.get("video_duration"):
                    parts.append(f"dur={_fmt_duration(info['video_duration'])}")
                if info.get("elapsed_extract", 0) > 0:
                    parts.append(f"ex={_fmt_elapsed(info['elapsed_extract'])}")
                if info.get("elapsed_transcribe", 0) > 0:
                    parts.append(f"tr={_fmt_elapsed(info['elapsed_transcribe'])}")
                parts.append(f"total={_fmt_elapsed(info['elapsed_total'])}")
                _status("✅", "green", "  ".join(parts))
            elif status == S_NO_AUDIO:
                run_no_audio += 1
                _status("🔇", "blue", f"[{seen + 1}/{total}] {short}  - no audio track")
            elif status == S_CORRUPT:
                run_corrupt += 1
                _status("🚫", "magenta",
                        f"[{seen + 1}/{total}] {short}  - unrecoverable source")
            else:
                run_errors += 1
                _status("❌", "red", f"[{seen + 1}/{total}] {short}")

            log.info("status=%-14s done=%d err=%d corrupt=%d no_audio=%d file=%s",
                     status, run_done, run_errors, run_corrupt, run_no_audio, name)

    extractor_thread.join()
    transcription_thread.join()
    store.flush(force=True)

    rep_corrupt, rep_no_audio = _write_unrecoverable_report(video_list, store)
    transcribed_total   = already_done + run_done
    transcribable_final = total - rep_no_audio       # recompute: run may add no-audio
    gap = transcribable_final - transcribed_total    # videos-with-audio still missing

    console.print()
    console.rule("[bold green]Run complete[/bold green]")
    console.print(f"  [bold]Newly transcribed    :[/bold] {run_done}")
    console.print(f"  [dim]Already done (skipped):[/dim] {already_done}")
    console.print(f"  [red]Errors (transient)   :[/red] {run_errors}")
    console.print(f"  [magenta]Unrecoverable sources:[/magenta] {rep_corrupt}")
    console.print(f"  [blue]No-audio (skipped)   :[/blue] {rep_no_audio}")
    console.print(
        f"  [bold]Transcript coverage  :[/bold] {transcribed_total}/{transcribable_final} "
        f"videos with audio  "
        + ("[green](1:1 ✓)[/green]" if gap <= 0
           else f"[yellow]({gap} unrecoverable)[/yellow]")
    )
    if rep_corrupt or rep_no_audio:
        console.print(f"  [dim]Unrecoverable list:[/dim] {UNRECOVERABLE_REPORT}  "
                      f"[dim](archive + youtube links)[/dim]")
    if rep_corrupt and not RETRY_CORRUPT:
        console.print("  [dim]→ re-attempt recovery with [/dim][cyan]--retry-corrupt[/cyan]")
    console.print(f"  [dim]Error log  :[/dim] {ERROR_LOG_FILE}")
    console.print(f"  [dim]Transcripts:[/dim] {TRANSCRIPT_DIR.resolve()}")
    console.print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted - progress saved; rerun to resume.[/yellow]")
        sys.exit(130)
