"""Local/public HTTP backend for ``analysis.run_replay``.

Standard library only (no new dependencies). One analysis at a time; every job is an
independent directory under ``--jobs-root``. The browser never chooses paths, scripts or
checkpoints: it may only upload one replay and read the job's own public report.

Run locally::

    .\\.venv\\Scripts\\python.exe -m server.app --team-dir <team-project>

Run the isolated public demo (loopback only; expose it through a tunnel)::

    .\\.venv\\Scripts\\python.exe -m server.app --public-demo --port 8766 \
        --jobs-root "runs\\public-jobs" --lock-file "runs\\backend-jobs\\.active.lock" \
        --team-dir <team-project>

Endpoints::

    GET  /api/support
    POST /api/jobs                     (multipart/form-data, one file field)
    GET  /api/jobs/<job_id>
    GET  /api/jobs/<job_id>/report
    GET  /api/jobs/<job_id>/preview?player=<player>&cutoff=<seconds>
    GET  /api/jobs/<job_id>/preview/<preview>.png
    GET  /jobs/<job_id>/report/<player>/{review.json,gradcam_1.png}
    GET  /ui/<file>                    (existing report viewer, served read-only)
"""

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, unquote, urlsplit

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.run_replay import (  # noqa: E402
    Failure, clean_env, inspect_replay, resolve_checkpoint, resolve_team_dir)
from analysis.gradcam_context import (  # noqa: E402
    enrich_explanation, load_minimap_codes, minimap_frame_png, observation_times,
    render_minimap_frames, sampling_metadata)
from config import SC2_PATH  # noqa: E402
from data_processing import processor  # noqa: E402
from server import multipart  # noqa: E402
from server.store import (  # noqa: E402
    JOB_ID_PATTERN, JobNotFound, JobStore, ActiveSlot, now, PLAYER_PATTERN,
    PUBLIC_IMAGE_PATTERN, atomic_write_json)
from tools import prepare_replay_dependencies as deps  # noqa: E402

ENGINE_LABEL = SimpleNamespace(version=processor.REPLAY_VERSION,
                              build=processor.ENGINE_VERSION.build_version,
                              data_version=processor.ENGINE_VERSION.data_version)

ANALYSIS_DEFAULTS = {"frames": 50, "step": 60.0, "draws": 1, "seed": 2026, "device": "cpu"}
INTERRUPTED_MESSAGE = ("The backend stopped while this analysis was running. Partial files are "
                       "kept; re-submit the replay to run it again.")
UI_ROOT = PROJECT_ROOT / "ui"
UI_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8", ".png": "image/png", ".svg": "image/svg+xml",
            ".json": "application/json; charset=utf-8"}
PUBLIC_FILES_BY_NAME = {"review.json": "application/json; charset=utf-8",
                        "gradcam_1.png": "image/png"}
FOCUS_LABELS = "0 = focus player loses, 1 = focus player wins"
TERRAN_LABELS = "0 = Protoss wins, 1 = Terran wins"
HEATMAP_ID_PATTERN = re.compile(r"^cam-player\d+-\d+s-[0-9a-f]{10}$")
PREVIEW_NAME_PATTERN = re.compile(r"^prev-player\d+-\d+s-[0-9a-f]{8}\.png$")
PREVIEW_MEMORY_ENTRIES = 6
FRAME_NAME_PATTERN = re.compile(r"^frame-(?P<player>player\d+)-(?P<index>\d{1,7})\.png$")
FRAME_MEMORY_ENTRIES = 1           # only the POV being browsed is kept in memory
FRAME_CACHE_MAX_FRAMES = 12000     # ~196 MB of uint8 codes; longer replays are refused
HEATMAP_REQUEST_LIMIT = 4096
HEATMAP_TIMEOUT_SECONDS = 1800
TARGET_TIMEOUT_SECONDS = 900
EPSILON = 1e-6


def public_content_type(name):
    """Return the content type for an exported report file, or None."""
    if name in PUBLIC_FILES_BY_NAME:
        return PUBLIC_FILES_BY_NAME[name]
    if PUBLIC_IMAGE_PATTERN.match(name):
        return "image/png"
    return None
MAX_ENVELOPE_SLACK = 64 * 1024
MAX_DRAIN_BYTES = 64 * 1024 * 1024
LOG = logging.getLogger("sc2-backend")


@dataclass
class Config:
    sc2_path: Path
    team_dir: Path
    checkpoint: Path
    jobs_root: Path
    allow_download_map: bool = False
    max_upload_bytes: int = 20 * 1024 * 1024
    python: str = sys.executable
    engine_timeout: int = 300
    public_mode: bool = False
    uploads_per_hour: int = 0
    rate_window_seconds: int = 3600
    job_timeout: int = 3600
    max_jobs_bytes: int = 0
    lock_path: Path | None = None


class UploadRateLimiter:
    """Small in-memory per-client limiter for public uploads.

    Cloudflare Quick Tunnel forwards a client identity in ``CF-Connecting-IP``. This is
    deliberately simple: the demo has one process and no queue, so an in-memory sliding
    window is enough to stop accidental or trivial repeated submissions.
    """

    def __init__(self, limit, window_seconds):
        self.limit = max(0, int(limit))
        self.window = max(1, int(window_seconds))
        self._events = {}
        self._lock = threading.Lock()

    def _prune(self, key, current):
        events = self._events.get(key)
        if not events:
            return []
        while events and current - events[0] >= self.window:
            events.popleft()
        return events

    def can_attempt(self, key):
        if self.limit <= 0:
            return True
        current = time.monotonic()
        with self._lock:
            events = self._prune(key, current)
            if events:
                self._events[key] = events
            else:
                self._events.pop(key, None)
            return len(events) < self.limit

    def allow(self, key):
        if self.limit <= 0:
            return True
        current = time.monotonic()
        with self._lock:
            events = self._prune(key, current)
            if len(events) >= self.limit:
                self._events[key] = events
                return False
            events.append(current)
            self._events[key] = events
            return True


def support_payload(config):
    """The single source of truth the frontend will read instead of hardcoding rules."""
    public_demo = {
        "enabled": bool(config.public_mode),
        "notice": ("Course demo. Analysis runs on this server and may take several minutes. "
                   "Only one heavy task runs at a time; there is no queue. Model estimates "
                   "and Grad-CAM explanations retain their existing limitations."),
        "report_access": ("Anyone who has a report URL can view that report. The URL is a "
                          "bearer link; do not share it if you do not want the report viewed."),
    }
    return {
        "format_version": 1,
        "service": "sc2-replay-backend",
        "public_demo": public_demo,
        "engine": {
            "game_version": ENGINE_LABEL.version,
            "base_build": ENGINE_LABEL.build,
            "data_version": ENGINE_LABEL.data_version,
            "note": "The backend always launches this installed build; it never substitutes "
                    "another version.",
        },
        "replay_requirements": {
            "game_version": ENGINE_LABEL.version,
            "players": 2,
            "player_type": "human participants (Participant)",
            "matchup": ["Terran", "Protoss"],
            "results": "one player wins and the other loses",
            "rejected": ["other game versions", "non 1v1", "mirror or other matchups",
                         "custom/observer-only recordings", "damaged archives"],
        },
        "upload": {
            "field": "file",
            "content_type": "multipart/form-data",
            "max_bytes": config.max_upload_bytes,
            "max_mib": round(config.max_upload_bytes / (1024 * 1024), 3),
            "enforced_on": "bytes actually received, not only Content-Length",
            "content_validation": "MPQ archive plus replay.gamemetadata.json plus the same "
                                  "scope rules the analysis uses",
            "filename": "stored under a server-generated name; the client filename is "
                        "display-only",
            "rate_limit": ("disabled" if not config.public_mode or config.uploads_per_hour <= 0
                           else f"{config.uploads_per_hour} uploads per "
                                f"{config.rate_window_seconds} s per client"),
        },
        "analysis": {
            "device": ANALYSIS_DEFAULTS["device"],
            "frames": ANALYSIS_DEFAULTS["frames"],
            "step_seconds": ANALYSIS_DEFAULTS["step"],
            "draws": ANALYSIS_DEFAULTS["draws"],
            "seed": ANALYSIS_DEFAULTS["seed"],
            "povs": 2,
            "static_explanations_per_pov": 1,
            "video": False,
            "training": False,
        },
        "heatmaps": {
            "on_demand": True,
            "custom_times": True,
            "background_candidates_per_pov": 3,
            "queueing": True,
            "povs": 2,
            "max_concurrent": 1,
            "states": ["queued", "running", "completed", "failed"],
            "progress_percent": None,
            "inputs": ["player", "cutoff_seconds"],
            "note": "Generated from the job's saved POV data and original checkpoint; "
                    "the analysis job stays completed. Opening a POV prepares at most three "
                    "candidate-interval explanations in the background; explicit requests "
                    "are queued ahead of remaining background work.",
        },
        "jobs": {
            "max_concurrent": 1,
            "queueing": False,
            "busy_status": 409,
            "suggested_poll_seconds": 3,
            "progress_percent": None,
            "timeout_seconds": config.job_timeout,
            "note": "Stages are real; there is no fabricated percentage.",
        },
        "map_cache": {
            "allow_download_map": config.allow_download_map,
            "note": "Only this replay's referenced archives are fetched from the official "
                    "depot, verified by content hash; no map sets or game packages.",
        },
        "limits": {
            "max_upload_bytes": config.max_upload_bytes,
            "single_active_job": True,
            "cors": "disabled (same-origin paths only)",
            "public_mode": bool(config.public_mode),
            "max_jobs_bytes": config.max_jobs_bytes if config.public_mode else None,
            "job_timeout_seconds": config.job_timeout,
            "uploads_per_hour": (config.uploads_per_hour if config.public_mode else None),
            "rate_window_seconds": (config.rate_window_seconds if config.public_mode else None),
        },
    }


def inspect_upload(path):
    """Validate upload content with the same rules the analysis uses."""
    try:
        return inspect_replay(path, lambda message: LOG.debug(message))
    except Failure as error:
        status = 400 if error.code == "invalid_replay" else 422
        raise multipart.UploadError(error.code, error.message, status)


def kill_process_tree(pid, log):
    """End a child and every descendant (Windows ``taskkill /T``), without a shell."""
    if os.name != "nt":
        return
    try:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=60, check=False)
    except Exception as error:  # noqa: BLE001 - cleanup must not raise
        log.warning("taskkill for pid %s failed: %s", pid, error)


class KillOnCloseJob:
    """Windows job object with KILL_ON_JOB_CLOSE.

    The analysis child spawns ``data_processing.processor``, a process-pool worker and the
    SC2 engine. Terminating only the direct child would leave those running, so the whole
    tree is assigned to a job: closing the handle (orderly shutdown, or the backend being
    killed) takes every descendant with it.
    """

    def __init__(self, log):
        self.log = log
        self._handle = None

    def assign(self, process):
        if os.name != "nt":
            return
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

            class Basic(ctypes.Structure):
                _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                            ("PerJobUserTimeLimit", ctypes.c_int64),
                            ("LimitFlags", wintypes.DWORD),
                            ("MinimumWorkingSetSize", ctypes.c_size_t),
                            ("MaximumWorkingSetSize", ctypes.c_size_t),
                            ("ActiveProcessLimit", wintypes.DWORD),
                            ("Affinity", ctypes.c_size_t),
                            ("PriorityClass", wintypes.DWORD),
                            ("SchedulingClass", wintypes.DWORD)]

            class IoCounters(ctypes.Structure):
                _fields_ = [("ReadOperationCount", ctypes.c_uint64),
                            ("WriteOperationCount", ctypes.c_uint64),
                            ("OtherOperationCount", ctypes.c_uint64),
                            ("ReadTransferCount", ctypes.c_uint64),
                            ("WriteTransferCount", ctypes.c_uint64),
                            ("OtherTransferCount", ctypes.c_uint64)]

            class Extended(ctypes.Structure):
                _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", IoCounters),
                            ("ProcessMemoryLimit", ctypes.c_size_t),
                            ("JobMemoryLimit", ctypes.c_size_t),
                            ("PeakProcessMemoryUsed", ctypes.c_size_t),
                            ("PeakJobMemoryUsed", ctypes.c_size_t)]

            handle = kernel32.CreateJobObjectW(None, None)
            if not handle:
                raise ctypes.WinError(ctypes.get_last_error())
            info = Extended()
            info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not kernel32.SetInformationJobObject(handle, 9, ctypes.byref(info),
                                                    ctypes.sizeof(info)):
                kernel32.CloseHandle(handle)
                raise ctypes.WinError(ctypes.get_last_error())
            if not kernel32.AssignProcessToJobObject(handle,
                                                     wintypes.HANDLE(int(process._handle))):
                kernel32.CloseHandle(handle)
                raise ctypes.WinError(ctypes.get_last_error())
            self._handle = handle
            self.log.info("analysis child and descendants assigned to a kill-on-close job")
        except Exception as error:  # noqa: BLE001 - explicit taskkill still applies
            self.log.warning("could not create a kill-on-close job: %s", error)

    def close(self):
        if self._handle is None or os.name != "nt":
            return
        try:
            import ctypes
            ctypes.WinDLL("kernel32").CloseHandle(ctypes.c_void_p(self._handle))
        except Exception as error:  # noqa: BLE001 - best effort
            self.log.warning("could not close the job handle: %s", error)
        self._handle = None


class JobManager:
    """Run one ``analysis.run_replay`` child at a time and mirror its real status."""

    def __init__(self, config, store, slot, log):
        self.config = config
        self.store = store
        self.slot = slot
        self.log = log
        self._mutex = threading.Lock()
        self._process = None
        self._thread = None
        self._active_job = None
        self._tree = None
        self._stopping = False

    @property
    def active_job_id(self):
        return self._active_job

    def start(self, job_id):
        with self._mutex:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("an analysis thread is already running")
            self._active_job = job_id
            self._thread = threading.Thread(target=self._run, args=(job_id,),
                                            name=f"analysis-{job_id}", daemon=True)
            self._thread.start()

    def _run(self, job_id):
        try:
            self._execute(job_id)
        except Exception as error:  # noqa: BLE001 - never leave a job running forever
            self.log.exception("job %s crashed", job_id)
            try:
                self.store.update(job_id, state="failed", stage="failed",
                                  error_code="internal_error",
                                  error_message=f"{type(error).__name__}: {error}",
                                  finished_at=now())
            except Exception:  # noqa: BLE001 - the record may be gone
                self.log.exception("could not record the crash of %s", job_id)
        finally:
            with self._mutex:
                self._process = None
                self._active_job = None
            if self._tree is not None:
                self._tree.close()
                self._tree = None
            self.slot.release()

    def _execute(self, job_id):
        job_dir = self.store.job_dir(job_id)
        run_dir = job_dir / "run"
        command = [self.config.python, "-m", "analysis.run_replay",
                   "--replay", str(self.store.upload_path(job_id)),
                   "--output", str(run_dir),
                   "--sc2-path", str(self.config.sc2_path),
                   "--team-dir", str(self.config.team_dir),
                   "--checkpoint", str(self.config.checkpoint),
                   "--frames", str(ANALYSIS_DEFAULTS["frames"]),
                   "--step", str(ANALYSIS_DEFAULTS["step"]),
                   "--draws", str(ANALYSIS_DEFAULTS["draws"]),
                   "--seed", str(ANALYSIS_DEFAULTS["seed"]),
                   "--device", ANALYSIS_DEFAULTS["device"],
                   "--engine-timeout", str(self.config.engine_timeout)]
        if self.config.allow_download_map:
            command.append("--allow-download-map")
        log_path = job_dir / "logs" / "run_replay.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write("$ " + subprocess.list2cmdline(command) + "\n")
            log_file.flush()
            process = subprocess.Popen(command, cwd=str(PROJECT_ROOT), stdout=log_file,
                                       stderr=subprocess.STDOUT, shell=False, env=clean_env())
            with self._mutex:
                self._process = process
            self._tree = KillOnCloseJob(self.log)
            self._tree.assign(process)
            self.store.update(job_id, state="running", stage="starting", started_at=now())
            self.log.info("job %s started (pid %s)", job_id, process.pid)
            started = time.monotonic()
            timed_out = False
            storage_exceeded = False
            last_cap_check = started
            last = (None, None)
            while process.poll() is None:
                now_monotonic = time.monotonic()
                if (self.config.job_timeout > 0
                        and now_monotonic - started > self.config.job_timeout):
                    timed_out = True
                    self.log.warning("job %s exceeded %s s; terminating its process tree",
                                     job_id, self.config.job_timeout)
                    kill_process_tree(process.pid, self.log)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    break
                if (self.config.max_jobs_bytes > 0
                        and now_monotonic - last_cap_check >= 5):
                    last_cap_check = now_monotonic
                    if self.store.size_bytes() > self.config.max_jobs_bytes:
                        storage_exceeded = True
                        self.log.warning("job %s exceeded the public storage cap; terminating",
                                         job_id)
                        kill_process_tree(process.pid, self.log)
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        break
                time.sleep(1.0)
                status = self.store.run_status(job_id)
                if not status:
                    continue
                current = (status.get("stage"), status.get("pov"))
                if current != last:
                    last = current
                    self.store.update(job_id, stage=current[0], pov=current[1],
                                      run_state=status.get("state"))
            exit_code = process.returncode
            with self._mutex:
                self._process = None
            if timed_out or storage_exceeded:
                status = self.store.run_status(job_id)
                code = "storage_full" if storage_exceeded else "job_timeout"
                message = ("Analysis exceeded the public storage cap. The process tree was "
                           "stopped; existing and partial files were kept."
                           if storage_exceeded else
                           f"Analysis exceeded the {self.config.job_timeout} s limit. The "
                           "process tree was stopped; partial files were kept.")
                self.store.update(
                    job_id, state="failed", stage="failed", error_code=code,
                    error_message=message, finished_at=now(), exit_code=exit_code,
                    run_state=status.get("state"))
                self.log.warning("job %s stopped (%s)", job_id, code)
                return
        self._finalize(job_id, exit_code)

    def _finalize(self, job_id, exit_code):
        """Success requires exit code, run state and every report artefact to agree."""
        record = self.store.read(job_id)
        if record.get("state") == "failed" and record.get("error_code") == "interrupted":
            return
        if self._stopping:
            self.store.update(job_id, state="failed", stage="interrupted",
                              error_code="interrupted", error_message=INTERRUPTED_MESSAGE,
                              finished_at=now(), run_state=self.store.run_status(job_id).get("state"),
                              exit_code=exit_code)
            return
        run_dir = self.store.run_dir(job_id)
        status = self.store.run_status(job_id)
        run_state = status.get("state")
        problems = []
        index = None
        index_path = run_dir / "report_index.json"
        if exit_code != 0:
            problems.append(f"analysis exited with code {exit_code}")
        if run_state != "completed":
            problems.append(f"analysis state is {run_state!r}")
        if index_path.is_file():
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
            except ValueError as error:
                problems.append(f"report index is unreadable: {error}")
        else:
            problems.append("report_index.json is missing")
        if index is not None:
            povs = index.get("povs") or []
            if len(povs) != 2:
                problems.append(f"report index lists {len(povs)} POV(s)")
            for pov in povs:
                for relative in (pov.get("review"),
                                 (pov.get("explanation") or {}).get("image")):
                    if not relative or not (run_dir / relative).is_file():
                        problems.append(f"missing report artefact {relative!r}")
        if problems:
            code = status.get("error_code")
            if not code:
                code = {3: "busy", 2: "output_exists"}.get(exit_code, "run_failed")
            message = status.get("error_message")
            if not message and code == "busy":
                message = ("Another analysis (a manual CLI run) holds the lock. "
                           "Retry this job later.")
            message = message or "; ".join(problems)
            self.store.update(job_id, state="failed", stage="failed", error_code=code,
                              error_message=str(message)[:800], exit_code=exit_code,
                              finished_at=now(), run_state=run_state)
            self.log.warning("job %s failed (%s): %s", job_id, code, message)
            return
        try:
            report = self.store.export_public(job_id)
        except Exception as error:  # noqa: BLE001 - publishing must not fake success
            self.log.exception("job %s could not publish its report", job_id)
            self.store.update(job_id, state="failed", stage="failed",
                              error_code="report_export_failed",
                              error_message=("The analysis finished but the backend could not "
                                             f"publish the report: {type(error).__name__}: {error}"),
                              exit_code=exit_code, finished_at=now(), run_state=run_state)
            return
        report["index_url"] = f"/api/jobs/{job_id}/report"
        self.store.update(job_id, state="completed", stage="completed", pov=None,
                          error_code=None, error_message=None, exit_code=exit_code,
                          finished_at=now(), run_state=run_state, report=report,
                          engine=(index or {}).get("engine"))
        self.log.info("job %s completed", job_id)

    def stop_all(self):
        """Terminate only our own child tree, then mark the job interrupted."""
        self._stopping = True
        with self._mutex:
            process = self._process
            job_id = self._active_job
        if process is not None and process.poll() is None:
            self.log.info("terminating analysis tree pid=%s", process.pid)
            kill_process_tree(process.pid, self.log)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if self._tree is not None:
            self._tree.close()
            self._tree = None
        if job_id:
            try:
                status = self.store.run_status(job_id)
                self.store.update(job_id, state="failed", stage="interrupted",
                                  error_code="interrupted",
                                  error_message=INTERRUPTED_MESSAGE,
                                  finished_at=now(), run_state=status.get("state"))
            except Exception:  # noqa: BLE001 - best effort during shutdown
                self.log.exception("could not mark %s interrupted", job_id)
        self.slot.release()


class HeatmapRunError(Exception):
    """Internal generation failure with a stable error code."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


class HeatmapRequestError(Exception):
    """Public request failure for on-demand heatmaps."""

    def __init__(self, status, code, message):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def heatmap_target_meaning(label_definition, target):
    if label_definition == FOCUS_LABELS:
        return "focus player wins" if target == 1 else "focus player loses"
    if label_definition == TERRAN_LABELS:
        return "Terran wins" if target == 1 else "Protoss wins"
    raise HeatmapRequestError(500, "unsupported_label", "The report has an unsupported label definition.")


def explanation_for_cutoff(review, cutoff, target=None):
    """Return an existing explanation for this exact cutoff, if any."""
    for explanation in reversed(review.get("explanations") or []):
        if abs(float(explanation.get("cutoff_seconds", float("nan"))) - float(cutoff)) > EPSILON:
            continue
        if target is not None and int(explanation.get("target_class", -1)) != int(target):
            continue
        image = explanation.get("image")
        if isinstance(image, str) and Path(image).name == image:
            return explanation
    return None


def heatmap_status_payload(job_id, status, cached=False):
    """Browser-safe heatmap task status; never exposes disk paths."""
    payload = {
        "format_version": 1,
        "heatmap_id": status.get("heatmap_id"),
        "job_id": job_id,
        "state": status.get("state"),
        "pov": status.get("pov"),
        "cutoff_seconds": status.get("cutoff_seconds"),
        "target_class": status.get("target_class"),
        "target_meaning": status.get("target_meaning"),
        "probability": status.get("probability"),
        "timeline_probability": status.get("timeline_probability"),
        "probability_match": status.get("probability_match"),
        "probability_note": status.get("probability_note"),
        "probability_source": status.get("probability_source"),
        "custom_cutoff": bool(status.get("custom_cutoff")),
        "requested_by": status.get("requested_by"),
        "request_hash": status.get("request_hash"),
        "cache_key": status.get("cache_key"),
        "native_temporal_bins": status.get("native_temporal_bins"),
        "cached": bool(cached or status.get("cached")),
        "created_at": status.get("created_at"),
        "updated_at": status.get("updated_at"),
        "finished_at": status.get("finished_at"),
        "error_code": status.get("error_code"),
        "error_message": status.get("error_message"),
        "analysis_state": "completed",
    }
    image_name = status.get("image_name")
    if status.get("state") == "completed" and isinstance(image_name, str):
        payload["image_url"] = f"/jobs/{job_id}/report/{status.get('pov')}/{image_name}"
    payload["status_url"] = f"/api/jobs/{job_id}/heatmaps/{status.get('heatmap_id')}"
    return payload


class HeatmapManager:
    """One shared Grad-CAM queue for background preparation and explicit requests.

    A single worker generates one explanation at a time under the shared heavy-task
    slot. Explicit browser requests are dequeued before remaining background work,
    and the slot is released between explanations. Both paths share the same cache,
    status files and review.json update logic.
    """

    def __init__(self, config, store, slot, log):
        self.config = config
        self.store = store
        self.slot = slot
        self.log = log
        self._mutex = threading.Lock()
        self._condition = threading.Condition(self._mutex)
        self._explicit = deque()
        self._background = deque()
        self._queued = set()
        self._active_key = None
        self._active_spec = None
        self._process = None
        self._tree = None
        self._stopping = False
        self._prepared = {}
        self._sha_cache = {}
        self._time_cache = OrderedDict()
        self.recover()
        self._worker = threading.Thread(target=self._work, name="heatmap-worker", daemon=True)
        self._worker.start()

    def recover(self):
        """Mark tasks left queued/running by a previous process as interrupted."""
        for path in self.store.root.glob("job-*/run/heatmaps/*/status.json"):
            try:
                status = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if status.get("state") in ("queued", "running"):
                status.update(state="failed", error_code="interrupted",
                              error_message="The backend stopped before this heatmap finished.",
                              updated_at=now(), finished_at=now())
                atomic_write_json(path, status)
                self.log.warning("marked heatmap %s interrupted", path.parent.name)

    # ---- context and cache identity ---------------------------------------

    def _checkpoint_sha(self, path):
        try:
            stat = path.stat()
        except OSError:
            return None
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        with self._mutex:
            cached = self._sha_cache.get(key)
        if cached is not None:
            return cached
        value = deps.sha256_file(path)
        with self._mutex:
            self._sha_cache[key] = value
        return value

    def _times(self, sample_dir):
        key = str(sample_dir)
        with self._mutex:
            cached = self._time_cache.get(key)
            if cached is not None:
                self._time_cache.move_to_end(key)
                return cached
        seconds, loops = observation_times(sample_dir)
        value = ([float(item) for item in seconds], [int(item) for item in loops])
        with self._mutex:
            self._time_cache[key] = value
            while len(self._time_cache) > 4:
                self._time_cache.popitem(last=False)
        return value

    def _base(self, job_id, player):
        record = self.store.read(job_id)
        if record.get("state") != "completed":
            raise HeatmapRequestError(409, "not_ready",
                                      "The analysis job is not completed; its report is not available.")
        if not re.match(r"^player\d+$", str(player)):
            raise HeatmapRequestError(400, "invalid_player", "Choose one of the job's player POVs.")
        run_dir = self.store.run_dir(job_id).resolve()
        index_path = run_dir / "report_index.json"
        if not index_path.is_file():
            raise HeatmapRequestError(409, "report_missing", "The report index is missing.")
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise HeatmapRequestError(500, "report_invalid", "The report index could not be read.") from error
        pov = next((item for item in index.get("povs") or [] if item.get("player") == player), None)
        if pov is None:
            raise HeatmapRequestError(404, "unknown_pov", f"The report has no {player} POV.")
        params = index.get("params") or {}
        review_path = (run_dir / str(pov.get("review"))).resolve()
        if run_dir not in review_path.parents or not review_path.is_file():
            raise HeatmapRequestError(500, "report_invalid", "The report path is outside the job.")
        try:
            review = json.loads(review_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise HeatmapRequestError(500, "report_invalid", "The report could not be read.") from error
        sample_dir = (run_dir / "processed" / "both" / str(pov.get("sample_id"))).resolve()
        if run_dir not in sample_dir.parents or not sample_dir.is_dir():
            raise HeatmapRequestError(409, "processed_data_missing",
                                      "The job's saved POV observations are not available.")
        checkpoint = Path(params.get("checkpoint") or self.config.checkpoint)
        if not checkpoint.is_file():
            raise HeatmapRequestError(409, "checkpoint_missing", "The original checkpoint is not available.")
        checkpoint_sha = (index.get("checkpoint") or {}).get("sha256")
        if not checkpoint_sha or self._checkpoint_sha(checkpoint) != checkpoint_sha:
            raise HeatmapRequestError(409, "checkpoint_changed",
                                      "The checkpoint no longer matches the one recorded by this report.")
        team_dir = Path(params.get("team_dir") or self.config.team_dir)
        if not (team_dir / "viz" / "gradcam.py").is_file():
            raise HeatmapRequestError(409, "team_code_missing", "The team Grad-CAM code is not available.")
        frames = int(params.get("frames") or review.get("frames") or 0)
        seed = int(params.get("seed") if params.get("seed") is not None else review.get("seed") or 0)
        draws = int(params.get("draws") or review.get("draws_per_point") or 1)
        device = str(params.get("device") or "cpu")
        if frames < 2:
            raise HeatmapRequestError(500, "report_invalid", "The report has no valid frame count.")
        return {
            "job_id": job_id, "record": record, "run_dir": run_dir, "index": index,
            "pov": pov, "player": player, "review_path": review_path, "review": review,
            "sample_dir": sample_dir, "checkpoint": checkpoint,
            "checkpoint_sha": checkpoint_sha, "team_dir": team_dir,
            "frames": frames, "seed": seed, "draws": draws, "device": device,
            "label_definition": review.get("label_definition"),
            "probability_key": review.get("probability_key") or "terran_win_probability",
        }

    def _request_spec(self, job_id, player, raw_cutoff):
        """Validate a cutoff and build the request identity (target resolved later)."""
        spec = self._base(job_id, player)
        try:
            cutoff = float(raw_cutoff)
        except (TypeError, ValueError) as error:
            raise HeatmapRequestError(400, "invalid_cutoff",
                                      "cutoff_seconds must be a finite number.") from error
        if not np.isfinite(cutoff) or cutoff <= 0:
            raise HeatmapRequestError(400, "invalid_cutoff", "cutoff_seconds must be a finite number.")
        review = spec["review"]
        points = review.get("points") or []
        point = next((item for item in points
                      if abs(float(item.get("cutoff_seconds")) - cutoff) <= EPSILON), None)
        timeline_probability = None
        target = None
        if point is not None:
            try:
                timeline_probability = float(point[spec["probability_key"]])
            except (KeyError, TypeError, ValueError):
                raise HeatmapRequestError(500, "report_invalid", "The report point has no probability.")
            target = 1 if timeline_probability >= 0.5 else 0
        else:
            seconds, _loops = self._times(spec["sample_dir"])
            if spec["frames"] >= len(seconds):
                raise HeatmapRequestError(
                    409, "insufficient_history",
                    "The recording has fewer saved observations than the report's frame count.")
            minimum = seconds[spec["frames"] - 1]
            maximum = seconds[-2]
            if cutoff < minimum - EPSILON:
                raise HeatmapRequestError(
                    422, "insufficient_history",
                    "This time has fewer than %d saved observations of history; the earliest "
                    "supported time is %.3f s." % (spec["frames"], minimum))
            if cutoff > maximum + EPSILON:
                raise HeatmapRequestError(
                    422, "after_last_observation",
                    "The terminal observation cannot be used for Grad-CAM; the latest "
                    "supported time is %.3f s." % maximum)
        identity = {
            "job_id": job_id,
            "algorithm": "viz.gradcam.block3.v1",
            "pov": player,
            "cutoff_seconds": repr(float(cutoff)),
            "checkpoint_sha256": spec["checkpoint_sha"],
            "frames": spec["frames"],
            "seed": spec["seed"],
            "draws": spec["draws"],
            "device": spec["device"],
            "label_definition": spec["label_definition"],
        }
        request_hash = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:10]
        heatmap_id = f"cam-{player}-{int(round(cutoff))}s-{request_hash}"
        image_name = f"gradcam_{player}_{int(round(cutoff))}s_{request_hash}.png"
        storage = spec["run_dir"] / "heatmaps" / heatmap_id
        spec.update({
            "cutoff": float(cutoff), "point": point,
            "timeline_probability": timeline_probability,
            "custom_cutoff": point is None,
            "target": target,
            "target_meaning": (heatmap_target_meaning(spec["label_definition"], target)
                               if target is not None else None),
            "request_hash": request_hash, "heatmap_id": heatmap_id,
            "image_name": image_name, "storage": storage,
            "status_path": storage / "status.json", "viz_dir": storage / "viz",
            "log_path": self.store.job_dir(job_id) / "logs" / f"heatmap-{heatmap_id}.log",
        })
        return spec

    def _read_status(self, spec):
        path = spec["status_path"]
        if not path.is_file():
            return None
        try:
            status = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if status.get("heatmap_id") != spec["heatmap_id"]:
            return None
        return status

    def _write_status(self, spec, status):
        status = dict(status)
        status["format_version"] = 1
        status["heatmap_id"] = spec["heatmap_id"]
        status["job_id"] = spec["job_id"]
        status["pov"] = spec["player"]
        status["cutoff_seconds"] = spec["cutoff"]
        status["target_class"] = spec.get("target")
        status["target_meaning"] = spec.get("target_meaning")
        status["timeline_probability"] = spec.get("timeline_probability")
        status["image_name"] = spec["image_name"]
        status["custom_cutoff"] = bool(spec["custom_cutoff"])
        status["request_hash"] = spec["request_hash"]
        status["requested_by"] = status.get("requested_by") or spec.get("source")
        status["updated_at"] = now()
        atomic_write_json(spec["status_path"], status)
        return status

    @staticmethod
    def _explanation_probability(explanation):
        generation = explanation.get("generation") if isinstance(explanation, dict) else None
        probability = (generation or {}).get("probability")
        if probability is not None:
            try:
                return float(probability)
            except (TypeError, ValueError):
                pass
        classes = explanation.get("class_probabilities") if isinstance(explanation, dict) else None
        if isinstance(classes, (list, tuple)) and len(classes) == 2:
            try:
                return float(classes[1])
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _policy_target(probability):
        if probability is None or not np.isfinite(probability):
            return None
        return 1 if float(probability) >= 0.5 else 0

    def _cached_payload(self, spec, export=True):
        """Return an exact-cutoff cached explanation for this request, if valid."""
        review = json.loads(spec["review_path"].read_text(encoding="utf-8"))
        if spec["point"] is not None:
            explanation = explanation_for_cutoff(review, spec["cutoff"], spec["target"])
            probability = self._explanation_probability(explanation) if explanation else None
            if probability is None:
                probability = spec["timeline_probability"]
            timeline_probability = spec["timeline_probability"]
            generation = (explanation or {}).get("generation") or {}
            probability_match = generation.get(
                "probability_match",
                abs(float(probability) - timeline_probability) <= EPSILON)
            probability_source = generation.get("probability_source", "saved_timeline")
            probability_note = generation.get("probability_note")
        else:
            explanation = explanation_for_cutoff(review, spec["cutoff"])
            probability = self._explanation_probability(explanation) if explanation else None
            if explanation is not None and self._policy_target(probability) != int(
                    explanation.get("target_class", -1)):
                explanation = None
            timeline_probability = None
            probability_match = None
            generation = (explanation or {}).get("generation") or {}
            probability_source = generation.get("probability_source", "inference")
            probability_note = generation.get(
                "probability_note",
                "Custom cutoff: the probability is this inference's output; no saved curve "
                "probability was copied or interpolated.")
        if explanation is None:
            return None
        image_name = explanation.get("image")
        if not isinstance(image_name, str) or Path(image_name).name != image_name:
            return None
        private_image = spec["review_path"].parent / image_name
        if not private_image.is_file():
            return None
        if export:
            public_image = self.store.public_dir(spec["job_id"]) / spec["player"] / image_name
            if not public_image.is_file():
                try:
                    self.store.export_public(spec["job_id"])
                except Exception as error:  # noqa: BLE001 - report remains intact
                    self.log.exception("could not export cached heatmap %s", spec["heatmap_id"])
                    raise HeatmapRequestError(
                        500, "report_export_failed",
                        f"The cached heatmap exists but could not be published: {error}") from error
        target = int(explanation.get("target_class", spec.get("target") or 0))
        status = {
            "state": "completed",
            "heatmap_id": generation.get("heatmap_id") or spec["heatmap_id"],
            "pov": spec["player"],
            "cutoff_seconds": spec["cutoff"],
            "target_class": target,
            "target_meaning": explanation.get("target_meaning") or heatmap_target_meaning(
                spec["label_definition"], target),
            "probability": probability,
            "timeline_probability": timeline_probability,
            "probability_match": probability_match,
            "probability_note": probability_note,
            "probability_source": probability_source,
            "custom_cutoff": bool(spec["custom_cutoff"]),
            "requested_by": generation.get("requested_by") or spec.get("source"),
            "request_hash": generation.get("request_hash") or spec["request_hash"],
            "cache_key": generation.get("cache_key"),
            "native_temporal_bins": generation.get("native_temporal_bins"),
            "image_name": image_name,
            "cached": True,
            "created_at": generation.get("created_at"),
            "updated_at": now(),
            "finished_at": generation.get("finished_at"),
            "error_code": None,
            "error_message": None,
        }
        return heatmap_status_payload(spec["job_id"], status, cached=True)

    @staticmethod
    def _key(spec):
        return (spec["job_id"], spec["player"], repr(float(spec["cutoff"])))

    def request(self, job_id, player, cutoff, source="explicit"):
        """Return a cached explanation or queue one Grad-CAM generation."""
        spec = self._request_spec(job_id, player, cutoff)
        spec["source"] = source
        cached = self._cached_payload(spec)
        if cached:
            return cached, 200
        key = self._key(spec)
        with self._condition:
            if self._active_key == key:
                status = self._read_status(spec) or {"state": "running"}
                return heatmap_status_payload(job_id, status), 202
            if key in self._queued:
                status = self._read_status(spec) or {"state": "queued"}
                return heatmap_status_payload(job_id, status), 202
            if source == "explicit":
                kept = deque()
                for item in self._explicit:
                    if item["job_id"] == job_id and item["player"] == player:
                        self._queued.discard(self._key(item))
                        try:
                            self._write_status(item, {
                                "state": "failed", "cached": False,
                                "error_code": "superseded",
                                "error_message": "A newer explicit heatmap request replaced this queued one.",
                                "finished_at": now(),
                            })
                        except Exception:  # noqa: BLE001 - the new request still proceeds
                            self.log.exception("could not mark heatmap %s superseded", item["heatmap_id"])
                    else:
                        kept.append(item)
                self._explicit = kept
                self._explicit.append(spec)
            else:
                self._background.append(spec)
            self._queued.add(key)
            status = self._write_status(spec, {
                "state": "queued",
                "requested_by": source,
                "probability": None,
                "probability_match": None,
                "probability_note": None,
                "created_at": now(),
                "started_at": None,
                "finished_at": None,
                "error_code": None,
                "error_message": None,
                "cached": False,
            })
            self._condition.notify_all()
        return heatmap_status_payload(job_id, status), 202

    # ---- single worker -----------------------------------------------------

    def _acquire_slot(self, spec):
        while not self._stopping:
            if self.slot.try_acquire():
                return True
            with self._condition:
                if spec.get("source") == "background" and self._explicit:
                    self._background.appendleft(spec)
                    self._queued.add(self._key(spec))
                    return False
                self._condition.wait(0.5)
        return False

    def _work(self):
        while True:
            with self._condition:
                while (not self._stopping and not self._explicit and not self._background):
                    self._condition.wait()
                if self._stopping:
                    return
                spec = self._explicit.popleft() if self._explicit else self._background.popleft()
                self._queued.discard(self._key(spec))
                self._active_key = self._key(spec)
                self._active_spec = spec
            acquired = False
            try:
                acquired = self._acquire_slot(spec)
                if not acquired:
                    continue
                try:
                    self._execute(spec)
                except Exception as error:  # noqa: BLE001 - every failure is recorded
                    self._record_failure(spec, error)
            finally:
                if acquired:
                    self.slot.release()
                with self._condition:
                    tree = self._tree
                    self._tree = None
                    self._process = None
                    self._active_key = None
                    self._active_spec = None
                    self._condition.notify_all()
                if tree is not None:
                    tree.close()

    def _record_failure(self, spec, error):
        code = getattr(error, "code", "heatmap_failed")
        message = getattr(error, "message", str(error))
        if self._stopping:
            code = "interrupted"
            message = "The backend stopped before this heatmap finished."
        self.log.warning("heatmap %s failed (%s): %s", spec["heatmap_id"], code, message)
        try:
            self._write_status(spec, {
                "state": "failed",
                "started_at": (self._read_status(spec) or {}).get("started_at"),
                "finished_at": now(),
                "error_code": code,
                "error_message": str(message)[:800],
                "cached": False,
            })
        except Exception:  # noqa: BLE001 - keep the original failure
            self.log.exception("could not record heatmap failure %s", spec["heatmap_id"])

    # ---- subprocesses ------------------------------------------------------

    def _run_process(self, spec, command, timeout, cwd):
        spec["log_path"].parent.mkdir(parents=True, exist_ok=True)
        with spec["log_path"].open("a", encoding="utf-8") as log_file:
            log_file.write("$ " + subprocess.list2cmdline(command) + "\n")
            log_file.flush()
            process = subprocess.Popen(command, cwd=str(cwd), stdout=log_file,
                                       stderr=subprocess.STDOUT, shell=False, env=clean_env())
            tree = KillOnCloseJob(self.log)
            with self._mutex:
                self._process = process
                self._tree = tree
            tree.assign(process)
            deadline = time.monotonic() + timeout
            while process.poll() is None:
                if time.monotonic() > deadline:
                    kill_process_tree(process.pid, self.log)
                    tree.close()
                    with self._mutex:
                        if self._tree is tree:
                            self._tree = None
                        self._process = None
                    raise HeatmapRunError("heatmap_timeout",
                                          f"Grad-CAM work exceeded {timeout} s.")
                time.sleep(0.5)
            code = process.returncode
            tree.close()
            with self._mutex:
                if self._tree is tree:
                    self._tree = None
                self._process = None
            return code

    def _infer_target(self, spec):
        """Run one forward pass to select the target class for a custom cutoff."""
        output = spec["storage"] / "target.json"
        try:
            output.unlink()
        except OSError:
            pass
        command = [
            self.config.python, "-m", "analysis.explain_target", str(spec["sample_dir"]),
            str(spec["checkpoint"]), "--device", spec["device"],
            "--frames", str(spec["frames"]), "--seed", str(spec["seed"]),
            "--cutoff", repr(float(spec["cutoff"])), "--output", str(output),
        ]
        code = self._run_process(spec, command, TARGET_TIMEOUT_SECONDS, PROJECT_ROOT)
        if code != 0 or not output.is_file():
            raise HeatmapRunError(
                "target_inference_failed",
                f"target selection exited {code}; see {spec['log_path'].name}")
        try:
            payload = json.loads(output.read_text(encoding="utf-8"))
            probability = float(payload["probability"])
            target = int(payload["target_class"])
            classes = [float(value) for value in payload["class_probabilities"]]
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise HeatmapRunError("target_inference_failed",
                                  f"target selection wrote invalid metadata: {error}") from error
        if (not np.isfinite(probability) or not 0.0 <= probability <= 1.0
                or len(classes) != 2 or target not in (0, 1)
                or target != (1 if probability >= 0.5 else 0)):
            raise HeatmapRunError("target_inference_failed",
                                  "target selection returned an inconsistent probability/target")
        if int(payload.get("frames", -1)) != spec["frames"] or int(payload.get("seed", -1)) != spec["seed"]:
            raise HeatmapRunError("sampling_mismatch",
                                  "target selection used different sampling settings")
        if abs(float(payload.get("cutoff_seconds", -1)) - spec["cutoff"]) > EPSILON:
            raise HeatmapRunError("cutoff_mismatch",
                                  "target selection used a different cutoff")
        if payload.get("label_definition") != spec["label_definition"]:
            raise HeatmapRunError("label_mismatch",
                                  "target selection used a different label definition")
        return {"probability": probability, "class_probabilities": classes,
                "target_class": target}

    # ---- one generation ----------------------------------------------------

    def _execute(self, spec):
        spec["storage"].mkdir(parents=True, exist_ok=True)
        spec["viz_dir"].mkdir(parents=True, exist_ok=True)
        spec["log_path"].parent.mkdir(parents=True, exist_ok=True)
        if spec["custom_cutoff"]:
            self._write_status(spec, {
                "state": "running", "started_at": now(), "updated_at": now(),
                "error_code": None, "error_message": None,
                "probability": None, "probability_match": None, "probability_note": None,
            })
            inference = self._infer_target(spec)
            spec["target"] = inference["target_class"]
            spec["target_meaning"] = heatmap_target_meaning(
                spec["label_definition"], spec["target"])
            spec["inference_probability"] = inference["probability"]
            spec["inference_classes"] = inference["class_probabilities"]
        target = spec["target"]
        identity = {
            "job_id": spec["job_id"], "algorithm": "viz.gradcam.block3.v1",
            "pov": spec["player"],
            "cutoff_seconds": repr(float(spec["cutoff"])), "target_class": target,
            "checkpoint_sha256": spec["checkpoint_sha"], "frames": spec["frames"],
            "seed": spec["seed"], "draws": spec["draws"], "device": spec["device"],
            "label_definition": spec["label_definition"],
        }
        cache_key = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:10]
        spec["cache_key"] = cache_key
        status = self._read_status(spec) or {}
        status.update(state="running", started_at=status.get("started_at") or now(),
                      updated_at=now(), error_code=None, error_message=None,
                      probability=None, probability_match=None, probability_note=None,
                      probability_source=("saved_timeline" if spec["point"] is not None else "inference"),
                      cache_key=cache_key, cached=False)
        self._write_status(spec, status)
        self.log.info("heatmap %s running (%s @ %ss, target %s, %s)",
                      spec["heatmap_id"], spec["player"], spec["cutoff"], target,
                      "background" if spec.get("source") == "background" else "explicit")
        command = [
            self.config.python, "-m", "viz.gradcam", str(spec["sample_dir"]),
            str(spec["checkpoint"]), "--device", spec["device"],
            "--frames", str(spec["frames"]), "--seed", str(spec["seed"]),
            "--target", str(target), "--cutoff", repr(float(spec["cutoff"])),
            "--output", str(spec["viz_dir"]),
        ]
        exit_code = self._run_process(spec, command, HEATMAP_TIMEOUT_SECONDS, spec["team_dir"])
        if exit_code != 0:
            raise HeatmapRunError("heatmap_failed",
                                  f"viz.gradcam exited {exit_code}; see {spec['log_path'].name}")
        for name in ("snapshot.png", "data.npz", "method.json"):
            if not (spec["viz_dir"] / name).is_file():
                raise HeatmapRunError("heatmap_incomplete", f"viz.gradcam did not write {name}")
        method = json.loads((spec["viz_dir"] / "method.json").read_text(encoding="utf-8"))
        if int(method.get("target_class", -1)) != target:
            raise HeatmapRunError("target_mismatch",
                                  f"Generated target {method.get('target_class')!r} != {target}")
        if method.get("label_definition") != spec["label_definition"]:
            raise HeatmapRunError("label_mismatch",
                                  f"Generated label definition {method.get('label_definition')!r} "
                                  f"!= {spec['label_definition']!r}")
        if int(method.get("frames", -1)) != spec["frames"] or int(method.get("seed", -1)) != spec["seed"]:
            raise HeatmapRunError("sampling_mismatch",
                                  "Generated frames/seed do not match the report's sampling settings.")
        checkpoint_recorded = method.get("checkpoint")
        if checkpoint_recorded and Path(checkpoint_recorded).resolve() != spec["checkpoint"].resolve():
            raise HeatmapRunError("checkpoint_mismatch",
                                  "Generated heatmap used a different checkpoint.")
        with np.load(spec["viz_dir"] / "data.npz", allow_pickle=False) as archive:
            cutoffs = archive["cutoff_seconds"].astype(float)
            probabilities = archive["probabilities"].astype(float)
            native_bins = int(archive["raw_heatmap"].shape[1]) if "raw_heatmap" in archive.files else None
        if len(cutoffs) != 1 or abs(float(cutoffs[0]) - spec["cutoff"]) > EPSILON:
            raise HeatmapRunError("cutoff_mismatch",
                                  f"Generated cutoff {cutoffs.tolist()} != {spec['cutoff']}")
        if probabilities.shape != (1, 2):
            raise HeatmapRunError("heatmap_incomplete", f"probabilities shape {probabilities.shape}")
        explained = float(probabilities[0][1])
        timeline_probability = spec["timeline_probability"]
        if spec["point"] is not None:
            if int(spec["draws"]) == 1:
                if abs(explained - timeline_probability) > EPSILON:
                    raise HeatmapRunError(
                        "probability_mismatch",
                        f"Generated P={explained:.9f} but the saved timeline has "
                        f"{timeline_probability:.9f} at {spec['cutoff']} s. The report and "
                        "timeline probability were not modified.")
                probability_match = True
                probability_note = "Generated probability equals the saved timeline probability (draws=1)."
            else:
                draws = [float(value) for value in spec["point"].get("probability_draws") or []]
                probability_match = bool(draws and min(draws) - 1e-9 <= explained <= max(draws) + 1e-9)
                if not probability_match:
                    raise HeatmapRunError(
                        "probability_mismatch",
                        f"Generated P={explained:.9f} is outside the saved draw range {draws}.")
                probability_note = "Generated probability lies inside the saved draw range."
            probability_source = "saved_timeline"
        else:
            if abs(explained - spec["inference_probability"]) > 1e-5:
                raise HeatmapRunError(
                    "probability_mismatch",
                    f"Custom-time Grad-CAM P={explained:.9f} differs from the target-selection "
                    f"inference P={spec['inference_probability']:.9f}.")
            if (explained >= 0.5) != (spec["target"] == 1):
                raise HeatmapRunError(
                    "target_policy_mismatch",
                    "The custom-time inference crossed the target-policy boundary between passes.")
            probability_match = None
            probability_source = "inference"
            probability_note = ("Custom cutoff: the probability is this inference's output; no saved "
                                "curve probability was copied or interpolated.")

        image_path = spec["review_path"].parent / spec["image_name"]
        shutil.copyfile(spec["viz_dir"] / "snapshot.png", image_path)
        explanation = {
            "cutoff_seconds": spec["cutoff"],
            "displayed_frame_seconds": None,
            "target_class": spec["target"],
            "target_meaning": spec["target_meaning"],
            "layer": method.get("layer"),
            "score": method.get("score"),
            "image": spec["image_name"],
            "arrays": None,
            "class_probabilities": [float(value) for value in probabilities[0]],
            "masking_check": {
                "available": False,
                "reason": "On-demand viz.gradcam export does not include a masking diagnostic",
            },
            "generated_on_demand": True,
            "generation": {
                "heatmap_id": spec["heatmap_id"],
                "request_hash": spec["request_hash"],
                "cache_key": cache_key,
                "checkpoint_sha256": spec["checkpoint_sha"],
                "frames": spec["frames"],
                "seed": spec["seed"],
                "draws": spec["draws"],
                "device": spec["device"],
                "label_definition": spec["label_definition"],
                "probability": explained,
                "timeline_probability": timeline_probability,
                "probability_match": probability_match,
                "probability_source": probability_source,
                "probability_note": probability_note,
                "custom_cutoff": bool(spec["custom_cutoff"]),
                "requested_by": spec.get("source") or "explicit",
                "target_policy": "class 1 if inference probability >= 0.5 else class 0",
                "native_temporal_bins": native_bins,
                "created_at": now(),
                "finished_at": now(),
            },
        }
        try:
            explanation.update(enrich_explanation(
                spec["sample_dir"], spec["review_path"].parent,
                spec["cutoff"], spec["frames"], spec["seed"],
                spec["viz_dir"] / "data.npz", spec["player"],
                "recorded_at_export", f"_{cache_key}"))
        except Exception as error:  # noqa: BLE001 - keep the heatmap with unknown context
            self.log.warning("heatmap %s context unavailable: %s", spec["heatmap_id"], error)
            explanation["sampling"] = None
            explanation["context_error"] = f"{type(error).__name__}: {error}"
        self._update_review(spec, explanation)
        try:
            self.store.export_public(spec["job_id"])
        except Exception as error:  # noqa: BLE001 - retry can export the cached review
            raise HeatmapRunError("report_export_failed",
                                  f"Heatmap generated but the report could not be published: {error}") from error
        status = self._read_status(spec) or {}
        status.update(
            state="completed", probability=explained,
            probability_match=probability_match,
            probability_source=probability_source,
            probability_note=probability_note,
            class_probabilities=[float(value) for value in probabilities[0]],
            native_temporal_bins=native_bins,
            finished_at=now(), error_code=None, error_message=None, cached=False)
        self._write_status(spec, status)
        self.log.info("heatmap %s completed P=%.6f", spec["heatmap_id"], explained)

    def _update_review(self, spec, explanation):
        review_path = spec["review_path"]
        review = json.loads(review_path.read_text(encoding="utf-8"))
        backup = review_path.with_name(review_path.stem + ".before-on-demand-heatmaps.json")
        if not backup.exists():
            shutil.copyfile(review_path, backup)
        explanations = []
        for existing in review.get("explanations") or []:
            is_same_cutoff = abs(float(existing.get("cutoff_seconds", float("nan")))
                                 - spec["cutoff"]) <= EPSILON
            if is_same_cutoff and existing.get("generated_on_demand"):
                continue
            explanations.append(existing)
        explanations.append(explanation)
        explanations.sort(key=lambda item: float(item.get("cutoff_seconds", 0)))
        review["explanations"] = explanations
        atomic_write_json(review_path, review)
        return review

    def stop_all(self):
        self._stopping = True
        with self._condition:
            self._condition.notify_all()
            process = self._process
            tree = self._tree
            spec = self._active_spec
        if process is not None and process.poll() is None:
            self.log.info("terminating heatmap process pid=%s", process.pid)
            kill_process_tree(process.pid, self.log)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if tree is not None:
            tree.close()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=15)
        if spec is not None:
            current = self._read_status(spec)
            if current is None or current.get("state") not in ("completed", "failed"):
                try:
                    self._write_status(spec, {
                        "state": "failed",
                        "finished_at": now(),
                        "error_code": "interrupted",
                        "error_message": "The backend stopped before this heatmap finished.",
                        "cached": False,
                    })
                except Exception:  # noqa: BLE001 - best effort during shutdown
                    self.log.exception("could not mark heatmap interrupted")

    # ---- status and background preparation ---------------------------------

    def status(self, job_id, heatmap_id):
        self.store.read(job_id)
        if not HEATMAP_ID_PATTERN.match(str(heatmap_id)):
            raise HeatmapRequestError(404, "heatmap_not_found", "Unknown heatmap id.")
        path = self.store.run_dir(job_id) / "heatmaps" / heatmap_id / "status.json"
        if path.is_file():
            status = json.loads(path.read_text(encoding="utf-8"))
            if status.get("job_id") != job_id:
                raise HeatmapRequestError(404, "heatmap_not_found", "Unknown heatmap id.")
            if status.get("state") == "completed" and status.get("image_name"):
                public = self.store.public_dir(job_id) / status["pov"] / status["image_name"]
                if not public.is_file():
                    try:
                        self.store.export_public(job_id)
                    except Exception:  # noqa: BLE001 - status remains readable
                        self.log.exception("could not re-export heatmap %s", heatmap_id)
            return heatmap_status_payload(job_id, status, cached=bool(status.get("cached")))
        match = re.match(r"^cam-(player\d+)-(\d+)s-[0-9a-f]{10}$", str(heatmap_id))
        if match:
            try:
                spec = self._request_spec(job_id, match.group(1), float(match.group(2)))
                cached = self._cached_payload(spec)
            except HeatmapRequestError:
                cached = None
            if cached:
                return cached
        raise HeatmapRequestError(404, "heatmap_not_found", "Unknown heatmap id.")

    @staticmethod
    def _candidate_cutoffs(review):
        """Top three candidate intervals by absolute net change, deduplicated."""
        intervals = review.get("pivotal_intervals") or []
        ranked = sorted(intervals,
                        key=lambda item: abs(float(item.get("net_change") or 0.0)),
                        reverse=True)
        candidates = []
        seen = set()
        for interval in ranked:
            value = interval.get("max_step_cutoff_seconds")
            if value is None:
                value = interval.get("peak_cutoff_seconds")
            try:
                cutoff = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(cutoff):
                continue
            key = repr(cutoff)
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                "cutoff_seconds": cutoff,
                "start_seconds": interval.get("start_seconds"),
                "end_seconds": interval.get("end_seconds"),
                "net_change": interval.get("net_change"),
            })
            if len(candidates) >= 3:
                break
        if not candidates:
            # Preserve the analysis's existing single-explanation fallback: explain the
            # later side of the largest adjacent change, or the only point when one exists.
            if review.get("explanations"):
                return []
            points = sorted(review.get("points") or [],
                            key=lambda item: float(item.get("cutoff_seconds") or 0.0))
            if points:
                index = 0
                if len(points) > 1:
                    probability_key = (review.get("probability_key")
                                       or "terran_win_probability")
                    deltas = [
                        abs(float(points[position + 1].get(probability_key) or 0.0)
                            - float(points[position].get(probability_key) or 0.0))
                        for position in range(len(points) - 1)
                    ]
                    index = max(range(len(deltas)), key=deltas.__getitem__) + 1
                candidates.append({
                    "cutoff_seconds": float(points[index]["cutoff_seconds"]),
                    "start_seconds": None, "end_seconds": None, "net_change": None,
                })
        return candidates

    def _prepare_entries(self, job_id, player, candidates, enqueue):
        entries = []
        for candidate in candidates:
            cutoff = candidate["cutoff_seconds"]
            entry = {"cutoff_seconds": cutoff,
                     "start_seconds": candidate.get("start_seconds"),
                     "end_seconds": candidate.get("end_seconds"),
                     "net_change": candidate.get("net_change")}
            try:
                spec = self._request_spec(job_id, player, cutoff)
                entry.update({"heatmap_id": spec["heatmap_id"],
                              "status_url": f"/api/jobs/{job_id}/heatmaps/{spec['heatmap_id']}"})
                if enqueue:
                    payload, _status = self.request(job_id, player, cutoff, source="background")
                    if payload.get("cached") or payload.get("state") == "completed":
                        entry["state"] = "cached"
                    else:
                        entry["state"] = payload.get("state") or "queued"
                    entry["error_code"] = payload.get("error_code")
                    entry["error_message"] = payload.get("error_message")
                else:
                    cached = self._cached_payload(spec, export=False)
                    status = self._read_status(spec)
                    if cached:
                        entry["state"] = "cached"
                    elif status is not None:
                        entry["state"] = status.get("state")
                        entry["error_code"] = status.get("error_code")
                        entry["error_message"] = status.get("error_message")
                    else:
                        entry["state"] = "not_prepared"
            except HeatmapRequestError as error:
                entry.update({"state": "unsupported", "error_code": error.code,
                              "error_message": error.message})
            entries.append(entry)
        return entries

    @staticmethod
    def _prepare_payload(job_id, player, entries, prepared):
        pending = sum(1 for item in entries if item.get("state") in ("queued", "running"))
        failed = sum(1 for item in entries if item.get("state") == "failed")
        completed = sum(1 for item in entries if item.get("state") in ("cached", "completed"))
        return {
            "format_version": 1,
            "job_id": job_id,
            "player": player,
            "analysis_state": "completed",
            "prepared": bool(prepared),
            "candidates": entries,
            "pending": pending,
            "failed": failed,
            "completed": completed,
        }

    def prepare(self, job_id, player):
        """Queue up to three candidate-interval explanations in the background."""
        base = self._base(job_id, player)
        candidates = self._candidate_cutoffs(base["review"])
        entries = self._prepare_entries(job_id, player, candidates, enqueue=True)
        self._prepared[(job_id, player)] = [dict(item) for item in candidates]
        return self._prepare_payload(job_id, player, entries, prepared=True)

    def prepare_status(self, job_id, player):
        """Read-only preparation status; never starts a generation itself."""
        base = self._base(job_id, player)
        candidates = self._prepared.get((job_id, player))
        if candidates is None:
            candidates = self._candidate_cutoffs(base["review"])
            entries = self._prepare_entries(job_id, player, candidates, enqueue=False)
            return self._prepare_payload(job_id, player, entries, prepared=False)
        entries = self._prepare_entries(job_id, player, candidates, enqueue=False)
        return self._prepare_payload(job_id, player, entries, prepared=True)


class GameViewManager:
    """Render one on-demand RGB context frame under the shared heavy-task slot."""

    def __init__(self, config, store, slot, log):
        self.config = config
        self.store = store
        self.slot = slot
        self.log = log
        self._mutex = threading.Lock()
        self._process = None
        self._tree = None
        self._stopping = False

    def _context(self, job_id, player, cutoff):
        record = self.store.read(job_id)
        if record.get("state") != "completed":
            raise HeatmapRequestError(409, "not_ready", "The analysis job is not completed.")
        if not re.match(r"^player\d+$", str(player)):
            raise HeatmapRequestError(400, "invalid_player", "Choose one of the job's POVs.")
        run_dir = self.store.run_dir(job_id)
        index = json.loads((run_dir / "report_index.json").read_text(encoding="utf-8"))
        pov = next((item for item in index.get("povs") or [] if item.get("player") == player), None)
        if pov is None:
            raise HeatmapRequestError(404, "unknown_pov", f"The report has no {player} POV.")
        review_path = run_dir / pov["review"]
        review = json.loads(review_path.read_text(encoding="utf-8"))
        explanation = explanation_for_cutoff(review, float(cutoff))
        if explanation is None:
            raise HeatmapRequestError(422, "no_explanation",
                                      "A heatmap explanation is required before rendering its game view.")
        frame = explanation.get("context_frame") if isinstance(explanation.get("context_frame"), dict) else {}
        sampling = explanation.get("sampling") if isinstance(explanation.get("sampling"), dict) else {}
        game_loop = frame.get("game_loop", sampling.get("last_game_loop"))
        seconds = frame.get("frame_seconds", sampling.get("last_frame_seconds"))
        if isinstance(game_loop, bool) or not isinstance(game_loop, (int, float)):
            raise HeatmapRequestError(409, "context_time_missing",
                                      "The explanation has no saved context game loop.")
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            seconds = None
        replay = self.store.upload_path(job_id)
        if not replay.is_file():
            raise HeatmapRequestError(409, "replay_missing", "The job's replay copy is not available.")
        build = index.get("engine", {}).get("base_build", 0)
        digest = hashlib.sha256(
            f"{job_id}|{player}|{int(game_loop)}|{build}".encode("utf-8")).hexdigest()[:8]
        view_id = f"rgb-{player}-{int(game_loop)}-{digest}"
        image_name = f"rgb_{player}_{int(game_loop)}_{digest}.png"
        storage = run_dir / "context_views" / view_id
        return {
            "job_id": job_id, "player": player, "pov": pov, "cutoff": float(cutoff),
            "explanation": explanation, "review_path": review_path, "review": review,
            "replay": replay, "game_loop": int(game_loop), "seconds": seconds,
            "source": frame.get("source") or sampling.get("source") or "unknown",
            "view_id": view_id, "image_name": image_name, "storage": storage,
            "output": storage / "game_view.png",
            "log_path": self.store.job_dir(job_id) / "logs" / f"game-view-{view_id}.log",
        }

    def _payload(self, context, game_view, cached):
        return {
            "format_version": 1,
            "state": "completed",
            "cached": bool(cached),
            "pov": context["player"],
            "cutoff_seconds": context["cutoff"],
            "label": "Re-rendered game view",
            "image_name": game_view.get("image"),
            "image_url": f"/jobs/{context['job_id']}/report/{context['player']}/{game_view.get('image')}",
            "requested_game_loop": game_view.get("requested_game_loop"),
            "actual_game_loop": game_view.get("actual_game_loop"),
            "actual_seconds": game_view.get("actual_seconds"),
            "source": game_view.get("source"),
            "note": "Re-rendered from the original replay and engine; not the model-input RGB.",
        }


    def _cached(self, context):
        game_view = context["explanation"].get("game_view")
        if not isinstance(game_view, dict):
            return None
        image_name = game_view.get("image")
        if not isinstance(image_name, str) or Path(image_name).name != image_name:
            return None
        private = context["review_path"].parent / image_name
        if not private.is_file():
            return None
        public = self.store.public_dir(context["job_id"]) / context["player"] / image_name
        if not public.is_file():
            self.store.export_public(context["job_id"])
        return self._payload(context, game_view, True)

    def render(self, job_id, player, cutoff):
        context = self._context(job_id, player, cutoff)
        cached = self._cached(context)
        if cached:
            return cached
        with self._mutex:
            if self._process is not None and self._process.poll() is None:
                raise HeatmapRequestError(409, "busy", "Another heavy task is running.")
            if not self.slot.try_acquire():
                raise HeatmapRequestError(409, "busy",
                                          "Another analysis or heatmap is running; retry after it finishes.")
            acquired = True
        tree = None
        try:
            context["storage"].mkdir(parents=True, exist_ok=True)
            context["log_path"].parent.mkdir(parents=True, exist_ok=True)
            command = [
                self.config.python, "-m", "analysis.render_game_view",
                "--replay", str(context["replay"]),
                "--output", str(context["output"]),
                "--player", context["player"].replace("player", ""),
                "--game-loop", str(context["game_loop"]),
            ]
            with context["log_path"].open("w", encoding="utf-8") as log_file:
                log_file.write("$ " + subprocess.list2cmdline(command) + "\n")
                log_file.flush()
                process = subprocess.Popen(command, cwd=str(PROJECT_ROOT), stdout=log_file,
                                           stderr=subprocess.STDOUT, shell=False, env=clean_env())
                with self._mutex:
                    self._process = process
                tree = KillOnCloseJob(self.log)
                with self._mutex:
                    self._tree = tree
                tree.assign(process)
                deadline = time.monotonic() + HEATMAP_TIMEOUT_SECONDS
                while process.poll() is None:
                    if self._stopping:
                        kill_process_tree(process.pid, self.log)
                        raise HeatmapRequestError(409, "interrupted",
                                                  "The backend stopped during game-view rendering.")
                    if time.monotonic() > deadline:
                        kill_process_tree(process.pid, self.log)
                        raise HeatmapRequestError(500, "game_view_timeout",
                                                  f"Game-view rendering exceeded {HEATMAP_TIMEOUT_SECONDS} s.")
                    time.sleep(0.5)
                exit_code = process.returncode
            if exit_code != 0:
                raise HeatmapRequestError(500, "game_view_failed",
                                          f"RGB renderer exited {exit_code}; see {context['log_path'].name}.")
            if not context["output"].is_file():
                raise HeatmapRequestError(500, "game_view_incomplete", "RGB renderer wrote no image.")
            metadata = json.loads(context["output"].with_suffix(".json").read_text(encoding="utf-8"))
            actual_loop = int(metadata["actual_game_loop"])
            if actual_loop != context["game_loop"]:
                raise HeatmapRequestError(
                    500, "game_view_time_mismatch",
                    f"Renderer reached loop {actual_loop}, expected {context['game_loop']}; no image attached.")
            image_path = context["review_path"].parent / context["image_name"]
            shutil.copyfile(context["output"], image_path)
            review = json.loads(context["review_path"].read_text(encoding="utf-8"))
            current = explanation_for_cutoff(review, context["cutoff"])
            if current is None:
                raise HeatmapRequestError(500, "review_changed", "The explanation disappeared during rendering.")
            backup = context["review_path"].with_name(
                context["review_path"].stem + ".before-game-view.json")
            if not backup.exists():
                shutil.copyfile(context["review_path"], backup)
            game_view = {
                "image": context["image_name"],
                "requested_game_loop": context["game_loop"],
                "actual_game_loop": actual_loop,
                "requested_seconds": context["seconds"],
                "actual_seconds": float(metadata.get(
                    "actual_seconds", actual_loop / processor.GAME_LOOPS_PER_SECOND)),
                "source": context["source"],
                "label": "Re-rendered game view",
                "note": "Rendered from the original replay and engine at the saved context game loop; "
                        "not the model-input RGB and not a peak-bin frame.",
            }
            current["game_view"] = game_view
            atomic_write_json(context["review_path"], review)
            self.store.export_public(context["job_id"])
            self.log.info("game view %s completed (loop %s)", context["image_name"], actual_loop)
            return self._payload(context, game_view, False)
        finally:
            with self._mutex:
                current_process = self._process
                current_tree = self._tree
                self._process = None
                self._tree = None
            if current_tree is not None:
                current_tree.close()
            elif tree is not None:
                tree.close()
            if acquired:
                self.slot.release()

    def stop_all(self):
        self._stopping = True
        with self._mutex:
            process = self._process
        if process is not None and process.poll() is None:
            self.log.info("terminating game-view process pid=%s", process.pid)
            kill_process_tree(process.pid, self.log)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        self.slot.release()


def job_payload(record):
    """Public job view: state, stage, times, error and (when ready) report links."""
    job_id = record["job_id"]
    payload = {
        "format_version": 1,
        "job_id": job_id,
        "state": record.get("state"),
        "stage": record.get("stage"),
        "pov": record.get("pov"),
        "run_state": record.get("run_state"),
        "created_at": record.get("created_at"),
        "queued_at": record.get("queued_at"),
        "started_at": record.get("started_at"),
        "updated_at": record.get("updated_at"),
        "finished_at": record.get("finished_at"),
        "error_code": record.get("error_code"),
        "error_message": record.get("error_message"),
        "original_name": record.get("original_name"),
        "upload": {"bytes": (record.get("upload") or {}).get("bytes"),
                   "sha256": (record.get("upload") or {}).get("sha256")},
        "replay": record.get("replay"),
        "config": record.get("config"),
        "status_url": f"/api/jobs/{job_id}",
        "report": None,
    }
    if record.get("state") == "completed":
        players = sorted(((record.get("report") or {}).get("players") or {}))
        payload["report"] = {
            "index_url": f"/api/jobs/{job_id}/report",
            "povs": [{"player": player,
                      "review_url": f"/jobs/{job_id}/report/{player}/review.json",
                      "image_url": f"/jobs/{job_id}/report/{player}/gradcam_1.png",
                      "ui_url": f"/ui/index.html?review=/jobs/{job_id}/report/{player}"}
                     for player in players],
        }
    return payload


class PreviewError(Exception):
    """Public request failure for a reference minimap preview."""

    def __init__(self, status, code, message):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


class PreviewManager:
    """Saved-observation minimaps for a completed POV.

    Two read-only views share the job's saved arrays and the same palette:

    * ``lookup``/``image`` rebuild the deterministic sampler result for a saved
      prediction cutoff and cache those PNGs on disk.
    * ``frames``/``lookup_frame``/``frame_image`` expose every saved observation
      from TimeInfo (not only prediction points). One combined uint8 code array
      per POV is kept in a one-entry memory cache, so dragging the time slider
      renders a single saved frame without decompressing the NPZ again.

    No model, Grad-CAM, SC2 or replay extraction runs here.
    """

    def __init__(self, store, log):
        self.store = store
        self.log = log
        self._mutex = threading.Lock()      # guards the caches and lock table
        self._key_locks = {}                # (job, player) -> Lock
        self._memory = OrderedDict()        # (job, player) -> cutoff entries
        self._frames = OrderedDict()        # (job, player) -> observation context
        self._read_lock = threading.Lock()  # one NPZ decompression at a time

    def _lock_for(self, key):
        with self._mutex:
            return self._key_locks.setdefault(key, threading.Lock())

    def _remember(self, key, entries):
        with self._mutex:
            self._memory[key] = entries
            self._memory.move_to_end(key)
            while len(self._memory) > PREVIEW_MEMORY_ENTRIES:
                self._memory.popitem(last=False)

    def _remember_frames(self, key, context):
        with self._mutex:
            self._frames[key] = context
            self._frames.move_to_end(key)
            while len(self._frames) > FRAME_MEMORY_ENTRIES:
                self._frames.popitem(last=False)

    def _context(self, job_id, player):
        record = self.store.read(job_id)
        if record.get("state") != "completed":
            raise PreviewError(409, "not_ready", f"Job is {record.get('state')!r}")
        players = (record.get("report") or {}).get("players") or {}
        if player not in players:
            raise PreviewError(404, "unknown_pov", f"The report has no {player} POV.")
        run_dir = self.store.run_dir(job_id).resolve()
        index_path = run_dir / "report_index.json"
        if not index_path.is_file():
            raise PreviewError(409, "report_missing", "The report index is missing.")
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise PreviewError(500, "report_invalid", "The report index could not be read.") from error
        pov = next((entry for entry in index.get("povs") or []
                    if entry.get("player") == player), None)
        if not pov:
            raise PreviewError(404, "unknown_pov", f"The report has no {player} POV.")
        review_path = (run_dir / str(pov.get("review"))).resolve()
        if run_dir not in review_path.parents:
            raise PreviewError(500, "report_invalid", "The report path is outside the job.")
        try:
            review = json.loads(review_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise PreviewError(500, "report_invalid", "The report could not be read.") from error
        frames = review.get("frames")
        seed = review.get("seed")
        if not isinstance(frames, int) or frames < 1 or not isinstance(seed, int):
            raise PreviewError(500, "report_invalid", "The report has no sampling settings.")
        sample_id = review.get("sample_id") or pov.get("sample_id")
        sample_dir = (run_dir / "processed" / "both" / str(sample_id or "")).resolve()
        if run_dir not in sample_dir.parents or not (sample_dir / "SpatialFeatures.npz").is_file():
            raise PreviewError(409, "processed_data_missing",
                               "The saved POV arrays are not available for this report.")
        points = []
        for point in review.get("points") or []:
            value = point.get("cutoff_seconds")
            if isinstance(value, (int, float)) and value == value:
                points.append(float(value))
        if not points:
            raise PreviewError(409, "report_missing", "The report has no prediction points.")
        return run_dir, sample_dir, frames, seed, points


    def _prepare(self, job_id, player):
        run_dir, sample_dir, frames, seed, points = self._context(job_id, player)
        output_dir = run_dir / "previews" / player
        output_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = hashlib.sha256(
            ("v2|%s|%d|%d|%s" % (sample_dir.name, frames, seed,
                                 ",".join("%.3f" % value for value in points))).encode("utf-8")
        ).hexdigest()[:8]
        meta_path = output_dir / ("previews-%s.json" % fingerprint)
        if meta_path.is_file():
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
                entries = payload.get("entries") or {}
                if entries and all((output_dir / entry["image"]).is_file()
                                   for entry in entries.values()):
                    return entries
            except (OSError, ValueError, KeyError):
                self.log.warning("rebuilding unreadable preview cache %s", meta_path.name)

        samplings = {}
        for cutoff in points:
            samplings[cutoff] = sampling_metadata(
                sample_dir, cutoff, frames, seed, "reconstructed_from_saved_observations")
        names = {}
        for cutoff, sampling in samplings.items():
            frame = int(sampling["last_frame_index"])
            names.setdefault(frame, "prev-%s-%ds-%s.png"
                             % (player, int(round(cutoff)), fingerprint))
        with self._read_lock:
            render_minimap_frames(sample_dir, list(names), [output_dir / name
                                                           for name in names.values()])
        entries = {}
        for cutoff, sampling in samplings.items():
            frame = int(sampling["last_frame_index"])
            entries[repr(float(cutoff))] = {
                "image": names[frame],
                "frame_index": frame,
                "frame_seconds": float(sampling["last_frame_seconds"]),
                "game_loop": int(sampling["last_game_loop"]),
                "frames": int(frames),
                "seed": int(seed),
                "count": int(sampling["count"]),
                "min_seconds": float(sampling["min_seconds"]),
                "max_seconds": float(sampling["max_seconds"]),
                "source": sampling["source"],
            }
        atomic_write_json(meta_path, {
            "job_id": job_id, "player": player, "frames": frames, "seed": seed,
            "sample": sample_dir.name, "entries": entries,
        })
        return entries

    def _entries(self, job_id, player):
        key = (job_id, player)
        with self._mutex:
            cached = self._memory.get(key)
        if cached is not None:
            return cached
        with self._lock_for(key):
            with self._mutex:
                cached = self._memory.get(key)
            if cached is not None:
                return cached
            try:
                entries = self._prepare(job_id, player)
            except PreviewError:
                raise
            except (OSError, ValueError) as error:
                self.log.exception("could not prepare previews for %s/%s", job_id, player)
                raise PreviewError(409, "preview_unavailable",
                                   f"Reference minimaps could not be rebuilt: {error}") from error
            self._remember(key, entries)
            return entries

    def lookup(self, job_id, player, raw_cutoff):
        """Metadata for one saved cutoff; the image is served by :meth:`image`."""
        if not PLAYER_PATTERN.match(str(player or "")):
            raise PreviewError(400, "invalid_player", "Choose one of the job's player POVs.")
        try:
            cutoff = float(raw_cutoff)
        except (TypeError, ValueError) as error:
            raise PreviewError(400, "invalid_cutoff",
                               "cutoff must be one of the report's saved prediction times.") from error
        if cutoff != cutoff or cutoff in (float("inf"), float("-inf")):
            raise PreviewError(400, "invalid_cutoff", "cutoff must be a finite number.")
        entries = self._entries(job_id, player)
        entry = None
        for key, candidate in entries.items():
            if abs(float(key) - cutoff) <= EPSILON:
                entry = candidate
                break
        if entry is None:
            raise PreviewError(422, "unknown_cutoff",
                               "cutoff must be one of the report's saved prediction times.")
        return {
            "cutoff_seconds": float(cutoff),
            "frame_seconds": entry["frame_seconds"],
            "game_loop": entry["game_loop"],
            "frame_index": entry["frame_index"],
            "source": entry["source"],
            "sampling": {
                "frames": entry["frames"], "seed": entry["seed"], "count": entry["count"],
                "min_seconds": entry["min_seconds"], "max_seconds": entry["max_seconds"],
            },
            "image_url": "/api/jobs/%s/preview/%s" % (job_id, entry["image"]),
        }

    def image(self, job_id, name):
        """Bytes of one cached preview image, restricted to this job's own cache."""
        if not PREVIEW_NAME_PATTERN.match(str(name or "")):
            raise PreviewError(404, "not_found", "Unknown preview image")
        player = name.split("-")[1]
        entries = self._entries(job_id, player)
        if name not in {entry["image"] for entry in entries.values()}:
            raise PreviewError(404, "not_found", "That preview is not part of this report")
        run_dir = self.store.run_dir(job_id).resolve()
        path = (run_dir / "previews" / player / name).resolve()
        if run_dir not in path.parents or not path.is_file():
            raise PreviewError(404, "not_found", "The preview image is not available")
        return path.read_bytes()


    # ---- continuous replay-minimap browsing --------------------------------

    def _frame_context(self, job_id, player):
        """TimeInfo timestamps plus one bounded code array for this POV."""
        key = (job_id, player)
        with self._mutex:
            cached = self._frames.get(key)
            if cached is not None:
                self._frames.move_to_end(key)
                return cached
        with self._lock_for(key):
            with self._mutex:
                cached = self._frames.get(key)
                if cached is not None:
                    self._frames.move_to_end(key)
                    return cached
            _run_dir, sample_dir, _frames, _seed, _points = self._context(job_id, player)
            try:
                seconds, loops = observation_times(sample_dir)
            except (OSError, ValueError) as error:
                raise PreviewError(409, "preview_unavailable",
                                   f"Saved observation times could not be read: {error}") from error
            if len(seconds) > FRAME_CACHE_MAX_FRAMES:
                raise PreviewError(
                    409, "recording_too_long",
                    f"This recording has {len(seconds)} saved observations; continuous "
                    f"browsing is limited to {FRAME_CACHE_MAX_FRAMES}.")
            with self._read_lock:
                try:
                    codes = load_minimap_codes(sample_dir)
                except (OSError, ValueError) as error:
                    raise PreviewError(
                        409, "preview_unavailable",
                        f"Saved minimap observations could not be read: {error}") from error
            if len(codes) != len(seconds):
                raise PreviewError(409, "preview_unavailable",
                                   "Saved minimap frames and TimeInfo timestamps disagree.")
            context = {
                "seconds": [float(value) for value in seconds],
                "loops": [int(value) for value in loops],
                "codes": codes,
            }
            self._remember_frames(key, context)
            return context

    def frames(self, job_id, player):
        """Every saved observation time of one POV (not just prediction points)."""
        if not PLAYER_PATTERN.match(str(player or "")):
            raise PreviewError(400, "invalid_player", "Choose one of the job's player POVs.")
        context = self._frame_context(job_id, player)
        seconds = context["seconds"]
        return {
            "job_id": job_id,
            "player": player,
            "frame_count": len(seconds),
            "first_seconds": seconds[0],
            "last_seconds": seconds[-1],
            "source": "TimeInfo.npz",
            "frame_seconds": [round(value, 6) for value in seconds],
        }

    def lookup_frame(self, job_id, player, raw_index):
        """Metadata for one saved observation; the PNG is served by frame_image()."""
        if not PLAYER_PATTERN.match(str(player or "")):
            raise PreviewError(400, "invalid_player", "Choose one of the job's player POVs.")
        try:
            index = int(str(raw_index).strip())
        except (TypeError, ValueError) as error:
            raise PreviewError(400, "invalid_frame",
                               "frame must be an integer observation index.") from error
        context = self._frame_context(job_id, player)
        if index < 0 or index >= len(context["seconds"]):
            raise PreviewError(422, "frame_out_of_range",
                               "frame must be a saved observation of this POV.")
        return {
            "job_id": job_id,
            "player": player,
            "frame_index": index,
            "frame_seconds": context["seconds"][index],
            "game_loop": context["loops"][index],
            "frame_count": len(context["seconds"]),
            "source": "saved_observation",
            "image_url": "/api/jobs/%s/preview/frame-%s-%d.png" % (job_id, player, index),
        }

    def frame_image(self, job_id, name):
        """PNG of one saved observation, rendered from the bounded code cache."""
        match = FRAME_NAME_PATTERN.match(str(name or ""))
        if not match:
            raise PreviewError(404, "not_found", "Unknown replay minimap frame")
        player = match.group("player")
        index = int(match.group("index"))
        if not PLAYER_PATTERN.match(player):
            raise PreviewError(404, "not_found", "Unknown replay minimap frame")
        context = self._frame_context(job_id, player)
        if index >= len(context["seconds"]):
            raise PreviewError(404, "not_found", "That observation is not in this recording")
        try:
            return minimap_frame_png(context["codes"], index)
        except ValueError as error:
            raise PreviewError(409, "preview_unavailable",
                               f"That observation could not be rendered: {error}") from error


class Handler(BaseHTTPRequestHandler):
    """Routes for the minimal local API plus read-only report/UI files."""

    server_version = "sc2-local-backend/1.0"
    protocol_version = "HTTP/1.1"

    @property
    def config(self):
        return self.server.config

    @property
    def store(self):
        return self.server.store

    @property
    def slot(self):
        return self.server.slot

    @property
    def manager(self):
        return self.server.manager

    @property
    def heatmaps(self):
        return self.server.heatmaps

    @property
    def game_views(self):
        return self.server.game_views

    @property
    def previews(self):
        return self.server.previews

    @property
    def upload_limiter(self):
        return self.server.upload_limiter

    def _upload_client_key(self):
        for header in ("CF-Connecting-IP", "X-Forwarded-For", "X-Real-IP"):
            value = self.headers.get(header)
            if value:
                return value.split(",")[0].strip()[:200]
        return str(self.client_address[0])

    def log_message(self, fmt, *args):
        LOG.debug("%s %s", self.address_string(), fmt % args)

    def _send(self, status, body, content_type, close=False):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        if close:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def json(self, status, payload, close=False):
        body = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8", close=close)

    def error(self, status, code, message, close=False):
        self.json(status, {"error": {"code": code, "message": message}}, close=close)

    def do_GET(self):
        path = urlsplit(self.path).path
        try:
            if path == "/api/support":
                self.json(200, support_payload(self.config))
            elif path.startswith("/api/jobs/"):
                self._api_job(path)
            elif path.startswith("/jobs/"):
                self._serve_public(path)
            elif path in ("/", "/index.html"):
                self.send_response(302)
                self.send_header("Location", "/ui/index.html")
                self.send_header("Content-Length", "0")
                self.end_headers()
            elif path.startswith("/ui/"):
                self._serve_ui(path[4:])
            else:
                self.error(404, "not_found", f"No route for {path}")
        except JobNotFound:
            self.error(404, "job_not_found", "Unknown job id")
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:  # noqa: BLE001 - never leak a stack trace to the browser
            LOG.exception("GET %s failed", self.path)
            self.error(500, "internal_error", "The request could not be completed.")

    def _api_job(self, path):
        match = re.match(r"^/api/jobs/(?P<job>[^/]+)(?P<rest>/.*)?$", path)
        if not match:
            return self.error(404, "not_found", "Unknown API route")
        job_id = match.group("job")
        rest = match.group("rest") or ""
        if not JOB_ID_PATTERN.match(job_id):
            raise JobNotFound(job_id)
        record = self.store.read(job_id)
        if rest in ("", "/"):
            return self.json(200, job_payload(record))
        if rest == "/report":
            if record.get("state") != "completed":
                return self.json(409, {
                    "error": {"code": "not_ready",
                              "message": f"Job is {record.get('state')!r} "
                                         f"(stage {record.get('stage')!r}); the report is not "
                                         "available yet."},
                    "job_id": job_id, "state": record.get("state"),
                    "stage": record.get("stage"), "pov": record.get("pov"),
                })
            try:
                payload = self.store.public_index(job_id)
            except Exception:  # noqa: BLE001 - a broken index is a real failure
                LOG.exception("could not build the report index for %s", job_id)
                return self.error(500, "report_unavailable",
                                  "The stored report index could not be read.")
            return self.json(200, payload)
        if rest == "/frames":
            query = parse_qs(urlsplit(self.path).query)
            player = (query.get("player") or [""])[0]
            try:
                payload = self.previews.frames(job_id, player)
            except PreviewError as error:
                return self.error(error.status, error.code, error.message)
            return self.json(200, payload)
        if rest == "/explanations/prepare":
            query = parse_qs(urlsplit(self.path).query)
            player = (query.get("player") or [""])[0]
            try:
                payload = self.heatmaps.prepare_status(job_id, player)
            except HeatmapRequestError as error:
                return self.error(error.status, error.code, error.message)
            return self.json(200, payload)
        if rest == "/preview":
            query = parse_qs(urlsplit(self.path).query)
            player = (query.get("player") or [""])[0]
            frame = (query.get("frame") or [""])[0]
            cutoff = (query.get("cutoff") or [""])[0]
            try:
                if frame != "":
                    payload = self.previews.lookup_frame(job_id, player, frame)
                else:
                    payload = self.previews.lookup(job_id, player, cutoff)
            except PreviewError as error:
                return self.error(error.status, error.code, error.message)
            return self.json(200, payload)
        preview_image = re.match(r"^/preview/(?P<name>[^/]+)$", rest)
        if preview_image:
            name = unquote(preview_image.group("name"))
            try:
                if FRAME_NAME_PATTERN.match(name):
                    data = self.previews.frame_image(job_id, name)
                else:
                    data = self.previews.image(job_id, name)
            except PreviewError as error:
                return self.error(error.status, error.code, error.message)
            return self._send(200, data, "image/png")
        heatmap_match = re.match(r"^/heatmaps/(?P<heatmap>[^/]+)$", rest)
        if heatmap_match:
            try:
                payload = self.heatmaps.status(job_id, unquote(heatmap_match.group("heatmap")))
            except HeatmapRequestError as error:
                return self.error(error.status, error.code, error.message)
            return self.json(200, payload)
        return self.error(404, "not_found", "Unknown job API route")

    def _serve_public(self, path):
        match = re.match(r"^/jobs/(?P<job>[^/]+)/report/(?P<player>[^/]+)/(?P<name>[^/]+)$", path)
        if not match:
            return self.error(404, "not_found", "Unknown report path")
        job_id = match.group("job")
        player = unquote(match.group("player"))
        name = unquote(match.group("name"))
        content_type = public_content_type(name)
        if content_type is None:
            return self.error(404, "not_found", "Only exported review.json and heatmap PNG files are served")
        record = self.store.read(job_id)
        players = ((record.get("report") or {}).get("players") or {})
        if player not in players:
            return self.error(404, "not_found", "Unknown player for this job")
        info = players[player] or {}
        allowed = {"review.json"}
        if isinstance(info.get("image"), str):
            allowed.add(Path(info["image"]).name)
        for image in info.get("images") or []:
            if isinstance(image, str):
                allowed.add(Path(image).name)
        if name not in allowed:
            return self.error(404, "not_found", "That file is not part of this report")
        if record.get("state") != "completed":
            return self.error(409, "not_ready", f"Job is {record.get('state')!r}")
        public_root = self.store.public_dir(job_id).resolve()
        target = (public_root / player / name).resolve()
        if public_root not in target.parents:
            return self.error(404, "not_found", "Path is outside the report directory")
        if not target.is_file():
            try:
                self.store.export_public(job_id)
            except Exception:  # noqa: BLE001 - keep the 404 honest
                LOG.exception("could not re-export %s", job_id)
        if not target.is_file():
            return self.error(404, "report_missing", "The report file is not available")
        self._send(200, target.read_bytes(), content_type)

    def _serve_ui(self, relative):
        relative = unquote(relative)
        if not relative or relative.endswith("/"):
            relative += "index.html"
        root = UI_ROOT.resolve()
        candidate = (root / relative).resolve()
        if candidate != root and root not in candidate.parents:
            return self.error(404, "not_found", "Path is outside the UI directory")
        if not candidate.is_file():
            return self.error(404, "not_found", "Unknown UI file")
        content_type = UI_TYPES.get(candidate.suffix.lower())
        if content_type is None:
            return self.error(404, "not_found", "Only UI assets are served")
        self._send(200, candidate.read_bytes(), content_type)

    # -------------------------------------------------------------------- POST
    def handle_expect_100(self):
        """Reject busy/oversized uploads before the client streams the body."""
        if self.command == "POST" and urlsplit(self.path).path == "/api/jobs":
            if not self.slot.is_free():
                self.error(409, "busy",
                           "Another analysis is running; retry after it finishes.", close=True)
                return False
            if (self.config.public_mode and self.upload_limiter is not None
                    and not self.upload_limiter.can_attempt(self._upload_client_key())):
                self.error(429, "rate_limited",
                           "This client has reached the public upload limit. Try again later.",
                           close=True)
                return False
            try:
                length = int(self.headers.get("Content-Length") or "")
            except ValueError:
                length = -1
            if length > self.config.max_upload_bytes + MAX_ENVELOPE_SLACK:
                self.error(413, "upload_too_large",
                           f"Upload exceeds {self.config.max_upload_bytes} bytes.", close=True)
                return False
        return super().handle_expect_100()

    def do_POST(self):
        path = urlsplit(self.path).path
        game_view_match = re.match(r"^/api/jobs/(?P<job>[^/]+)/game-view$", path)
        if game_view_match:
            try:
                self._handle_game_view_request(game_view_match.group("job"))
            except HeatmapRequestError as error:
                self.error(error.status, error.code, error.message)
            except JobNotFound:
                self.error(404, "job_not_found", "Unknown job id")
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
            except Exception:  # noqa: BLE001
                LOG.exception("POST %s failed", self.path)
                self.error(500, "internal_error", "The game-view request could not be processed.")
            return
        prepare_match = re.match(r"^/api/jobs/(?P<job>[^/]+)/explanations/prepare$", path)
        if prepare_match:
            try:
                self._handle_prepare_request(prepare_match.group("job"))
            except HeatmapRequestError as error:
                self.error(error.status, error.code, error.message)
            except JobNotFound:
                self.error(404, "job_not_found", "Unknown job id")
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
            except Exception:  # noqa: BLE001
                LOG.exception("POST %s failed", self.path)
                self.error(500, "internal_error",
                           "The explanation preparation request could not be processed.")
            return
        heatmap_match = re.match(r"^/api/jobs/(?P<job>[^/]+)/heatmaps$", path)
        if heatmap_match:
            try:
                self._handle_heatmap_request(heatmap_match.group("job"))
            except HeatmapRequestError as error:
                self.error(error.status, error.code, error.message)
            except JobNotFound:
                self.error(404, "job_not_found", "Unknown job id")
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
            except Exception:  # noqa: BLE001 - keep stack traces local
                LOG.exception("POST %s failed", self.path)
                self.error(500, "internal_error", "The heatmap request could not be processed.")
            return
        if path != "/api/jobs":
            return self.error(404, "not_found", f"No route for POST {path}")
        try:
            self._handle_upload()
        except multipart.UploadError as error:
            self.error(error.status, error.code, error.message,
                       close=error.status == 413)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:  # noqa: BLE001 - keep stack traces local
            LOG.exception("POST %s failed", self.path)
            self.error(500, "internal_error", "The upload could not be processed.")

    def _handle_heatmap_request(self, job_id):
        if not JOB_ID_PATTERN.match(job_id):
            raise JobNotFound(job_id)
        content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if content_type != "application/json":
            raise HeatmapRequestError(400, "unsupported_content_type",
                                      "Send application/json with player and cutoff_seconds.")
        length_header = self.headers.get("Content-Length")
        if not length_header:
            raise HeatmapRequestError(411, "length_required", "Send Content-Length.")
        try:
            length = int(length_header)
        except ValueError:
            raise HeatmapRequestError(400, "invalid_length", "Content-Length is not an integer.")
        if length <= 0:
            raise HeatmapRequestError(400, "empty_body", "The request body is empty.")
        if length > HEATMAP_REQUEST_LIMIT:
            raise HeatmapRequestError(413, "request_too_large",
                                      f"JSON body exceeds {HEATMAP_REQUEST_LIMIT} bytes.")
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise HeatmapRequestError(400, "invalid_json", "The JSON body is invalid.")
        if not isinstance(body, dict):
            raise HeatmapRequestError(400, "invalid_json", "The JSON body must be an object.")
        if set(body) - {"player", "cutoff_seconds"}:
            raise HeatmapRequestError(400, "invalid_request",
                                      "Only player and cutoff_seconds are accepted.")
        player = body.get("player")
        cutoff = body.get("cutoff_seconds")
        if not isinstance(player, str) or not re.match(r"^player\d+$", player):
            raise HeatmapRequestError(400, "invalid_player", "player must be playerN.")
        if isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)):
            raise HeatmapRequestError(400, "invalid_cutoff", "cutoff_seconds must be a number.")
        payload, status = self.heatmaps.request(job_id, player, float(cutoff))
        self.json(status, payload)

    def _handle_prepare_request(self, job_id):
        if not JOB_ID_PATTERN.match(job_id):
            raise JobNotFound(job_id)
        content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if content_type != "application/json":
            raise HeatmapRequestError(400, "unsupported_content_type",
                                      "Send application/json with player.")
        length_header = self.headers.get("Content-Length")
        if not length_header:
            raise HeatmapRequestError(411, "length_required", "Send Content-Length.")
        try:
            length = int(length_header)
        except ValueError:
            raise HeatmapRequestError(400, "invalid_length", "Content-Length is not an integer.")
        if length <= 0:
            raise HeatmapRequestError(400, "empty_body", "The request body is empty.")
        if length > HEATMAP_REQUEST_LIMIT:
            raise HeatmapRequestError(413, "request_too_large",
                                      f"JSON body exceeds {HEATMAP_REQUEST_LIMIT} bytes.")
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise HeatmapRequestError(400, "invalid_json", "The JSON body is invalid.")
        if not isinstance(body, dict):
            raise HeatmapRequestError(400, "invalid_json", "The JSON body must be an object.")
        if set(body) - {"player"}:
            raise HeatmapRequestError(400, "invalid_request", "Only player is accepted.")
        player = body.get("player")
        if not isinstance(player, str) or not re.match(r"^player\d+$", player):
            raise HeatmapRequestError(400, "invalid_player", "player must be playerN.")
        payload = self.heatmaps.prepare(job_id, player)
        self.json(202, payload)

    def _handle_game_view_request(self, job_id):
        if not JOB_ID_PATTERN.match(job_id):
            raise JobNotFound(job_id)
        content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if content_type != "application/json":
            raise HeatmapRequestError(400, "unsupported_content_type",
                                      "Send application/json with player and cutoff_seconds.")
        length_header = self.headers.get("Content-Length")
        if not length_header:
            raise HeatmapRequestError(411, "length_required", "Send Content-Length.")
        try:
            length = int(length_header)
        except ValueError:
            raise HeatmapRequestError(400, "invalid_length", "Content-Length is not an integer.")
        if length <= 0 or length > HEATMAP_REQUEST_LIMIT:
            raise HeatmapRequestError(413, "request_too_large",
                                      f"JSON body must be 1..{HEATMAP_REQUEST_LIMIT} bytes.")
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise HeatmapRequestError(400, "invalid_json", "The JSON body is invalid.")
        if not isinstance(body, dict) or set(body) - {"player", "cutoff_seconds"}:
            raise HeatmapRequestError(400, "invalid_request",
                                      "Only player and cutoff_seconds are accepted.")
        player = body.get("player")
        cutoff = body.get("cutoff_seconds")
        if not isinstance(player, str) or not re.match(r"^player\d+$", player):
            raise HeatmapRequestError(400, "invalid_player", "player must be playerN.")
        if isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)):
            raise HeatmapRequestError(400, "invalid_cutoff", "cutoff_seconds must be a number.")
        payload = self.game_views.render(job_id, player, float(cutoff))
        self.json(200, payload)

    def _handle_upload(self):
        limit = self.config.max_upload_bytes
        envelope = limit + MAX_ENVELOPE_SLACK
        if not self.slot.is_free():
            return self.error(409, "busy",
                              "Another analysis is running; retry after it finishes.", close=True)
        if (self.config.public_mode and self.upload_limiter is not None
                and not self.upload_limiter.allow(self._upload_client_key())):
            return self.error(429, "rate_limited",
                              "This client has reached the public upload limit. Try again later.",
                              close=True)
        length_header = self.headers.get("Content-Length")
        if not length_header:
            return self.error(411, "length_required",
                              "Send Content-Length; chunked uploads are not supported.",
                              close=True)
        try:
            length = int(length_header)
        except ValueError:
            return self.error(400, "invalid_length", "Content-Length is not an integer.")
        if length <= 0:
            return self.error(400, "empty_body", "The request body is empty.")
        if length > envelope:
            try:
                multipart.drain(self.rfile, min(length, MAX_DRAIN_BYTES))
            except OSError:
                self.close_connection = True
            return self.error(413, "upload_too_large",
                              f"Upload exceeds {limit} bytes.", close=True)
        body = multipart.read_body(self.rfile, length, envelope)
        part = multipart.parse_file_part(self.headers.get("Content-Type"), body)
        data = part["data"]
        if not data:
            return self.error(400, "empty_file", "The uploaded file is empty.")
        if len(data) > limit:
            return self.error(413, "upload_too_large",
                              f"The replay is {len(data)} bytes; the limit is {limit}.")
        if self.config.max_jobs_bytes > 0:
            used = self.store.size_bytes()
            if used + len(data) > self.config.max_jobs_bytes:
                return self.error(
                    507, "storage_full",
                    "The public demo storage limit is reached. Existing reports were kept; "
                    "try again after the operator makes room.")
        incoming = self.store.root / ".incoming" / f"{uuid.uuid4().hex}.upload"
        incoming.write_bytes(data)
        acquired = False
        try:
            details = inspect_upload(incoming)
            if not self.slot.try_acquire():
                return self.error(409, "busy",
                                  "Another analysis started first; retry later.")
            acquired = True
            config_summary = dict(ANALYSIS_DEFAULTS)
            record = self.store.create(part["filename"], data, details, config_summary)
            self.manager.start(record["job_id"])
            acquired = False  # ownership passes to the runner, which releases the slot
            LOG.info("accepted job %s (%s bytes) from %r", record["job_id"],
                     record["upload"]["bytes"], record["original_name"])
            self.json(202, {
                "format_version": 1,
                "job_id": record["job_id"],
                "state": "queued",
                "status_url": f"/api/jobs/{record['job_id']}",
                "report_url": f"/api/jobs/{record['job_id']}/report",
                "poll_after_seconds": 3,
                "note": "Analysis runs in the background; poll the status URL.",
            })
        finally:
            incoming.unlink(missing_ok=True)
            if acquired:
                self.slot.release()


class BackendServer(ThreadingHTTPServer):
    """Exclusive bind (no SO_REUSEADDR hijacking on Windows) with daemon threads."""

    allow_reuse_address = False
    daemon_threads = True


def make_server(host, port, attempts=20):
    """Bind the requested port, or the next free one within ``attempts``."""
    last_error = None
    for offset in range(attempts):
        candidate = (port + offset) if port else 0
        try:
            return BackendServer((host, candidate), Handler)
        except OSError as error:
            last_error = error
    raise SystemExit(f"could not bind {host}:{port} (+{attempts}): {last_error}")


def configure_logging(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOG.handlers.clear()
    LOG.addHandler(file_handler)
    LOG.addHandler(stream_handler)
    LOG.setLevel(logging.INFO)
    LOG.propagate = False


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address; keep the loopback default for local use")
    parser.add_argument("--port", type=int, default=8765,
                        help="0 lets the OS choose; otherwise the next free port is used")
    parser.add_argument("--sc2-path", default=str(SC2_PATH),
                        help="StarCraft II folder containing Versions/")
    parser.add_argument("--team-dir", required=True,
                        help="Team project with viz/ and checkpoints/ (read-only)")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt",
                        help="Resolved inside --team-dir when relative")
    parser.add_argument("--jobs-root", default=None,
                        help="Where job directories live; default runs/backend-jobs, or "
                             "runs/public-jobs with --public-demo")
    parser.add_argument("--public-demo", action="store_true",
                        help="Isolated public-demo mode: separate default jobs root, 128-bit "
                             "job ids, upload rate limit and storage cap")
    parser.add_argument("--lock-file", default=None,
                        help="Shared heavy-task lock; default <jobs-root>/.active.lock. "
                             "Local and public backends must use the same file")
    parser.add_argument("--uploads-per-hour", type=int, default=6,
                        help="Public-demo upload attempts per client per window (0 disables)")
    parser.add_argument("--rate-window-seconds", type=int, default=3600,
                        help="Public-demo upload rate-limit window")
    parser.add_argument("--job-timeout", type=int, default=3600,
                        help="Maximum wall-clock seconds for one analysis (0 disables)")
    parser.add_argument("--max-jobs-mib", type=float, default=4096.0,
                        help="Public-demo storage cap for its jobs root, in MiB")
    parser.add_argument("--allow-download-map", action="store_true",
                        help="Allow the analysis to fetch this replay's missing cache archives")
    parser.add_argument("--max-upload-mib", type=float, default=20.0,
                        help="Upload limit in MiB, enforced on received bytes")
    parser.add_argument("--python", default=sys.executable,
                        help="Interpreter used for the analysis subprocess")
    parser.add_argument("--engine-timeout", type=int, default=300,
                        help="Engine startup timeout passed to the analysis")
    parser.add_argument("--log-file", default=None,
                        help="Backend log; default <jobs-root>/server.log")
    args = parser.parse_args(argv)

    sc2_path = Path(args.sc2_path).expanduser().resolve()
    if not sc2_path.is_dir():
        raise SystemExit(f"SC2 install not found: {sc2_path}")
    try:
        team_dir = resolve_team_dir(Path(args.team_dir), args.checkpoint)
        checkpoint = resolve_checkpoint(args.checkpoint, team_dir)
    except Failure as error:
        raise SystemExit(f"{error.code}: {error.message}")
    install = deps.describe_install(sc2_path)
    if not install["required_exe_exists"]:
        raise SystemExit(f"Base{ENGINE_LABEL.build} is missing under {sc2_path}. Run "
                         f"`python -m tools.prepare_replay_dependencies --download` once.")

    default_jobs = PROJECT_ROOT / "runs" / ("public-jobs" if args.public_demo else "backend-jobs")
    jobs_root = Path(args.jobs_root).expanduser().resolve() if args.jobs_root else default_jobs.resolve()
    legacy_jobs = (PROJECT_ROOT / "runs" / "backend-jobs").resolve()
    if args.public_demo and jobs_root == legacy_jobs:
        raise SystemExit("public-demo must not use runs/backend-jobs; choose a separate "
                         "--jobs-root (for example runs/public-jobs)")
    jobs_root.mkdir(parents=True, exist_ok=True)
    configure_logging(Path(args.log_file) if args.log_file else jobs_root / "server.log")

    store = JobStore(jobs_root, public_mode=args.public_demo)
    recovered = store.recover()
    if recovered:
        LOG.warning("marked as interrupted at startup: %s", ", ".join(recovered))
    lock_path = (Path(args.lock_file).expanduser().resolve() if args.lock_file
                 else jobs_root / ".active.lock")
    slot = ActiveSlot(lock_path)
    if not slot.is_free():
        LOG.warning("another heavy task holds %s; new uploads will return 409 until it ends",
                    slot.path)
    config = Config(sc2_path=sc2_path, team_dir=team_dir, checkpoint=checkpoint,
                    jobs_root=jobs_root, allow_download_map=args.allow_download_map,
                    max_upload_bytes=int(args.max_upload_mib * 1024 * 1024),
                    python=args.python, engine_timeout=args.engine_timeout,
                    public_mode=args.public_demo,
                    uploads_per_hour=max(0, args.uploads_per_hour),
                    rate_window_seconds=max(1, args.rate_window_seconds),
                    job_timeout=max(0, args.job_timeout),
                    max_jobs_bytes=(int(args.max_jobs_mib * 1024 * 1024)
                                    if args.public_demo else 0),
                    lock_path=lock_path)
    manager = JobManager(config, store, slot, LOG)
    heatmaps = HeatmapManager(config, store, slot, LOG)
    game_views = GameViewManager(config, store, slot, LOG)
    previews = PreviewManager(store, LOG)
    server = make_server(args.host, args.port)
    server.config = config
    server.store = store
    server.slot = slot
    server.manager = manager
    server.heatmaps = heatmaps
    server.game_views = game_views
    server.previews = previews
    server.upload_limiter = (UploadRateLimiter(config.uploads_per_hour,
                                               config.rate_window_seconds)
                             if config.public_mode else None)
    host, port = server.server_address[0], server.server_address[1]

    stopping = threading.Event()

    def request_stop(signum, frame):  # noqa: ARG001 - signal handler signature
        if stopping.is_set():
            return
        stopping.set()
        LOG.info("stop requested (signal %s)", signum)
        threading.Thread(target=server.shutdown, daemon=True).start()

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            try:
                signal.signal(getattr(signal, name), request_stop)
            except (OSError, ValueError):
                pass

    print(f"listening on http://{host}:{port}", flush=True)
    print(f"mode: {'public-demo' if config.public_mode else 'local'}", flush=True)
    print(f"support: http://{host}:{port}/api/support", flush=True)
    print(f"jobs root: {jobs_root}", flush=True)
    print(f"shared lock: {lock_path}", flush=True)
    LOG.info("listening on %s:%s (mode=%s, team=%s, jobs=%s, lock=%s, "
             "allow_download_map=%s, max_upload=%s bytes)",
             host, port, "public-demo" if config.public_mode else "local", team_dir,
             jobs_root, lock_path, config.allow_download_map, config.max_upload_bytes)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        manager.stop_all()
        heatmaps.stop_all()
        game_views.stop_all()
        LOG.info("backend stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
