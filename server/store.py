"""On-disk job store for the local backend.

Every job owns one directory; ``job.json`` is replaced atomically on each change so the
API never exposes a half-written record. Uploads keep a server-generated name, the client
filename is display-only, and only ``public/`` is ever served to a browser.
"""

import json
import os
import re
import secrets
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path

if os.name == "nt":  # pragma: no cover - platform lock primitive
    import msvcrt
else:  # pragma: no cover
    import fcntl

JOB_ID_PATTERN = re.compile(r"^job-\d{8}T\d{6}Z-(?:[0-9a-f]{6}|[0-9a-f]{32})$")
PLAYER_PATTERN = re.compile(r"^player\d+$")
PUBLIC_FILES = ("review.json", "gradcam_1.png")
PUBLIC_IMAGE_PATTERN = re.compile(r"^(gradcam|minimap|rgb)_[A-Za-z0-9_-]+\.png$")


class JobNotFound(Exception):
    """Unknown or malformed job id."""


def now():
    """UTC timestamp with second precision."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_json(path, payload):
    """Write JSON through a temporary file and ``os.replace``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def new_job_id(public=False):
    """Server-generated id: timestamp plus a path-safe random suffix.

    Local jobs keep the historical 24-bit suffix so existing links remain readable.
    Public-demo jobs use 16 random bytes (128 bits) and are not enumerable in practice.
    """
    prefix = datetime.now(timezone.utc).strftime("job-%Y%m%dT%H%M%SZ")
    suffix = secrets.token_hex(16 if public else 3)
    return prefix + "-" + suffix


def display_name(filename):
    """Client filename reduced to display text; it never influences disk paths."""
    name = (filename or "").replace("\\", "/").split("/")[-1]
    name = "".join(ch for ch in name if ch.isprintable()).strip()
    name = re.sub(r"\s+", " ", name)[:120]
    return name or "replay.sc2replay"


class ActiveSlot:
    """Exclusive single-analysis slot, released automatically when this process exits."""

    def __init__(self, path):
        self.path = Path(path)
        self._handle = None
        self._mutex = threading.Lock()

    def try_acquire(self):
        with self._mutex:
            if self._handle is not None:
                return False
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle = self.path.open("a+b")
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                if os.name == "nt":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                handle.close()
                return False
            handle.seek(0)
            handle.truncate()
            handle.write(f"{os.getpid()}\n".encode("ascii"))
            handle.flush()
            self._handle = handle
            return True

    def release(self):
        with self._mutex:
            if self._handle is None:
                return
            handle, self._handle = self._handle, None
            try:
                handle.seek(0)
                if os.name == "nt":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()

    def is_free(self):
        """Best-effort probe; the authoritative test is ``try_acquire``."""
        if self._handle is not None:
            return False
        if self.try_acquire():
            self.release()
            return True
        return False


class JobStore:
    """Create, update and serve job records under one root directory."""

    def __init__(self, root, public_mode=False):
        self.root = Path(root)
        self.public_mode = bool(public_mode)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / ".incoming").mkdir(exist_ok=True)
        self._lock = threading.RLock()

    def job_dir(self, job_id):
        if not JOB_ID_PATTERN.match(str(job_id)):
            raise JobNotFound(job_id)
        directory = self.root / job_id
        if not directory.is_dir():
            raise JobNotFound(job_id)
        return directory

    def record_path(self, job_id):
        return self.job_dir(job_id) / "job.json"

    def read(self, job_id):
        path = self.record_path(job_id)
        if not path.is_file():
            raise JobNotFound(job_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def update(self, job_id, **fields):
        with self._lock:
            record = self.read(job_id)
            record.update(fields)
            record["updated_at"] = now()
            atomic_write_json(self.record_path(job_id), record)
            return record

    def create(self, original_name, data, details, config):
        """Materialise one job directory; the upload is renamed by the server."""
        with self._lock:
            job_id = new_job_id(self.public_mode)
            directory = self.root / job_id
            while directory.exists():
                job_id = new_job_id(self.public_mode)
                directory = self.root / job_id
            (directory / "upload").mkdir(parents=True)
            (directory / "logs").mkdir()
            upload_path = directory / "upload" / "input.SC2Replay"
            upload_path.write_bytes(data)
            written = upload_path.stat().st_size
            if written != len(data):
                raise OSError(f"wrote {written} of {len(data)} upload bytes")
            created = now()
            record = {
                "format_version": 1,
                "job_id": job_id,
                "state": "queued",
                "stage": None,
                "pov": None,
                "run_state": None,
                "created_at": created,
                "queued_at": created,
                "started_at": None,
                "updated_at": created,
                "finished_at": None,
                "error_code": None,
                "error_message": None,
                "exit_code": None,
                "original_name": display_name(original_name),
                "upload": {"bytes": len(data), "sha256": details["sha256"],
                           "stored_name": upload_path.name},
                "replay": {
                    "replay_id": details["replay_id"],
                    "game_version": details["game_version"],
                    "map_title": details["map_title"],
                    "duration_seconds": details["duration_seconds"],
                    "terran_win_label": details["label"],
                },
                "config": config,
                "report": None,
            }
            atomic_write_json(directory / "job.json", record)
            return record

    def upload_path(self, job_id):
        return self.job_dir(job_id) / "upload" / "input.SC2Replay"

    def size_bytes(self):
        """Bytes currently stored in this root's job directories (not .incoming)."""
        total = 0
        for directory in self.root.iterdir():
            if not directory.is_dir() or not JOB_ID_PATTERN.match(directory.name):
                continue
            for path in directory.rglob("*"):
                try:
                    if path.is_file() and not path.is_symlink():
                        total += path.stat().st_size
                except OSError:
                    continue
        return total

    def run_dir(self, job_id):
        return self.job_dir(job_id) / "run"

    def run_status(self, job_id):
        path = self.run_dir(job_id) / "status.json"
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def public_dir(self, job_id):
        return self.job_dir(job_id) / "public"

    def recover(self):
        """Mark jobs left ``queued``/``running`` by a previous process as interrupted."""
        recovered = []
        for directory in sorted(self.root.iterdir()):
            if not directory.is_dir() or not JOB_ID_PATTERN.match(directory.name):
                continue
            path = directory / "job.json"
            if not path.is_file():
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            if record.get("state") in ("queued", "running"):
                run_state = self.run_status(directory.name).get("state")
                self.update(directory.name, state="failed", stage="interrupted",
                            error_code="interrupted",
                            error_message=("The backend stopped before this analysis finished. "
                                           "Re-submit the replay; partial files are kept on disk."),
                            finished_at=now(), run_state=run_state)
                recovered.append(directory.name)
        return recovered

    def export_public(self, job_id):
        """Copy every referenced review image into ``public/`` for serving.

        The original static heatmap is ``gradcam_1.png``; on-demand heatmaps get
        unique ``gradcam_<pov>_<cutoff>s_<key>.png`` names. Only files referenced by
        the review's explanation list are copied, and every destination stays under
        ``public/<player>/``.
        """
        record = self.read(job_id)
        run_dir = self.run_dir(job_id)
        index_path = run_dir / "report_index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"missing {index_path}")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        public_root = self.public_dir(job_id)
        players = {}
        for pov in index.get("povs") or []:
            player = pov.get("player")
            if not PLAYER_PATTERN.match(str(player)):
                raise ValueError(f"unexpected player name in report index: {player!r}")
            target = public_root / player
            target.mkdir(parents=True, exist_ok=True)
            review_path = run_dir / pov["review"]
            review = json.loads(review_path.read_text(encoding="utf-8"))
            review.pop("sources", None)
            checkpoint = review.get("checkpoint")
            review["checkpoint"] = Path(checkpoint).name if isinstance(checkpoint, str) else None
            checkpoint_info = index.get("checkpoint") or {}
            review["checkpoint_sha256"] = checkpoint_info.get("sha256")
            review["engine"] = index.get("engine")
            params = index.get("params") or {}
            review["run_params"] = {
                key: params.get(key) for key in
                ("frames", "step", "draws", "seed", "device", "threshold", "merge_seconds")
            }
            review["paths_hidden"] = "absolute local paths are removed by the backend"

            images = []
            for explanation in review.get("explanations") or []:
                game_view = explanation.get("game_view") if isinstance(explanation.get("game_view"), dict) else {}
                for image in (explanation.get("image"), explanation.get("context_image"),
                              game_view.get("image")):
                    if not image:
                        continue
                    name = Path(image).name
                    if name != image or not PUBLIC_IMAGE_PATTERN.match(name):
                        raise ValueError(f"unsafe explanation image name: {image!r}")
                    source = review_path.parent / name
                    if not source.is_file():
                        raise FileNotFoundError(f"missing explanation image {source}")
                    if name not in images:
                        shutil.copyfile(source, target / name)
                        images.append(name)
            # Compatibility with an older index entry that predates explanations[].image.
            main_image = (pov.get("explanation") or {}).get("image")
            if main_image:
                source = (run_dir / main_image).resolve()
                if run_dir.resolve() not in source.parents:
                    raise ValueError(f"index image escapes the run directory: {main_image!r}")
                main_name = source.name
                if not PUBLIC_IMAGE_PATTERN.match(main_name):
                    raise ValueError(f"unsafe index image name: {main_image!r}")
                if main_name not in images:
                    if not source.is_file():
                        raise FileNotFoundError(f"missing index image {source}")
                    shutil.copyfile(source, target / main_name)
                    images.append(main_name)
            atomic_write_json(target / "review.json", review)
            image_name = "gradcam_1.png" if "gradcam_1.png" in images else (images[0] if images else None)
            players[player] = {
                "review": f"public/{player}/review.json",
                "image": f"public/{player}/{image_name}" if image_name else None,
                "images": images,
            }
        report = {"players": players, "exported_at": now()}
        self.update(job_id, report=report)
        return report

    def public_index(self, job_id):
        """Browser-safe report index: relative URLs only, no disk paths."""
        record = self.read(job_id)
        index_path = self.run_dir(job_id) / "report_index.json"
        index = {}
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
        povs = []
        for pov in index.get("povs") or []:
            player = pov.get("player")
            if not PLAYER_PATTERN.match(str(player)):
                continue
            explanation = {key: value for key, value in (pov.get("explanation") or {}).items()
                           if key != "image"}
            povs.append({
                "player": player,
                "focus_player_id": pov.get("focus_player_id"),
                "focus_player_race": pov.get("focus_player_race"),
                "result": pov.get("result"),
                "label": pov.get("focus_player_label"),
                "label_meaning": pov.get("label_meaning"),
                "review_url": f"/jobs/{job_id}/report/{player}/review.json",
                "image_url": f"/jobs/{job_id}/report/{player}/gradcam_1.png",
                "ui_url": f"/ui/index.html?review=/jobs/{job_id}/report/{player}",
                "points": pov.get("points") or [],
                "explanation": explanation,
            })
        return {
            "format_version": 1,
            "job_id": job_id,
            "state": record.get("state"),
            "replay": record.get("replay"),
            "config": record.get("config"),
            "engine": record.get("engine"),
            "povs": povs,
        }
