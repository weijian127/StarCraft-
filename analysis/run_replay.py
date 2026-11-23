"""Run one raw replay end to end through the existing local pipeline.

Stages, in order: ``validating`` -> ``checking_dependencies`` -> ``extracting`` ->
``predicting`` -> ``explaining`` -> ``exporting`` -> ``completed`` (or ``failed``).

Nothing is reimplemented here; each stage drives an existing component:

  * ``data_processing.processor``          extraction; its version checks are untouched
  * ``tools.prepare_replay_dependencies``  cache-handle parsing, official depot fetch,
                                           and the engine proof (Ping + ReplayInfo)
  * team ``viz.timeline`` / ``viz.gradcam`` executed as subprocesses with the team
                                           directory as the working directory, so its
                                           ``config``/``data_processing`` modules are never
                                           confused with this project's copies
  * ``analysis.adapt_viz``                 review.json conversion (the single label flip)

The team directory is read-only input. It is never imported into this process and never
modified; only two subprocesses run there.

Every artefact of one invocation lives under its own run directory::

    <run>/input/<replay>            verbatim copy of the input
    <run>/input.json                source path, sha256, size
    <run>/params.json               resolved parameters
    <run>/status.json               machine-readable state (atomic replacement)
    <run>/logs/<stage>.log          stage logs, including every subprocess transcript
    <run>/processed/both/           processor output (manifest + both POV folders)
    <run>/timeline/<player>/        viz.timeline probability.json/png/svg
    <run>/gradcam/<player>-<s>s/    viz.gradcam snapshot + data.npz + method.json
    <run>/review/<player>/          adapted review.json + gradcam_1.png
    <run>/report_index.json         index for later backends and POV switches

Exit codes: ``0`` success, ``1`` stage failure, ``2`` output directory already exists,
``3`` another run holds the lock. Nothing is trained, no video is produced, and a failed
run directory is never deleted.
"""

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mpyq  # noqa: E402
import numpy as np  # noqa: E402

from config import PROJECT_ROOT as CONFIG_ROOT, SC2_PATH, resolve_path  # noqa: E402
from data_processing import processor  # noqa: E402
from tools import prepare_replay_dependencies as deps  # noqa: E402

if os.name == "nt":  # pragma: no cover - platform lock primitive
    import msvcrt
else:  # pragma: no cover
    import fcntl

STATUS_FORMAT_VERSION = 1
REPORT_INDEX_FORMAT_VERSION = 1
STAGES = ("validating", "checking_dependencies", "extracting",
          "predicting", "explaining", "exporting")
FOCUS_LABELS = "0 = focus player loses, 1 = focus player wins"
TERRAN_LABELS = "0 = Protoss wins, 1 = Terran wins"
MINIMAP_SHAPE = (processor.MINIMAP_SIZE, processor.MINIMAP_SIZE)


class Failure(Exception):
    """A stage failure with a stable machine-readable code."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


class Busy(Exception):
    """Another run already holds the process lock."""


def now():
    """UTC timestamp with second precision, used for every status field."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean_env():
    """Environment for subprocesses: no PYTHONPATH, so ``-m`` resolves locally."""
    env = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(name, None)
    return env


class StageLogger:
    """Append-only stage log that also echoes to stdout."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def __call__(self, message):
        line = f"[{now()}] {message}"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)


class Journal:
    """``status.json`` written by atomic replacement after every transition."""

    def __init__(self, run_dir):
        self.path = Path(run_dir) / "status.json"
        self.started = now()
        self.data = {
            "format_version": STATUS_FORMAT_VERSION,
            "state": "running",
            "stage": None,
            "started_at": self.started,
            "updated_at": None,
            "finished_at": None,
            "pov": None,
            "error_code": None,
            "error_message": None,
            "report_index": None,
            "stages": [],
        }
        self._open = None
        self._t0 = None

    def write(self):
        self.data["updated_at"] = now()
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)

    def stage(self, name, pov=None):
        self.close_stage("completed")
        if name not in STAGES:
            raise ValueError(f"Unknown stage: {name}")
        self.data["state"] = "running"
        self.data["stage"] = name
        self.data["pov"] = pov
        self._open = {"name": name, "pov": pov, "started_at": now(),
                      "finished_at": None, "seconds": None, "status": "running"}
        self._t0 = time.monotonic()
        self.data["stages"].append(self._open)
        self.write()

    def close_stage(self, status):
        if self._open is None or self._open["status"] != "running":
            return
        self._open["finished_at"] = now()
        if self._t0 is not None:
            self._open["seconds"] = round(time.monotonic() - self._t0, 3)
        self._open["status"] = status
        self._open = None
        self._t0 = None

    def fail(self, code, message):
        self.close_stage("failed")
        self.data.update(state="failed", error_code=code, error_message=message,
                         finished_at=now())
        self.write()

    def complete(self, report_index):
        self.close_stage("completed")
        self.data.update(state="completed", stage="completed", pov=None,
                         error_code=None, error_message=None,
                         report_index=report_index, finished_at=now())
        self.write()


class RunLock:
    """One-byte advisory lock; the OS releases it when this process exits."""

    def __init__(self, path):
        self.path = Path(path)
        self._handle = None

    def acquire(self):
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
        except OSError as error:
            handle.close()
            raise Busy(f"lock {self.path} is held by another run") from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n".encode("ascii"))
        handle.flush()
        self._handle = handle

    def release(self):
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


_ACTIVE_CHILDREN = []


def terminate_children():
    """End only the subprocesses this run started."""
    for process in list(_ACTIVE_CHILDREN):
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        except Exception:  # noqa: BLE001 - cleanup must not mask the real error
            pass
        finally:
            if process in _ACTIVE_CHILDREN:
                _ACTIVE_CHILDREN.remove(process)


def run_command(command, *, cwd, log_path, timeout=None, env=None):
    """Run ``command`` without a shell and tee its merged output into ``log_path``."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8", errors="replace")
    log.write("$ " + subprocess.list2cmdline([str(part) for part in command]) + "\n")
    log.write(f"# cwd: {cwd}\n")
    log.flush()
    process = subprocess.Popen(
        [str(part) for part in command], cwd=str(cwd), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, shell=False, env=env, bufsize=0)
    chunks = []

    def pump():
        for raw in iter(process.stdout.readline, b""):
            text = raw.decode("utf-8", "replace")
            chunks.append(text)
            log.write(text)
            log.flush()

    _ACTIVE_CHILDREN.append(process)
    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    try:
        code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        reader.join(10)
        log.write(f"\n[killed after {timeout}s]\n")
        log.close()
        raise Failure("stage_timeout",
                      f"{Path(str(command[0])).name} exceeded {timeout}s; see {log_path}")
    finally:
        reader.join(10)
        if process.stdout:
            process.stdout.close()
        if process in _ACTIVE_CHILDREN:
            _ACTIVE_CHILDREN.remove(process)
    log.write(f"[exit {code}]\n")
    log.close()
    return code, "".join(chunks)


def describe_scope(metadata):
    """Human explanation for an out-of-scope replay (used in the failure message)."""
    players = metadata.get("Players") or []
    races = ", ".join(
        f"id={player.get('PlayerID')} {player.get('AssignedRace')}={player.get('Result')}"
        for player in players) or "none"
    return (f"GameVersion={metadata.get('GameVersion')!r}; players={len(players)} ({races}); "
            f"Duration={metadata.get('Duration')!r}. Supported: 5.0.4.82457, exactly two "
            f"players, Terr vs Prot, one Win and one Loss.")


def inspect_replay(path, log):
    """Read metadata only and reject damaged or out-of-scope input before engine work."""
    data = path.read_bytes()
    if not data:
        raise Failure("invalid_replay", f"{path} is empty")
    try:
        archive = mpyq.MPQArchive(io.BytesIO(data))
        metadata = json.loads(archive.read_file("replay.gamemetadata.json"))
    except Exception as error:  # noqa: BLE001 - any decode failure means damaged input
        raise Failure("invalid_replay",
                      f"Cannot open replay archive: {type(error).__name__}: {error}")
    if not isinstance(metadata, dict):
        raise Failure("invalid_replay", "replay.gamemetadata.json is not an object")
    if not processor.validate_replay(metadata):
        raise Failure("out_of_scope_replay", describe_scope(metadata))
    try:
        game = processor.read_replay(path)
    except Exception as error:  # noqa: BLE001 - missing streams inside the archive
        raise Failure("invalid_replay",
                      f"Cannot read replay streams: {type(error).__name__}: {error}")
    if game is None:
        raise Failure("out_of_scope_replay", describe_scope(metadata))
    record = {
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "game_version": metadata.get("GameVersion"),
        "map_title": metadata.get("Title"),
        "duration_seconds": metadata.get("Duration"),
        "label": int(game["label"]),
        "replay_id": game["replay_id"],
        "player_ids": [int(value) for value in game["player_ids"]],
        "players": [{"id": int(player.get("PlayerID")), "race": player.get("AssignedRace"),
                     "result": player.get("Result")} for player in metadata.get("Players", [])],
    }
    log("replay metadata: " + json.dumps(record))
    return record


def resolve_team_dir(explicit, checkpoint):
    """Find the team project without hardcoding a machine-specific path."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    env_value = os.environ.get("SC2_TEAM_DIR")
    if env_value:
        candidates.append(Path(env_value))
    if checkpoint:
        probe = Path(checkpoint).expanduser().resolve()
        for parent in (probe.parent, *probe.parents):
            if (parent / "viz" / "timeline.py").is_file() and (parent / "config.py").is_file():
                candidates.append(parent)
                break
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if (candidate / "viz" / "timeline.py").is_file() and \
                (candidate / "viz" / "gradcam.py").is_file():
            return candidate
    raise Failure("team_dir_missing",
                  "Pass --team-dir (or set SC2_TEAM_DIR) to the team project that contains "
                  "viz/timeline.py and viz/gradcam.py")


def resolve_checkpoint(value, team_dir):
    """Resolve a checkpoint path, treating relative values as team-dir relative."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(team_dir) / path
    path = path.resolve()
    if not path.is_file():
        raise Failure("checkpoint_missing", f"Checkpoint not found: {path}")
    return path


def check_team_dir(python, team_dir, log_path):
    """Prove the team subprocess imports the team's own modules, not this project's."""
    command = [python, "-c",
               "import json,sys,config,viz;print(json.dumps({'config_root': "
               "str(config.PROJECT_ROOT), 'viz': viz.__file__, 'python': sys.executable}))"]
    code, text = run_command(command, cwd=team_dir, log_path=log_path, timeout=120,
                             env=clean_env())
    if code != 0:
        raise Failure("team_dir_missing", f"Team preflight failed; see {log_path}")
    lines = [line for line in text.strip().splitlines() if line.startswith("{")]
    if not lines:
        raise Failure("team_dir_missing", f"Team preflight printed nothing usable; see {log_path}")
    payload = json.loads(lines[-1])
    expected = Path(team_dir).resolve()
    if Path(payload["config_root"]).resolve() != expected:
        raise Failure("team_dir_missing",
                      f"team-dir loaded config from {payload['config_root']}, expected {expected}")
    if expected not in Path(payload["viz"]).resolve().parents:
        raise Failure("team_dir_missing",
                      f"viz module came from {payload['viz']}, outside {expected}")
    return payload


def cache_rows(handles, cache_root):
    """Map each cache handle to the exact file the engine reads."""
    rows = []
    for handle in handles:
        digest = handle["digest"]
        name = f"{digest}.{handle['file_type']}"
        path = Path(cache_root) / digest[0:2] / digest[2:4] / name
        rows.append({"handle": handle, "path": path, "present": path.is_file()})
    return rows


def fetch_missing(rows, staging, timeout, log):
    """Fetch only the uncached archives that this one replay references."""
    staging = Path(staging)
    staging.mkdir(parents=True, exist_ok=True)
    fetched = []
    for row in rows:
        if row["present"]:
            continue
        handle = row["handle"]
        digest = handle["digest"]
        staged = staging / f"{digest}.{handle['file_type']}"
        record = {"digest": digest, "file_type": handle["file_type"], "attempts": []}
        for url in deps.depot_candidates(handle["region"], digest, handle["file_type"]):
            log(f"fetch {url}")
            attempt = deps.download_archive(url, staged, timeout, log)
            record["attempts"].append(attempt)
            log("  attempt: " + json.dumps(attempt))
            if attempt["error"] is None and attempt["bytes"] > 0:
                break
        if not staged.is_file():
            raise Failure("download_failed",
                          f"No official depot URL returned {digest}.{handle['file_type']}")
        verification = deps.verify_archive(staged, digest)
        record["verification"] = verification
        log("  verify: " + json.dumps(verification))
        if not verification["ok"]:
            raise Failure("download_failed", f"{digest} failed verification: {verification}")
        target = row["path"]
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if deps.sha256_file(target) != verification["sha256"]:
                    raise Failure("cache_conflict",
                                  f"{target} already holds different content; not overwriting")
            else:
                shutil.copy2(staged, target)
        except OSError as error:
            raise Failure("cache_write_failed",
                          f"Cannot write {target}: {type(error).__name__}: {error}")
        record["cache_path"] = str(target)
        fetched.append(record)
    return fetched


PING_RE = re.compile(r"RequestPing -> running engine: game_version=(\S+) base_build=(\d+) "
                     r"data_build=(\d+) data_version=(\S+)")
MAP_RE = re.compile(r"map_name=('.*?'|None) local_map_path=('.*?'|None)")


def parse_engine_check(text, log_path):
    """Extract the measured engine identity from the dependency tool transcript."""
    ping = PING_RE.search(text)
    mapping = MAP_RE.search(text)
    if ping is None:
        raise Failure("engine_check_failed", f"No RequestPing line; see {log_path}")
    if "running engine is the required build: True" not in text:
        raise Failure("engine_check_failed",
                      f"The running engine is not build {processor.ENGINE_VERSION.build_version}; "
                      f"see {log_path}")
    if "ResponseReplayInfo ->" not in text or "ResponseReplayInfo ERROR" in text:
        raise Failure("engine_check_failed",
                      f"The engine could not read the replay; see {log_path}")
    info = {"game_version": ping.group(1), "base_build": int(ping.group(2)),
            "data_build": int(ping.group(3)), "data_version": ping.group(4)}
    if mapping:
        info["map_name"] = mapping.group(1).strip("'")
        info["local_map_path"] = mapping.group(2).strip("'")
    return info


def verify_extraction(processed_dir, expected_replay_id, minimum_frames, log):
    """Check the manifest, every NPZ and the player identity of both POVs."""
    manifest_path = Path(processed_dir) / "manifest.json"
    if not manifest_path.is_file():
        raise Failure("extraction_incomplete", f"{manifest_path} was not written")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise Failure("extraction_incomplete",
                      f"unexpected manifest schema {manifest.get('schema_version')!r}")
    if int(manifest.get("match_count", 0)) != 1 or int(manifest.get("sample_count", 0)) != 2:
        raise Failure("extraction_incomplete",
                      f"expected 1 match and 2 POVs, got match_count="
                      f"{manifest.get('match_count')!r} sample_count="
                      f"{manifest.get('sample_count')!r}")
    povs = []
    for row in manifest["samples"]:
        sample_id = row.get("sample_id")
        if row.get("replay_id") != expected_replay_id:
            raise Failure("extraction_incomplete",
                          f"{sample_id}: replay_id {row.get('replay_id')!r} != "
                          f"{expected_replay_id!r}")
        folder = Path(processed_dir) / str(sample_id)
        for name in ("SpatialFeatures.npz", "TimeInfo.npz", "PlayerMetaInfo.json"):
            if not (folder / name).is_file():
                raise Failure("extraction_incomplete", f"missing {folder / name}")
        with np.load(folder / "SpatialFeatures.npz", allow_pickle=False) as bag:
            shapes = {name: bag[name].shape for name in bag.files}
        expected_layers = set(processor.LAYER_TYPES)
        if set(shapes) != expected_layers:
            raise Failure("extraction_incomplete",
                          f"{sample_id}: layers {sorted(shapes)} != {sorted(expected_layers)}")
        frames = int(row.get("frames", 0))
        if any(shape != (frames,) + MINIMAP_SHAPE for shape in shapes.values()):
            raise Failure("extraction_incomplete", f"{sample_id}: layer shapes {shapes}")
        with np.load(folder / "TimeInfo.npz", allow_pickle=False) as bag:
            seconds = bag["faster_seconds"].astype(float)
            loops = bag["game_loops"].astype(np.int64)
        if len(seconds) != frames or len(loops) != frames:
            raise Failure("extraction_incomplete", f"{sample_id}: time rows do not match frames")
        if not np.all(np.diff(seconds) > 0) or not np.isfinite(seconds).all():
            raise Failure("extraction_incomplete", f"{sample_id}: timestamps are not increasing")
        if frames < minimum_frames:
            raise Failure("extraction_incomplete",
                          f"{sample_id}: {frames} frames < requested {minimum_frames}")
        meta = json.loads((folder / "PlayerMetaInfo.json").read_text(encoding="utf-8"))
        focus = int(row["focus_player_id"])
        if str(focus) not in meta:
            raise Failure("extraction_incomplete", f"{sample_id}: focus player missing from metadata")
        if meta[str(focus)]["race"] != row["focus_player_race"]:
            raise Failure("extraction_incomplete", f"{sample_id}: focus race mismatch")
        povs.append({
            "key": f"player{focus}",
            "sample_id": sample_id,
            "dir": folder,
            "focus_player_id": focus,
            "race": row["focus_player_race"],
            "label": int(row["label"]),
            "frames": frames,
            "result": meta[str(focus)]["result"],
            "players": meta,
            "last_observed_seconds": float(seconds[-2]),
        })
    if len({pov["focus_player_id"] for pov in povs}) != 2:
        raise Failure("extraction_incomplete", "the two POVs share a focus player")
    log("extraction verified: " + json.dumps(
        [{key: pov[key] for key in ("key", "frames", "focus_player_id", "race", "label",
                                    "result", "last_observed_seconds")} for pov in povs]))
    return manifest, povs


def run_timeline(python, run_dir, pov, args):
    """Run team viz.timeline for one POV and validate the exported points."""
    output = Path(run_dir) / "timeline" / pov["key"] / "probability"
    log_path = Path(run_dir) / "logs" / f"predicting-{pov['key']}.log"
    command = [python, "-m", "viz.timeline", str(pov["dir"]), str(args.checkpoint),
               "--device", args.device, "--frames", str(args.frames), "--seed", str(args.seed),
               "--step", str(args.step), "--draws", str(args.draws), "--output", str(output)]
    code, _ = run_command(command, cwd=args.team_dir, log_path=log_path,
                          timeout=args.predict_timeout, env=clean_env())
    if code != 0:
        raise Failure("timeline_failed", f"viz.timeline exited {code}; see {log_path}")
    path = output.with_suffix(".json")
    if not path.is_file():
        raise Failure("timeline_incomplete", f"{path} was not written")
    report = json.loads(path.read_text(encoding="utf-8"))
    for name, expected in (("frames", args.frames), ("seed", args.seed),
                           ("draws", args.draws), ("step", float(args.step))):
        if float(report.get(name)) != float(expected):
            raise Failure("timeline_incomplete",
                          f"{path}: {name}={report.get(name)!r}, expected {expected!r}")
    if report.get("label_definition") not in (FOCUS_LABELS, TERRAN_LABELS):
        raise Failure("timeline_incomplete",
                      f"{path}: unsupported labels {report.get('label_definition')!r}")
    points = report.get("points") or []
    if not points:
        raise Failure("no_prediction_points", f"{path} contains no prediction points")
    previous = -1.0
    for point in points:
        seconds = float(point["seconds"])
        probability = float(point["probability"])
        if not 0.0 <= probability <= 1.0:
            raise Failure("timeline_incomplete", f"{path}: probability {probability} out of range")
        if seconds <= previous:
            raise Failure("timeline_incomplete", f"{path}: cutoffs are not increasing")
        if seconds > pov["last_observed_seconds"] + 1e-6:
            raise Failure("timeline_incomplete",
                          f"{path}: cutoff {seconds} exceeds the last observed frame")
        previous = seconds
    return {"path": path, "report": report, "points": points}


def choose_explanation(points):
    """Pick the point to explain: the later side of the largest adjacent change."""
    if not points:
        raise Failure("no_prediction_points", "the timeline contains no prediction points")
    deltas = [abs(float(points[index + 1]["probability"]) - float(points[index]["probability"]))
              for index in range(len(points) - 1)]
    if not deltas:
        index, rule, delta = 0, "single_point", None
    else:
        index = max(range(len(deltas)), key=deltas.__getitem__) + 1
        rule, delta = "largest_adjacent_change", deltas[index - 1]
    probability = float(points[index]["probability"])
    return {
        "index": index,
        "rule": rule,
        "previous_index": index - 1 if index else None,
        "delta": delta,
        "seconds": float(points[index]["seconds"]),
        "probability": probability,
        "target": 1 if probability >= 0.5 else 0,
        "next_seconds": (float(points[index + 1]["seconds"])
                         if index + 1 < len(points) else None),
    }


def run_gradcam(python, run_dir, pov, choice, point, args):
    """Explain the chosen cutoff and prove it matches the timeline's own sampling."""
    label = f"{pov['key']}-{int(round(choice['seconds']))}s"
    output = Path(run_dir) / "gradcam" / label
    log_path = Path(run_dir) / "logs" / f"explaining-{pov['key']}.log"
    command = [python, "-m", "viz.gradcam", str(pov["dir"]), str(args.checkpoint),
               "--device", args.device, "--frames", str(args.frames), "--seed", str(args.seed),
               "--target", str(choice["target"]),
               "--cutoff", repr(float(choice["seconds"])), "--output", str(output)]
    code, _ = run_command(command, cwd=args.team_dir, log_path=log_path,
                          timeout=args.explain_timeout, env=clean_env())
    if code != 0:
        raise Failure("gradcam_failed", f"viz.gradcam exited {code}; see {log_path}")
    for name in ("snapshot.png", "data.npz", "method.json"):
        if not (output / name).is_file():
            raise Failure("gradcam_incomplete", f"{output / name} was not written")
    method = json.loads((output / "method.json").read_text(encoding="utf-8"))
    if int(method.get("target_class", -1)) != choice["target"]:
        raise Failure("gradcam_mismatch",
                      f"{output}: target {method.get('target_class')!r} != {choice['target']}")
    with np.load(output / "data.npz", allow_pickle=False) as archive:
        cutoffs = archive["cutoff_seconds"].astype(float)
        probabilities = archive["probabilities"].astype(float)
    if len(cutoffs) != 1 or abs(float(cutoffs[0]) - choice["seconds"]) > 1e-6:
        raise Failure("gradcam_mismatch",
                      f"{output}: cutoff {cutoffs.tolist()} != {choice['seconds']}")
    if probabilities.shape != (1, 2):
        raise Failure("gradcam_incomplete", f"{output}: probabilities {probabilities.shape}")
    explained = float(probabilities[0][1])
    if int(args.draws) == 1:
        if abs(explained - choice["probability"]) > 1e-6:
            raise Failure("explanation_mismatch",
                          f"{output}: explained P={explained} but the timeline recorded "
                          f"{choice['probability']} at {choice['seconds']}s")
    else:
        draws = [float(value) for value in point.get("draws") or []]
        if draws and not (min(draws) - 1e-9 <= explained <= max(draws) + 1e-9):
            raise Failure("explanation_mismatch",
                          f"{output}: explained P={explained} outside the draw range {draws}")
    return {"dir": output, "explained_probability": explained,
            "class_probabilities": [float(value) for value in probabilities[0]],
            "method": method}


def run_adapter(python, run_dir, pov, timeline_path, gradcam_dir, choice, args):
    """Adapt the team exports once, then check the label flip and the explanation."""
    output = Path(run_dir) / "review" / pov["key"]
    log_path = Path(run_dir) / "logs" / f"exporting-{pov['key']}.log"
    command = [python, "-m", "analysis.adapt_viz", "--timeline", str(timeline_path),
               "--sample", str(pov["dir"]), "--gradcam", str(gradcam_dir),
               "--output", str(output), "--threshold", str(args.threshold),
               "--merge-seconds", str(args.merge_seconds)]
    code, _ = run_command(command, cwd=PROJECT_ROOT, log_path=log_path,
                          timeout=args.export_timeout, env=clean_env())
    if code != 0:
        raise Failure("adapter_failed", f"analysis.adapt_viz exited {code}; see {log_path}")
    review_path = output / "review.json"
    if not review_path.is_file():
        raise Failure("adapter_incomplete", f"{review_path} was not written")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    explanations = review.get("explanations") or []
    if len(explanations) != 1:
        raise Failure("adapter_incomplete",
                      f"{review_path}: expected one explanation, got {len(explanations)}")
    explanation = explanations[0]
    if abs(float(explanation["cutoff_seconds"]) - choice["seconds"]) > 1e-6:
        raise Failure("adapter_incomplete", f"{review_path}: explanation cutoff mismatch")
    if int(explanation["target_class"]) != choice["target"]:
        raise Failure("adapter_incomplete", f"{review_path}: explanation target mismatch")
    if not (output / explanation["image"]).is_file():
        raise Failure("adapter_incomplete", f"{review_path}: image {explanation['image']} missing")
    expected_label = pov["label"] if pov["race"] == "Terran" else 1 - pov["label"]
    if int(review["label"]) != expected_label:
        raise Failure("adapter_label_mismatch",
                      f"{review_path}: label {review['label']} != focus-player outcome "
                      f"{expected_label} for {pov['race']}")
    if review.get("probability_key") not in ("focus_player_win_probability",
                                             "terran_win_probability"):
        raise Failure("adapter_incomplete",
                      f"{review_path}: unexpected probability_key {review.get('probability_key')!r}")
    return {"dir": output, "path": review_path, "review": review,
            "label": int(review["label"]),
            "label_meaning": review.get("label_meaning")}


def relative(run_dir, path):
    """Posix-style path relative to the run directory, for the report index."""
    return Path(path).resolve().relative_to(Path(run_dir).resolve()).as_posix()


def target_meaning(label_definition, target):
    """Say what the explained class means; never assume class 0 is Protoss."""
    if label_definition == FOCUS_LABELS:
        return "focus player wins" if target == 1 else "focus player loses"
    if label_definition == TERRAN_LABELS:
        return "Terran wins" if target == 1 else "Protoss wins"
    raise Failure("timeline_incomplete", f"Unsupported label definition: {label_definition!r}")


def check_install_and_cache(python, run_dir, input_copy, args, log):
    """Engine presence, cache handles (fetch on request), then the real engine proof."""
    install = deps.describe_install(Path(args.sc2_path))
    log("install: " + json.dumps(install))
    if not install["install_exists"]:
        raise Failure("missing_engine", f"{install['sc2_path']} is not a StarCraft II folder")
    if not install["required_exe_exists"]:
        raise Failure("missing_engine",
                      f"{install['required_exe']} is missing. Run "
                      f"`python -m tools.prepare_replay_dependencies --download` once, then "
                      f"re-run this command.")

    handles = deps.replay_cache_handles(Path(input_copy))
    log("cache handles: " + json.dumps(handles))
    cache_root = Path(args.bnet_cache) if args.bnet_cache else deps.default_bnet_cache()
    rows = cache_rows(handles["handles"], cache_root)
    missing = [row for row in rows if not row["present"]]
    log(f"cache root {cache_root}: {len(rows) - len(missing)}/{len(rows)} present")
    if missing and not args.allow_download_map:
        raise Failure("missing_map_cache",
                      "Missing cache archives: "
                      + ", ".join(str(row["path"]) for row in missing)
                      + ". Re-run with --allow-download-map to fetch only this replay's "
                        "referenced archives from the official depot.")
    fetched = fetch_missing(missing, Path(run_dir) / "cache-staging", args.fetch_timeout, log)
    if fetched:
        log("fetched: " + json.dumps(fetched))
    still_missing = [row for row in rows if not row["path"].is_file()]
    if still_missing:
        raise Failure("missing_map_cache",
                      "Still missing: " + ", ".join(str(row["path"]) for row in still_missing))

    log_path = Path(run_dir) / "logs" / "engine_check.log"
    command = [python, "-m", "tools.prepare_replay_dependencies",
               "--replay", str(input_copy), "--sc2-path", str(args.sc2_path),
               "--engine-build", str(processor.ENGINE_VERSION.build_version),
               "--timeout", str(args.engine_timeout),
               "--log", str(Path(run_dir) / "logs" / "prepare_replay_dependencies.log")]
    code, text = run_command(command, cwd=PROJECT_ROOT, log_path=log_path,
                             timeout=args.engine_timeout + 180, env=clean_env())
    if code != 0:
        raise Failure("engine_check_failed",
                      f"prepare_replay_dependencies exited {code}; see {log_path}")
    engine = parse_engine_check(text, log_path)
    log("engine: " + json.dumps(engine))
    return engine


def extract_povs(python, run_dir, args, replay, log):
    """Re-extract both POVs from the copied replay with the untouched processor."""
    log_path = Path(run_dir) / "logs" / "extracting-processor.log"
    command = [python, "-m", "data_processing.processor", "--pov", "both", "--limit", "1",
               "--workers", "1", "--replays", str(Path(run_dir) / "input"),
               "--output", str(Path(run_dir) / "processed"),
               "--sc2-path", str(args.sc2_path)]
    code, _ = run_command(command, cwd=PROJECT_ROOT, log_path=log_path,
                          timeout=args.extract_timeout, env=clean_env())
    if code != 0:
        raise Failure("extraction_failed", f"processor exited {code}; see {log_path}")
    processed = Path(run_dir) / "processed" / "both"
    return verify_extraction(processed, replay["replay_id"], int(args.frames), log)


def write_report_index(run_dir, params, replay, engine, entries):
    """Write the POV/report index that later backends and POV buttons will read."""
    checkpoint = Path(params["checkpoint"])
    index = {
        "format_version": REPORT_INDEX_FORMAT_VERSION,
        "created_at": now(),
        "replay": {
            "replay_id": replay["replay_id"],
            "sha256": replay["sha256"],
            "bytes": replay["bytes"],
            "game_version": replay["game_version"],
            "map_title": replay["map_title"],
            "duration_seconds": replay["duration_seconds"],
            "terran_win_label": replay["label"],
            "player_ids": replay["player_ids"],
            "stored_path": f"input/{Path(replay['path']).name}",
            "source_path": replay["source_path"],
        },
        "engine": engine,
        "checkpoint": {"path": str(checkpoint), "sha256": deps.sha256_file(checkpoint)},
        "params": params,
        "povs": sorted(entries, key=lambda entry: entry["focus_player_id"]),
        "files": {"status": "status.json", "index": "report_index.json"},
    }
    path = Path(run_dir) / "report_index.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path, index


def run_stages(args, run_dir, journal):
    """Walk the six stages, recording state and verifying each artefact."""
    python = args.python
    run_dir = Path(run_dir)
    vlog = StageLogger(run_dir / "logs" / "validating.log")

    source = Path(args.replay)
    data = source.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    copy_path = run_dir / "input" / source.name
    copy_path.parent.mkdir(parents=True, exist_ok=True)
    copy_path.write_bytes(data)
    if hashlib.sha256(copy_path.read_bytes()).hexdigest() != digest:
        raise Failure("input_copy_failed", f"{copy_path} does not match the source bytes")
    (run_dir / "input.json").write_text(json.dumps({
        "source_path": str(source), "stored_path": f"input/{source.name}",
        "sha256": digest, "bytes": len(data), "copied_at": now(),
    }, indent=2) + "\n", encoding="utf-8")
    vlog(f"input copied: {copy_path} ({len(data)} bytes, sha256={digest})")

    replay = inspect_replay(copy_path, vlog)
    replay["source_path"] = str(source)
    replay["sha256"] = digest

    team_dir = resolve_team_dir(args.team_dir, args.checkpoint)
    checkpoint = resolve_checkpoint(args.checkpoint, team_dir)
    args.team_dir = str(team_dir)
    args.checkpoint = str(checkpoint)
    args.sc2_path = str(Path(args.sc2_path))
    vlog(f"team dir: {team_dir}")
    vlog(f"checkpoint: {checkpoint}")

    params = {
        "python": python,
        "sc2_path": args.sc2_path,
        "team_dir": args.team_dir,
        "checkpoint": args.checkpoint,
        "frames": int(args.frames),
        "step": float(args.step),
        "draws": int(args.draws),
        "seed": int(args.seed),
        "device": args.device,
        "allow_download_map": bool(args.allow_download_map),
        "threshold": float(args.threshold),
        "merge_seconds": float(args.merge_seconds),
        "bnet_cache": str(args.bnet_cache) if args.bnet_cache else str(deps.default_bnet_cache()),
        "engine_build": processor.ENGINE_VERSION.build_version,
        "engine_data_version": processor.ENGINE_VERSION.data_version,
        "created_at": now(),
    }
    (run_dir / "params.json").write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")

    journal.stage("checking_dependencies")
    clog = StageLogger(run_dir / "logs" / "checking_dependencies.log")
    check_team_dir(python, team_dir, run_dir / "logs" / "team_preflight.log")
    engine = check_install_and_cache(python, run_dir, copy_path, args, clog)

    journal.stage("extracting")
    xlog = StageLogger(run_dir / "logs" / "extracting.log")
    manifest, povs = extract_povs(python, run_dir, args, replay, xlog)
    xlog("manifest: " + json.dumps({"schema_version": manifest.get("schema_version"),
                                    "sample_count": manifest.get("sample_count"),
                                    "extraction": manifest.get("extraction")}))

    plog = StageLogger(run_dir / "logs" / "predicting.log")
    elog = StageLogger(run_dir / "logs" / "explaining.log")
    olog = StageLogger(run_dir / "logs" / "exporting.log")
    entries = []
    for pov in povs:
        journal.stage("predicting", pov["key"])
        timeline = run_timeline(python, run_dir, pov, args)
        plog(f"{pov['key']}: {len(timeline['points'])} points, "
             f"last cutoff {timeline['points'][-1]['seconds']}s")

        journal.stage("explaining", pov["key"])
        choice = choose_explanation(timeline["points"])
        point = timeline["points"][choice["index"]]
        gradcam = run_gradcam(python, run_dir, pov, choice, point, args)
        elog(f"{pov['key']}: cutoff {choice['seconds']}s target {choice['target']} "
             f"P={gradcam['explained_probability']:.6f} rule={choice['rule']}")

        journal.stage("exporting", pov["key"])
        adapted = run_adapter(python, run_dir, pov, timeline["path"], gradcam["dir"], choice, args)
        olog(f"{pov['key']}: review label {adapted['label']} ({adapted['label_meaning']})")

        entries.append({
            "player": pov["key"],
            "focus_player_id": pov["focus_player_id"],
            "focus_player_race": pov["race"],
            "result": pov["result"],
            "focus_player_label": adapted["label"],
            "label_meaning": adapted["label_meaning"],
            "sample_id": pov["sample_id"],
            "review": relative(run_dir, adapted["path"]),
            "timeline": relative(run_dir, timeline["path"]),
            "gradcam": relative(run_dir, gradcam["dir"]),
            "points": [{"cutoff_seconds": float(item["seconds"]),
                        "probability": float(item["probability"])}
                       for item in timeline["points"]],
            "explanation": {
                "cutoff_seconds": choice["seconds"],
                "target_class": choice["target"],
                "target_meaning": target_meaning(
                    timeline["report"]["label_definition"], choice["target"]),
                "probability": gradcam["explained_probability"],
                "timeline_probability": choice["probability"],
                "selection_rule": choice["rule"],
                "delta": choice["delta"],
                "previous_cutoff_seconds": (
                    float(timeline["points"][choice["previous_index"]]["seconds"])
                    if choice["previous_index"] is not None else None),
                "next_cutoff_seconds": choice["next_seconds"],
                "image": f"{relative(run_dir, adapted['dir'])}/gradcam_1.png",
            },
        })

    journal.stage("exporting")
    index_path, index = write_report_index(run_dir, params, replay, engine, entries)
    olog(f"report index: {index_path}")
    journal.complete("report_index.json")
    print(json.dumps({"state": "completed", "run_dir": str(run_dir),
                      "report_index": relative(run_dir, index_path),
                      "povs": [{"player": entry["player"],
                                "cutoff_seconds": entry["explanation"]["cutoff_seconds"],
                                "target": entry["explanation"]["target_class"]}
                               for entry in index["povs"]]}, indent=2), flush=True)
    return 0


def _run(args):
    """Create the run directory, then run the stages with honest status reporting."""
    source = Path(args.replay)
    if not source.is_file():
        print(f"error: replay not found: {source}", file=sys.stderr, flush=True)
        return 1
    if args.output is not None:
        run_dir = Path(args.output)
        if run_dir.exists():
            print(f"error: output directory already exists: {run_dir}",
                  file=sys.stderr, flush=True)
            return 2
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base = PROJECT_ROOT / "runs" / f"run-replay-{stamp}"
        run_dir, counter = base, 1
        while run_dir.exists():
            counter += 1
            run_dir = base.with_name(f"{base.name}-{counter}")
    run_dir.mkdir(parents=True, exist_ok=False)
    journal = Journal(run_dir)
    journal.stage("validating")
    journal.write()
    try:
        return run_stages(args, run_dir, journal)
    except Failure as error:
        journal.fail(error.code, error.message)
        print(f"FAILED [{error.code}]: {error.message}", file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        journal.fail("interrupted", "Interrupted by user")
        raise
    except Exception as error:  # noqa: BLE001 - unexpected errors still get a failed status
        import traceback
        journal.fail("internal_error", f"{type(error).__name__}: {error}")
        traceback.print_exc()
        return 1


def make_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay", type=resolve_path, required=True,
                        help="Raw .SC2Replay to analyse")
    parser.add_argument("--output", type=resolve_path, default=None,
                        help="New run directory; a path that already exists is refused "
                             "(default: runs/run-replay-<UTC timestamp>)")
    parser.add_argument("--sc2-path", type=resolve_path, default=SC2_PATH,
                        help="StarCraft II folder containing Versions/")
    parser.add_argument("--team-dir", type=resolve_path, default=None,
                        help="Team project holding viz/timeline.py and viz/gradcam.py "
                             "(default: SC2_TEAM_DIR, else inferred from --checkpoint)")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt",
                        help="Checkpoint path; relative values resolve inside --team-dir")
    parser.add_argument("--frames", type=int, default=50, help="Frames per prediction")
    parser.add_argument("--step", type=float, default=60.0, help="Game seconds per point")
    parser.add_argument("--draws", type=int, default=1, help="Frame draws averaged per point")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--allow-download-map", action="store_true",
                        help="Fetch this replay's missing cache archives from the official "
                             "depot (content hash verified, no map sets or game packages)")
    parser.add_argument("--bnet-cache", type=resolve_path, default=None,
                        help="Battle.net Cache root; default is the engine's own cache")
    parser.add_argument("--fetch-timeout", type=int, default=60,
                        help="Connect/read timeout for depot downloads")
    parser.add_argument("--engine-timeout", type=int, default=300,
                        help="Engine startup timeout for the dependency check")
    parser.add_argument("--predict-timeout", type=int, default=1800)
    parser.add_argument("--explain-timeout", type=int, default=1800)
    parser.add_argument("--extract-timeout", type=int, default=7200)
    parser.add_argument("--export-timeout", type=int, default=600)
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Pivotal-interval threshold passed to the adapter")
    parser.add_argument("--merge-seconds", type=float, default=120.0,
                        help="Maximum span for merging consecutive same-direction interval changes")
    parser.add_argument("--python", default=sys.executable,
                        help="Interpreter used for every subprocess")
    parser.add_argument("--lock-file", type=resolve_path,
                        default=PROJECT_ROOT / "runs" / ".run_replay.lock",
                        help="Single-run lock for this project")
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.frames < 2:
        parser.error("--frames must be at least 2")
    if args.step <= 0:
        parser.error("--step must be greater than zero")
    if args.draws < 1:
        parser.error("--draws must be at least 1")
    if args.merge_seconds <= 0:
        parser.error("--merge-seconds must be greater than zero")
    if not args.output:
        (PROJECT_ROOT / "runs").mkdir(parents=True, exist_ok=True)
    lock = RunLock(args.lock_file)
    try:
        lock.acquire()
    except Busy as error:
        print(f"busy: {error}", file=sys.stderr, flush=True)
        return 3
    try:
        return _run(args)
    finally:
        terminate_children()
        lock.release()


if __name__ == "__main__":
    sys.exit(main())
