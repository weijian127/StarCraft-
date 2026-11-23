"""Check (and optionally request) the official old-replay dependency download.

Two independent modes:

1. Engine check (default): ask the installed client's ``RequestReplayInfo`` whether
   it can parse the replay, optionally with ``download_data=True`` so the client
   fetches the old build itself (verified: this installs ``Versions/Base82457``).
2. ``--fetch-map``: read the replay's own ``m_cacheHandles`` and fetch a *single*
   referenced ``.s2ma`` map archive from Blizzard's official depot into the
   Battle.net cache the engine reads. Nothing is fetched unless ``--map-hash`` is
   given explicitly, so this never downloads a whole map set.

``download_data`` only produces the game binary/data; it does not populate the
map cache. The map archive must come from the depot, addressed by the content
hash recorded in the replay (not by map name).

Local API facts this tool relies on (verified in the installed packages):
  * ``s2clientprotocol.sc2api_pb2.RequestReplayInfo`` has ``download_data``.
  * ``pysc2.lib.remote_controller.RemoteController.replay_info()`` builds that
    request *without* the field, so the tool sends the same request through
    ``controller._client`` (``StarcraftProtocol.send(**kwargs)``) with the field set.
  * ``RunConfig`` reports the *running* engine through ``ResponsePing`` and the
    *replay* through ``ResponseReplayInfo``; the two are reported separately.
  * ``s2protocol`` (installed) decodes ``replay.details`` for build 82457, so the
    ``m_cacheHandles`` layout is the same one ``pysc2/bin/update_battle_net_cache``
    and sc2reader's ``DepotFile`` use: ``type[0:4] | region[4:8] | digest[8:]``.

Depot URL rules: the legacy template
``http://{region}.depot.battle.net:1119/{hash}.{type}`` (pysc2's
``update_battle_net_cache`` and sc2reader's ``DepotFile.url_template``) no longer
resolves. The current non-CN form is
``https://{region}-s2-depot.classic.blizzard.com/{hash}.{type}``. The tool tries
the current form first and records the legacy attempt instead of guessing.
"""

import argparse
import binascii
import hashlib
import io
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mpyq  # noqa: E402
from absl import flags  # noqa: E402
from pysc2 import run_configs  # noqa: E402
from pysc2.run_configs import lib  # noqa: E402
from s2clientprotocol import common_pb2 as common  # noqa: E402
from s2clientprotocol import sc2api_pb2 as sc2_api  # noqa: E402
from s2protocol import versions as s2versions  # noqa: E402

from config import SC2_PATH, resolve_path  # noqa: E402
from data_processing import processor  # noqa: E402

OLD_BUILD = processor.ENGINE_VERSION.build_version          # 82457
OLD_DATA_VERSION = processor.ENGINE_VERSION.data_version     # D2707E26...

DEPOT_CDN_TEMPLATE = "https://{region}-s2-depot.classic.blizzard.com/{hash}.{type}"
DEPOT_LEGACY_TEMPLATE = "http://{region}.depot.battle.net:1119/{hash}.{type}"
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024


def describe_install(sc2_path):
    """Report the engine folders that matter for the required build."""
    versions_dir = sc2_path / "Versions"
    builds = []
    if versions_dir.is_dir():
        for entry in sorted(versions_dir.iterdir()):
            if entry.is_dir() and entry.name.startswith("Base") and entry.name[4:].isdigit():
                builds.append(int(entry.name[4:]))
    target_dir = versions_dir / f"Base{OLD_BUILD}"
    target_exe = target_dir / "SC2_x64.exe"
    return {
        "sc2_path": str(sc2_path),
        "install_exists": sc2_path.is_dir(),
        "builds_present": builds,
        "required_build": OLD_BUILD,
        "required_dir": str(target_dir),
        "required_dir_exists": target_dir.is_dir(),
        "required_exe": str(target_exe),
        "required_exe_exists": target_exe.is_file(),
        "maps_dir": str(sc2_path / "Maps"),
        "maps_dir_exists": (sc2_path / "Maps").is_dir(),
    }


def replay_summary(replay_path):
    """Metadata only (mpyq), plus the processor's own eligibility filter."""
    data = replay_path.read_bytes()
    archive = mpyq.MPQArchive(io.BytesIO(data))
    metadata = json.loads(archive.read_file("replay.gamemetadata.json"))
    accepted = processor.validate_replay(metadata)
    summary = {
        "path": str(replay_path),
        "bytes": len(data),
        "GameVersion": metadata.get("GameVersion"),
        "map_title": metadata.get("Title"),
        "Duration": metadata.get("Duration"),
        "players": [{"id": player.get("PlayerID"), "race": player.get("AssignedRace"),
                     "result": player.get("Result")} for player in metadata.get("Players", [])],
        "passes_processor_filter": accepted,
    }
    if accepted:
        info = processor.read_replay(replay_path)
        summary.update(replay_id=info["replay_id"], terran_win_label=info["label"],
                       player_ids=info["player_ids"])
    return summary, data


def player_rows(replay_info):
    rows = []
    for player in replay_info.player_info:
        inner = player.player_info
        row = {"id": int(inner.player_id)}
        if inner.HasField("type"):
            row["type"] = sc2_api.PlayerType.Name(inner.type)
        try:
            row["race_actual"] = common.Race.Name(inner.race_actual)
        except Exception:
            row["race_actual"] = "unknown"
        if player.HasField("player_result"):
            row["result"] = sc2_api.Result.Name(player.player_result.result)
        rows.append(row)
    return rows


def default_bnet_cache():
    """Same directory the engine reported when the map archive was missing."""
    program_data = os.environ.get("PROGRAMDATA") or r"C:\ProgramData"
    return Path(program_data) / "Blizzard Entertainment" / "Battle.net" / "Cache"


def as_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


def replay_cache_handles(replay_path):
    """Decode ``m_cacheHandles`` the way pysc2/bin/update_battle_net_cache does."""
    data = replay_path.read_bytes()
    archive = mpyq.MPQArchive(io.BytesIO(data))
    header = s2versions.latest().decode_replay_header(
        archive.header["user_data_header"]["content"])
    base_build = header["m_version"]["m_baseBuild"]
    protocol = s2versions.build(base_build)
    details_bytes = archive.read_file(b"replay.details")
    details_source = "replay.details"
    if not details_bytes:
        details_bytes = archive.read_file(b"replay.details.backup")
        details_source = "replay.details.backup"
    if not details_bytes:
        raise RuntimeError("Replay has neither replay.details nor replay.details.backup")
    details = protocol.decode_replay_details(details_bytes)
    handles = []
    for raw in details["m_cacheHandles"]:
        region = raw[4:8].decode("utf-8", "replace").strip("\x00 ")
        if region == "SEA":  # sc2reader: "There is no SEA depot, use US instead"
            region = "US"
        handles.append({
            "file_type": raw[0:4].decode("utf-8", "replace"),
            "region": region,
            "digest": binascii.b2a_hex(raw[8:]).decode("utf-8"),
            "raw_hex": raw.hex(),
        })
    return {
        "base_build": base_build,
        "details_source": details_source,
        "title": as_text(details.get("m_title")),
        "map_file_name": as_text(details.get("m_mapFileName")),
        "is_blizzard_map": bool(details.get("m_isBlizzardMap")),
        "handles": handles,
    }


def depot_candidates(region, digest, file_type):
    """Current CDN first, then the legacy template (recorded, not trusted)."""
    region = (region or "US").upper()
    return [
        DEPOT_CDN_TEMPLATE.format(region=region.lower(), hash=digest, type=file_type),
        DEPOT_LEGACY_TEMPLATE.format(region=region, hash=digest, type=file_type),
    ]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path, expected_digest):
    """Content hash plus a format check: readable MPQ or official marker.

    Ladder maps reference the game's built-in mods through tiny cache entries
    whose whole content is ``Standard Data: <Mod>.SC2Mod``. Those are served by
    the same depot with a matching SHA-256, so they are accepted as a known
    official placeholder instead of being forced through mpyq.
    """
    result = {
        "path": str(path), "bytes": path.stat().st_size, "sha256": None,
        "sha256_matches_handle": False, "mpq_ok": False, "mpq_files": 0,
        "mpq_sample": None, "standard_data_marker": None, "ok": False,
        "problem": None,
    }
    if result["bytes"] == 0:
        result["problem"] = "zero-byte file"
        return result
    result["sha256"] = sha256_file(path)
    result["sha256_matches_handle"] = result["sha256"] == (expected_digest or "").lower()
    with path.open("rb") as handle:
        head = handle.read(64)
    if head.startswith(b"Standard Data: ") and b"\x00" not in head:
        result["standard_data_marker"] = head.decode("ascii", "replace").strip()
    try:
        archive = mpyq.MPQArchive(str(path))
        result["mpq_files"] = len(archive.files)
        listfile = archive.read_file(b"(listfile)")
        if listfile:
            result["mpq_sample"] = f"(listfile) {len(listfile)} bytes"
    except Exception as error:  # noqa: BLE001 - report whatever mpyq raises
        result["problem"] = f"{type(error).__name__}: {error}"
    result["mpq_ok"] = result["mpq_files"] > 0 and result["mpq_sample"] is not None
    result["ok"] = result["sha256_matches_handle"] and (
        result["mpq_ok"] or result["standard_data_marker"] is not None)
    return result


def download_archive(url, dest, timeout_seconds, log):
    """Stream one depot file to ``<dest>.part``, sanity-check it, then rename."""
    partial = dest.with_name(dest.name + ".part")
    record = {"url": url, "status": None, "final_url": None, "content_type": None,
              "content_length": None, "bytes": 0, "error": None}
    request = urllib.request.Request(url, headers={"User-Agent": "sc2-replay-deps/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            record["status"] = response.status
            record["final_url"] = response.geturl()
            record["content_type"] = response.headers.get("Content-Type")
            record["content_length"] = response.headers.get("Content-Length")
            written = 0
            with partial.open("wb") as handle:
                while True:
                    chunk = response.read(256 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > MAX_ARCHIVE_BYTES:
                        raise RuntimeError(
                            f"response exceeded {MAX_ARCHIVE_BYTES} bytes; aborted")
                    handle.write(chunk)
            record["bytes"] = written
    except urllib.error.HTTPError as error:
        record["status"] = error.code
        record["error"] = f"HTTPError {error.code} {error.reason}"
        partial.unlink(missing_ok=True)
        return record
    except Exception as error:  # noqa: BLE001 - network/timeout/DNS
        record["error"] = f"{type(error).__name__}: {error}"
        partial.unlink(missing_ok=True)
        return record
    with partial.open("rb") as handle:
        head = handle.read(256).lstrip().lower()
    if head.startswith(b"<") or b"<html" in head or b"<!doctype" in head:
        record["error"] = "response looks like an HTML page, not an MPQ archive"
        partial.unlink(missing_ok=True)
        return record
    partial.replace(dest)
    return record


def run_fetch_map(args, log):
    """List the replay's cache handles; fetch only ``--map-hash`` when given."""
    info = replay_cache_handles(args.replay)
    log("cache handles: " + json.dumps(info))
    if not info["handles"]:
        log("FAILED: replay.details lists no m_cacheHandles")
        return 2
    cache_root = args.bnet_cache or default_bnet_cache()
    staging = args.staging_dir
    staging.mkdir(parents=True, exist_ok=True)
    log(f"battle.net cache root: {cache_root}")
    log(f"staging dir: {staging}")

    selected = (args.map_hash or "").lower() or None
    digests = [handle["digest"] for handle in info["handles"]]
    if args.all_handles and selected is not None:
        log("FAILED: --all-handles and --map-hash are mutually exclusive")
        return 2
    if selected is not None and selected not in digests:
        log(f"FAILED: {selected} is not one of this replay's cache handles")
        return 2
    if args.all_handles:
        log(f"--all-handles: this replay references {len(digests)} archives; "
            f"only uncached ones will be fetched")

    exit_code = 0
    for handle in info["handles"]:
        digest = handle["digest"]
        name = f"{digest}.{handle['file_type']}"
        cache_path = cache_root / digest[0:2] / digest[2:4] / name
        log(f"handle type={handle['file_type']} region={handle['region']} hash={digest} "
            f"cached={cache_path.is_file()} cache_path={cache_path}")
        if not args.all_handles and (selected is None or digest != selected):
            continue
        if cache_path.is_file():
            log("  already cached; verifying the existing file")
            log("  verify(cache): " + json.dumps(verify_archive(cache_path, digest)))
            continue

        staged = staging / name
        for url in depot_candidates(handle["region"], digest, handle["file_type"]):
            log(f"  GET {url} (timeout {args.fetch_timeout}s)")
            record = download_archive(url, staged, args.fetch_timeout, log)
            log("  attempt: " + json.dumps(record))
            if record["error"] is None and record["bytes"] > 0:
                break
        if not staged.is_file():
            log("  FAILED: no candidate URL produced a file; nothing installed")
            exit_code = 3
            continue

        verification = verify_archive(staged, digest)
        log("  verify(staging): " + json.dumps(verification))
        if not verification["ok"]:
            log("  NOT installing: verification failed (staged file kept for inspection)")
            exit_code = 4
            continue

        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if cache_path.exists():
                existing = verify_archive(cache_path, digest)
                if existing["sha256"] != verification["sha256"]:
                    log("  NOT overwriting: cache holds different content: "
                        + json.dumps(existing))
                    exit_code = 5
                    continue
                log("  cache already holds identical content; copy skipped")
            else:
                shutil.copy2(staged, cache_path)
                log(f"  installed: {cache_path}")
        except OSError as error:
            log(f"  FAILED to install into cache: {type(error).__name__}: {error}")
            log("  staged file kept; no elevation attempted")
            exit_code = 6
            continue

        installed = verify_archive(cache_path, digest)
        log("  verify(cache): " + json.dumps(installed))
        if not installed["ok"]:
            exit_code = 6
    if selected is None and not args.all_handles:
        log("no --map-hash/--all-handles given: listed handles only, downloaded nothing")
    return exit_code


def launch_version(build):
    """PySC2 Version used to start ``build``.

    The old engine must be told the old data pack: sc_process passes
    ``version.data_version`` as ``-dataVersion``. Leaving it None makes SC2 use the
    installed (current) game data with the old binary, which is not a valid check.
    """
    if build == OLD_BUILD:
        return lib.Version("5.0.4", build, OLD_DATA_VERSION, "SC2_x64.exe")
    return lib.Version("latest", build, None, None)


def start_engine(sc2_path, timeout_seconds, log, build=None):
    """Launch an installed client build; never a build that is not on disk."""
    os.environ["SC2PATH"] = str(sc2_path)
    if not flags.FLAGS.is_parsed():
        flags.FLAGS(["prepare-replay-deps"], known_only=True)
    install = describe_install(sc2_path)
    if not install["builds_present"]:
        raise RuntimeError(f"No Versions/Base* folder under {sc2_path}")
    build = build or max(install["builds_present"])
    if build not in install["builds_present"]:
        raise RuntimeError(f"Base{build} is not installed under {sc2_path}")
    version = launch_version(build)
    config = run_configs.get(version=version)
    log(f"launching installed build Base{build} (data_dir={config.data_dir}, "
        f"data_version={version.data_version or 'engine default'})")
    engine = config.start(want_rgb=False, window_loc=(50, 50), timeout_seconds=timeout_seconds)
    return engine, build


def make_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay", type=resolve_path, required=True,
                        help="Raw .SC2Replay to inspect")
    parser.add_argument("--sc2-path", type=resolve_path, default=SC2_PATH,
                        help="StarCraft II folder containing Versions/")
    parser.add_argument("--download", action="store_true",
                        help="Set RequestReplayInfo.download_data=True (default: check only)")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Engine startup timeout in seconds (integer)")
    parser.add_argument("--engine-build", type=int, default=None,
                        help="Installed Base build to launch (default: highest present)")
    parser.add_argument("--fetch-map", action="store_true",
                        help="Read m_cacheHandles from the replay and (with --map-hash) "
                             "fetch that archive from the official depot; no engine launch")
    parser.add_argument("--map-hash", default=None,
                        help="Exact 64-hex digest from m_cacheHandles to fetch")
    parser.add_argument("--all-handles", action="store_true",
                        help="Fetch every uncached archive that *this* replay references "
                             "(its own m_cacheHandles only)")
    parser.add_argument("--bnet-cache", type=Path, default=None,
                        help="Battle.net Cache root (default: %%PROGRAMDATA%%/Blizzard "
                             "Entertainment/Battle.net/Cache)")
    parser.add_argument("--staging-dir", type=Path,
                        default=PROJECT_ROOT / "runs" / "replay-deps" / "staging",
                        help="Where verified downloads are staged before installation")
    parser.add_argument("--fetch-timeout", type=int, default=60,
                        help="Connect/read timeout in seconds for depot downloads")
    parser.add_argument("--log", type=resolve_path, default=None,
                        help="Log file; default runs/replay-deps/prepare_replay_dependencies.log")
    return parser


def main():
    args = make_parser().parse_args()
    log_path = args.log or (PROJECT_ROOT / "runs" / "replay-deps"
                            / "prepare_replay_dependencies.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(message):
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        lines.append(line)

    engine = controller = None
    exit_code = 0
    try:
        if args.fetch_map:
            log("mode: fetch map archive from the official depot (no engine launch)")
            log(f"replay: {args.replay}")
            exit_code = run_fetch_map(args, log)
            return exit_code
        log("mode: " + ("download requested" if args.download
                        else "check only (download_data=False)"))
        log(f"sc2 path: {args.sc2_path}")
        install_before = describe_install(args.sc2_path)
        log("install before: " + json.dumps(install_before))
        if not args.replay.is_file():
            raise FileNotFoundError(f"Replay not found: {args.replay}")
        summary, data = replay_summary(args.replay)
        log("replay metadata: " + json.dumps(summary))

        engine, build = start_engine(args.sc2_path, args.timeout, log,
                                     build=args.engine_build)
        controller = engine.controller
        ping = controller.ping()
        log(f"RequestPing -> running engine: game_version={ping.game_version} "
            f"base_build={ping.base_build} data_build={ping.data_build} "
            f"data_version={ping.data_version}")
        log(f"running engine is the required build: {ping.base_build == OLD_BUILD}")

        request = sc2_api.RequestReplayInfo(replay_data=data,
                                           download_data=bool(args.download))
        log(f"RequestReplayInfo: download_data={request.download_data} "
            f"replay_bytes={len(request.replay_data)}")
        # PySC2's RemoteController.replay_info() sends this same request without the
        # download_data field, so the protocol client is used directly.
        replay_info = controller._client.send(replay_info=request)
        if replay_info.HasField("error"):
            log("ResponseReplayInfo ERROR: "
                f"{sc2_api.ResponseReplayInfo.Error.Name(replay_info.error)} | "
                f"details={replay_info.error_details!r}")
            exit_code = 2
        else:
            log(f"ResponseReplayInfo -> replay: game_version={replay_info.game_version} "
                f"base_build={replay_info.base_build} data_build={replay_info.data_build} "
                f"data_version={replay_info.data_version}")
            log(f"map_name={replay_info.map_name!r} "
                f"local_map_path={replay_info.local_map_path!r}")
            log(f"duration: loops={replay_info.game_duration_loops} "
                f"seconds={replay_info.game_duration_seconds:.1f}")
            log("players: " + json.dumps(player_rows(replay_info)))
            matches = (replay_info.base_build == OLD_BUILD
                       and replay_info.data_version.upper() == OLD_DATA_VERSION)
            log(f"replay matches required build and data_version: {matches}")

        install_after = describe_install(args.sc2_path)
        log("install after: required_exe_exists="
            f"{install_after['required_exe_exists']} "
            f"builds_present={json.dumps(install_after['builds_present'])}")
        if args.download and not install_after["required_exe_exists"]:
            log("NOTE: the required binary is still absent. A download may still be in "
                "progress; re-run this check later.")
    except Exception as error:
        log(f"FAILED: {type(error).__name__}: {error}")
        exit_code = exit_code or 1
    finally:
        if controller is not None:
            try:
                controller.quit()
                log("controller: RequestQuit sent")
            except Exception as error:
                log(f"controller.quit failed: {type(error).__name__}: {error}")
        if engine is not None:
            try:
                engine.close()
                log("engine process closed")
            except Exception as error:
                log(f"engine.close failed: {type(error).__name__}: {error}")
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n===== run {datetime.now(timezone.utc).isoformat()} =====\n")
            handle.write("\n".join(lines) + "\n")
        print(f"log: {log_path}", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
