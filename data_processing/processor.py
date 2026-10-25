"""Convert SC2 5.0.4 Terran-versus-Protoss replays into minimap arrays.

Labels: 0 = Protoss wins, 1 = Terran wins.
Each saved layer has shape [frames, 128, 128].
"""

import argparse
import hashlib
import io
import json
import multiprocessing
import os
import signal
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing.util import Finalize

import mpyq
import numpy as np
from absl import flags
from pysc2 import run_configs
from pysc2.lib import features, protocol, remote_controller
from pysc2.run_configs import lib
from s2clientprotocol import common_pb2 as races
from s2clientprotocol import sc2api_pb2 as sc2_api

from config import PROCESSED_DIR, REPLAY_DIR, SC2_PATH, resolve_path

REPLAY_VERSION = "5.0.4.82457"
ENGINE_VERSION = lib.Version(
    game_version="5.0.4",
    build_version=82457,
    data_version="D2707E265785612D12B381AF6ED9DBF4",
    binary="SC2_x64.exe",
)
MINIMAP_SIZE = 128
STEP_SIZE = 8
GAME_LOOPS_PER_SECOND = 22.4
# single = only one of the player's POV
POV_MODES = ("single", "both", "fogless")

LAYER_TYPES = {
    "visibility_map": np.uint8,
    "player_relative": np.uint8,
    "unit_type": np.uint16,
}
CATEGORY_LIMITS = {
    "visibility_map": 4,
    "player_relative": 5,
    "unit_type": 65536,
}


# Each spawned worker owns these resources; they are never shared between processes.
_worker_config = None
_worker_controller = None


def validate_replay(metadata) -> bool:
    """We only deal with matches that are:
        1. version 5.0.4
        2. 1v1
        3. Terran vs. Protoss
        4. Ended in Win/Loss

    Args:
        metadata (dict): Decoded replay.gamemetadata.json from the replay archive.
    """
    if metadata.get("GameVersion") != REPLAY_VERSION:
        return False

    players = metadata.get("Players", [])
    if len(players) != 2:
        return False

    player_races = {player.get("AssignedRace") for player in players}
    if player_races != {"Terr", "Prot"}:
        return False

    player_results = {player.get("Result") for player in players}
    if player_results != {"Win", "Loss"}:
        return False

    if metadata.get("Duration", 0) <= 0:
        return False

    return True


def read_replay(path):
    """Read a game's ID and label. Return None when it should not be in our dataset.

    Args:
        path (Path): Path to a raw .SC2Replay file.
    """
    replay_bytes = path.read_bytes()
    archive = mpyq.MPQArchive(io.BytesIO(replay_bytes))
    metadata = json.loads(archive.read_file("replay.gamemetadata.json"))

    if not validate_replay(metadata):
        return None

    # Hash the game contents so renamed or alternate recordings can be deduplicated.
    game_hash = hashlib.sha256(metadata["DataVersion"].encode("ascii"))
    for filename in ("replay.initData", "replay.game.events"):
        contents = archive.read_file(filename)
        if not contents:
            contents = archive.read_file(filename + ".backup")
        if not contents:
            raise ValueError(f"Missing {filename}")

        # Include lengths to distinguish the boundaries between the two streams.
        game_hash.update(len(contents).to_bytes(8, "little"))
        game_hash.update(contents)

    players = metadata.get("Players", [])
    terran_player = next(player for player in players if player["AssignedRace"] == "Terr")
    terran_won = terran_player["Result"] == "Win"
    return {
        "path": path,
        "replay_id": game_hash.hexdigest(),
        "label": int(terran_won),
        "player_ids": sorted(int(player["PlayerID"]) for player in players),
    }


def validate_replay_info(replay_info):
    """Check engine compatibility and human participants before playback.

    Args:
        replay_info (ResponseReplayInfo): Replay details returned by the SC2 engine.
    """
    if replay_info.HasField("error"):
        raise ValueError("SC2 could not read this replay")
    if replay_info.base_build != ENGINE_VERSION.build_version:
        raise ValueError("Replay engine version mismatch")
    if replay_info.data_version.upper() != ENGINE_VERSION.data_version:
        raise ValueError("Replay data version mismatch")

    # Version, matchup, and outcome were already filtered by validate_replay.
    for player in replay_info.player_info:
        if (
            not player.player_info.HasField("type")
            or player.player_info.type != sc2_api.Participant
        ):
            raise ValueError("Expected human participants")


def validate_minimap(layer, name):
    """Check one decoded layer's presence, dimensions, and categorical values.

    Args:
        layer (ndarray or None): Decoded categorical minimap, expected shape [128, 128].
        name (str): Layer name from LAYER_TYPES, used to check allowed category IDs.
    """
    if layer is None:
        raise ValueError(f"Missing {name} minimap")
    if layer.shape != (MINIMAP_SIZE, MINIMAP_SIZE):
        raise ValueError(f"Incorrect {name} minimap size")
    if np.any(layer < 0) or np.any(layer >= CATEGORY_LIMITS[name]):
        raise ValueError(f"Invalid categories in {name}")


def extract_replay(
    controller, run_config, game, replay_info, focus_player_id=1, disable_fog=False
):
    """Simulate a game and collect the three minimap layers every eight game loops.

    Args:
        controller (RemoteController): Connection to a running SC2 engine.
        run_config (RunConfig): PySC2 configuration used to locate map files.
        game (dict): Record from read_replay; supplies the replay path.
        replay_info (ResponseReplayInfo): Engine replay details checked by preprocess.
        focus_player_id (int): Player ID used for visibility and self/enemy ownership.
        disable_fog (bool): Reveal enemy units outside the focus player's normal vision.
    """
    replay_bytes = game["path"].read_bytes()
    players = list(replay_info.player_info)
    if focus_player_id not in {player.player_info.player_id for player in players}:
        raise ValueError(f"Focus player {focus_player_id} is not in this replay")

    # setup the view for the SC2 in-engine replay
    options = sc2_api.InterfaceOptions()
    camera = options.feature_layer
    camera.width = 24
    camera.resolution.x = MINIMAP_SIZE
    camera.resolution.y = MINIMAP_SIZE
    camera.minimap_resolution.x = MINIMAP_SIZE
    camera.minimap_resolution.y = MINIMAP_SIZE
    camera.crop_to_playable_area = False
    camera.allow_cheating_layers = True  # Required for the author's unit_type minimap.

    map_bytes = None
    if replay_info.local_map_path:
        map_bytes = run_config.map_data(replay_info.local_map_path)
    request = sc2_api.RequestStartReplay(
        replay_data=replay_bytes,
        map_data=map_bytes,
        options=options,
        observed_player_id=focus_player_id,
        disable_fog=disable_fog,
    )
    controller.start_replay(request)
    controller.step(1)

    # run the game simulation using the replay_data
    layer_frames = {name: [] for name in LAYER_TYPES}
    game_loops = [] # game loops are "timestamps" measured in SC2 sim ticks (~.357 seconds)
    while True:
        response = controller.observe(disable_fog=disable_fog)
        observation = response.observation
        current_loop = int(observation.game_loop)
        if game_loops and current_loop <= game_loops[-1]:
            raise ValueError("Replay stopped advancing")

        minimaps = observation.feature_layer_data.minimap_renders
        for name, storage_type in LAYER_TYPES.items(): # create all spatial feature layers
            packed_layer = getattr(minimaps, name)
            layer = features.Feature.unpack_layer(packed_layer)
            validate_minimap(layer, name)
            layer_frames[name].append(layer.astype(storage_type, copy=True))

        game_loops.append(current_loop)
        if response.player_result: # when the game ends, stop the sim
            break
        if current_loop > replay_info.game_duration_loops + STEP_SIZE:
            raise ValueError("Replay reached its duration without a result")
        controller.step(STEP_SIZE)

    spatial_layers = {name: np.stack(frames) for name, frames in layer_frames.items()}

    race_names = {races.Terran: "Terran", races.Protoss: "Protoss"}
    result_names = {sc2_api.Victory: "Victory", sc2_api.Defeat: "Defeat"}
    player_metadata = {}
    for player in players:
        player_id = str(player.player_info.player_id)
        player_metadata[player_id] = {
            "race": race_names[player.player_info.race_actual],
            "result": result_names[player.player_result.result],
        }
    return spatial_layers, np.asarray(game_loops, dtype=np.int64), player_metadata


def save_game(
    output_directory,
    game,
    spatial_layers,
    game_loops,
    player_metadata,
    focus_player_id,
    pov_mode,
):
    """Save one POV. Keep its sample ID separate from the shared match ID.

    Args:
        output_directory (Path): Output folder for the selected POV mode.
        game (dict): Record from read_replay containing the shared match ID and label.
        spatial_layers (dict[str, ndarray]): Layer arrays, each shaped [frames, 128, 128].
        game_loops (ndarray): One-dimensional game-loop numbers aligned with the frames.
        player_metadata (dict): Race and result for each player, keyed by string player ID.
        focus_player_id (int): Player ID whose perspective these arrays represent.
        pov_mode (str): Extraction mode: single, both, or fogless.
    """
    replay_id = game["replay_id"]
    sample_id = f"{replay_id}_{pov_mode}_player_{focus_player_id}"
    game_directory = output_directory / sample_id
    game_directory.mkdir(exist_ok=True)

    np.savez_compressed(game_directory / "SpatialFeatures.npz", **spatial_layers)
    seconds = game_loops.astype(np.float64) / GAME_LOOPS_PER_SECOND
    np.savez_compressed(
        game_directory / "TimeInfo.npz", game_loops=game_loops, faster_seconds=seconds
    )
    metadata_path = game_directory / "PlayerMetaInfo.json"
    metadata_path.write_text(json.dumps(player_metadata), encoding="utf-8")

    return {
        "replay_id": replay_id,
        "sample_id": sample_id,
        "focus_player_id": focus_player_id,
        "focus_player_race": player_metadata[str(focus_player_id)]["race"],
        "pov_mode": pov_mode,
        "disable_fog": pov_mode == "fogless",
        "label": game["label"],
        "frames": len(game_loops),
        "source_path": f"{sample_id}/SpatialFeatures.npz",
        "timing_path": f"{sample_id}/TimeInfo.npz",
    }


def start_worker(sc2_path):
    """Start a reusable SC2 engine in this worker process.

    Args:
        sc2_path (Path): SC2 installation folder containing Versions.
    """
    global _worker_config, _worker_controller
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ["SC2PATH"] = str(sc2_path.resolve())
    # PySC2 uses absl flags internally, even though our command uses argparse.
    if not flags.FLAGS.is_parsed():
        flags.FLAGS(["preprocess"], known_only=True)

    _worker_config = run_configs.get(version=ENGINE_VERSION)
    engine = _worker_config.start(want_rgb=False, window_loc=(50, 50), timeout_seconds=120)
    # Close the engine when the worker process exits.
    Finalize(engine, engine.close, exitpriority=0)
    _worker_controller = engine.controller


def process_replay(game, output, pov_mode):
    """Extract and save one match using this worker's SC2 engine.

    Args:
        game (dict): Eligible match returned by read_replay.
        output (Path): Output folder for the selected POV mode.
        pov_mode (str): single, both, or fogless; both views stay together.
    """
    try:
        # Engine-only checks run once per match, before extracting either POV.
        replay_info = _worker_controller.replay_info(game["path"].read_bytes())
        validate_replay_info(replay_info)
        focus_player_ids = game["player_ids"]
        if pov_mode != "both":
            focus_player_ids = focus_player_ids[:1]

        recordings = []
        spatial_hash = hashlib.sha256()
        for focus_player_id in focus_player_ids:
            spatial_layers, game_loops, player_metadata = extract_replay(
                _worker_controller, _worker_config, game, replay_info, focus_player_id,
                disable_fog=pov_mode == "fogless",
            )
            if not any(np.any(frame) for frame in spatial_layers["unit_type"]):
                raise ValueError("Unit-type minimap is empty")
            spatial_hash.update(str(focus_player_id).encode())
            for layer in spatial_layers.values():
                spatial_hash.update(str(layer.shape).encode())
                spatial_hash.update(layer.tobytes())
            recordings.append(
                {
                    "focus_player_id": focus_player_id,
                    "spatial_layers": spatial_layers,
                    "game_loops": game_loops,
                    "player_metadata": player_metadata,
                }
            )

        entries = [
            save_game(output, game, pov_mode=pov_mode, **recording)
            for recording in recordings
        ]
        return entries, spatial_hash.hexdigest(), None
    except protocol.ConnectionError:
        raise  # Stop if the SC2 process has disconnected.
    except (OSError, ValueError, protocol.ProtocolError, remote_controller.RequestError) as error:
        return [], None, str(error)


def find_games(replays, limit=None):
    """Find eligible unique matches in a consistent order.

    Args:
        replays (Path): Folder to search recursively for raw .SC2Replay files.
        limit (int or None): Maximum eligible matches to return; None means all.
    """
    replay_paths = [path for path in replays.rglob("*") if path.suffix.lower() == ".sc2replay"]
    replay_paths.sort()
    if not replay_paths:
        raise ValueError(f"No .SC2Replay files found in {replays}")

    games = []
    seen_replay_ids = set()
    for path in replay_paths:
        try:
            game = read_replay(path)
        except (OSError, ValueError, KeyError) as error:
            print(f"Skipping {path.name}: {error}", flush=True)
            continue
        if game is None:
            continue
        if game["replay_id"] in seen_replay_ids:
            continue
        games.append(game)
        seen_replay_ids.add(game["replay_id"])

    print(f"{len(replay_paths)} replays; {len(games)} eligible unique games", flush=True)
    if limit is not None:
        games = games[:limit]
    if not games:
        raise ValueError("No eligible games")

    return games


def run_workers(games, output, sc2_path, pov_mode, workers):
    """Process matches in parallel and combine their metadata in input order.

    Args:
        games (list[dict]): Eligible matches in their original input order.
        output (Path): Output folder for the selected POV mode.
        sc2_path (Path): SC2 installation folder containing Versions.
        pov_mode (str): single, both, or fogless.
        workers (int): Maximum simultaneous SC2 extraction processes.
    """
    worker_count = min(workers, len(games))
    print(f"Using {worker_count} extraction worker(s)", flush=True)
    executor = ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=start_worker,
        initargs=(sc2_path,),
    )
    task = partial(process_replay, output=output, pov_mode=pov_mode)
    manifest_entries = []
    seen_spatial_hashes = set()
    try:
        # Accept results in input order so parallel runs keep the same dataset.
        for index, (entries, fingerprint, error) in enumerate(executor.map(task, games), start=1):
            game = games[index - 1]
            if error is not None:
                print(f"Skipping {game['path'].name}: {error}", flush=True)
            # Deduplicate complete matches, retaining both legitimate POVs together.
            elif fingerprint in seen_spatial_hashes:
                print(f"Skipping {game['path'].name}: identical spatial recording", flush=True)
                for entry in entries:
                    folder = output / entry["sample_id"]
                    if folder.resolve().parent != output.resolve():
                        raise ValueError("Sample directory is outside the output folder")
                    for filename in ("SpatialFeatures.npz", "TimeInfo.npz", "PlayerMetaInfo.json"):
                        (folder / filename).unlink(missing_ok=True)
                    if not any(folder.iterdir()):
                        folder.rmdir()
            else:
                manifest_entries.extend(entries)
                seen_spatial_hashes.add(fingerprint)
            print(
                f"{index}/{len(games)} matches checked; {len(manifest_entries)} POVs accepted",
                flush=True,
            )
    finally:
        # Let each worker finish its current match and close its own SC2 engine.
        executor.shutdown(wait=True, cancel_futures=True)
    return manifest_entries


def save_manifest(manifest_entries, output, pov_mode):
    """Write the accepted samples and extraction settings to manifest.json.

    Args:
        manifest_entries (list[dict]): Accepted sample metadata returned by the workers.
        output (Path): Output folder for the selected POV mode.
        pov_mode (str): single, both, or fogless.
    """
    manifest_path = output / "manifest.json"
    if not manifest_entries:
        raise ValueError("No games were extracted; see the playback errors above")
    manifest = {
        "schema_version": 2,
        "label_definition": "1 = Terran wins, 0 = Protoss wins",
        "match_count": len({entry["replay_id"] for entry in manifest_entries}),
        "sample_count": len(manifest_entries),
        "samples": manifest_entries,
        "extraction": {
            "game_version": REPLAY_VERSION,
            "step_mul": STEP_SIZE,
            "pov_mode": pov_mode,
            "focus_policy": "both players"
            if pov_mode == "both"
            else "player 1",
            "disable_fog": pov_mode == "fogless",
            "allow_cheating_layers": True,
            "minimap_resolution": [MINIMAP_SIZE, MINIMAP_SIZE],
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(manifest_entries)} POVs to {manifest_path}", flush=True)


def preprocess(replays, output, sc2_path, limit=None, pov_mode="single", workers=1):
    """Find eligible games, extract them, and write the dataset manifest.

    Args:
        replays (Path): Folder to search recursively for raw .SC2Replay files.
        output (Path): Parent output folder; a subfolder is created for the POV mode.
        sc2_path (Path): SC2 installation folder containing the Versions directory.
        limit (int or None): Maximum eligible unique matches to attempt; None means all.
        pov_mode (str): single = first player with fog; both = both players with fog;
            fogless = first player without fog. The first player has the lowest player ID.
        workers (int): Maximum simultaneous extraction processes, each with its own SC2 engine.
    """
    if pov_mode not in POV_MODES:
        raise ValueError(f"Choose one of these POV modes: {POV_MODES}")
    if workers < 1:
        raise ValueError("workers must be positive")
    output = output / pov_mode

    games = find_games(replays, limit)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Output already has a manifest; choose a new output directory")

    manifest_entries = run_workers(games, output, sc2_path, pov_mode, workers)
    save_manifest(manifest_entries, output, pov_mode)
    return manifest_entries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--replays", type=resolve_path, default=REPLAY_DIR, help="Folder of raw .SC2Replay files"
    )
    parser.add_argument(
        "--output",
        type=resolve_path,
        default=PROCESSED_DIR,
        help="Parent folder; each POV mode gets its own subdirectory",
    )
    parser.add_argument(
        "--sc2-path", type=resolve_path, default=SC2_PATH, help="SC2 folder containing Versions/"
    )
    parser.add_argument("--limit", type=int, help="Process only the first N eligible games")
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of simultaneous SC2 extraction workers"
    )
    parser.add_argument(
        "--pov",
        choices=POV_MODES,
        default="single",
        help="single: Player 1 with fog; both: both players with fog; fogless: Player 1 without fog",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    preprocess(args.replays, args.output, args.sc2_path, args.limit, args.pov, args.workers)
