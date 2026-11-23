"""Render one RGB game-view frame from the original replay at an exact game loop.

Not a model input. The caller holds the backend heavy-task lock.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from absl import flags
from PIL import Image
from pysc2 import run_configs
from pysc2.lib import features
from s2clientprotocol import sc2api_pb2 as sc2_api

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import SC2_PATH
from data_processing import processor


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--player", type=int, required=True)
    parser.add_argument("--game-loop", type=int, required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args(argv)

    os.environ["SC2PATH"] = str(SC2_PATH.resolve())
    if not flags.FLAGS.is_parsed():
        flags.FLAGS(["render_game_view"], known_only=True)
    run_config = run_configs.get(version=processor.ENGINE_VERSION)
    engine = run_config.start(want_rgb=True, window_loc=(50, 50), timeout_seconds=120)
    try:
        controller = engine.controller
        replay_bytes = args.replay.read_bytes()
        info = controller.replay_info(replay_bytes)
        processor.validate_replay_info(info)
        options = sc2_api.InterfaceOptions()
        options.render.resolution.x = 640
        options.render.resolution.y = 480
        options.render.minimap_resolution.x = 128
        options.render.minimap_resolution.y = 128
        options.render.crop_to_playable_area = False
        options.feature_layer.width = 24
        options.feature_layer.resolution.x = 128
        options.feature_layer.resolution.y = 128
        options.feature_layer.minimap_resolution.x = 128
        options.feature_layer.minimap_resolution.y = 128
        options.feature_layer.crop_to_playable_area = False
        options.feature_layer.allow_cheating_layers = True
        map_data = run_config.map_data(info.local_map_path) if info.local_map_path else None
        request = sc2_api.RequestStartReplay(
            replay_data=replay_bytes, map_data=map_data, options=options,
            observed_player_id=int(args.player), disable_fog=False)
        controller.start_replay(request)
        controller.step(1)
        observation = controller.observe()
        current = int(observation.observation.game_loop)
        if args.game_loop > current:
            controller.step(int(args.game_loop) - current)
            observation = controller.observe()
        actual = int(observation.observation.game_loop)
        render = observation.observation.render_data.map
        if not render.data:
            raise RuntimeError("render_data.map is empty; RGB rendering was not returned")
        rgb = features.Feature.unpack_rgb_image(render)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb, mode="RGB").save(args.output)
        metadata = {
            "requested_game_loop": int(args.game_loop),
            "actual_game_loop": actual,
            "actual_seconds": actual / processor.GAME_LOOPS_PER_SECOND,
            "rgb_shape": list(rgb.shape),
            "player": int(args.player),
            "map_title": info.map_name,
            "label": "Re-rendered game view",
            "note": "Rendered from the original replay and engine; not the model-input RGB.",
        }
        args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(metadata), flush=True)
        return 0
    finally:
        engine.close()


if __name__ == "__main__":
    sys.exit(main())
