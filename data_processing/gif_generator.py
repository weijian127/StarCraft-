"""Export a GIF and MP4: uv run python -m data_processing.gif_generator.

MP4 conversion uses ffmpeg from PATH.
"""

import colorsys
import argparse
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from config import PROCESSED_DIR

FPS = 20
FRAME_STEP = 10  # Render every tenth recorded frame. Use 1 to include every frame.
OUTPUT_PATH = PROCESSED_DIR / "game.gif"
EXPORT_MP4 = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export processed minimaps as GIF and MP4.")
    parser.add_argument(
        "--pov", choices=("single", "both", "fogless"), help="Omit for older exports"
    )
    parser.add_argument("--player", type=int, default=1, help="Focus player ID in a POV export")
    args = parser.parse_args()

    if FPS <= 0:
        raise ValueError("FPS must be positive")
    if FRAME_STEP < 1:
        raise ValueError("FRAME_STEP must be at least 1")
    # GIF delays use 10 ms units. Extremely high FPS must not produce a zero delay.
    frame_duration_ms = max(10, round(100 / FPS) * 10)

    game_paths = sorted(PROCESSED_DIR.glob("*/SpatialFeatures.npz"))
    output_path = OUTPUT_PATH
    if args.pov is not None:
        game_paths = sorted(
            (PROCESSED_DIR / args.pov).glob(f"*_player_{args.player}/SpatialFeatures.npz")
        )
    if not game_paths:
        parser.error("No matching output found. Extract this POV mode first.")
    game_path = game_paths[0]
    if args.pov is not None:
        output_path = game_path.parent / "game.gif"
    with np.load(game_path, allow_pickle=False) as data:
        visibility = data["visibility_map"]
        ownership = data["player_relative"]
        unit_types = data["unit_type"]
    with np.load(game_path.with_name("TimeInfo.npz"), allow_pickle=False) as timing:
        timestamps = timing["faster_seconds"]

    visibility_colors = np.array(
        [(0, 0, 0), (100, 100, 100), (255, 255, 255), (180, 0, 255)], dtype=np.uint8
    )
    ownership_colors = np.array(
        [(0, 0, 0), (40, 120, 255), (0, 200, 0), (150, 150, 150), (255, 60, 60)],
        dtype=np.uint8,
    )

    # Build one color lookup for the whole game, so colors stay consistent over time.
    unit_ids = np.unique(unit_types)
    unit_colors = np.zeros((len(unit_ids), 3), dtype=np.uint8)
    for index, unit_id in enumerate(unit_ids):
        if unit_id != 0:
            color = colorsys.hsv_to_rgb(index / len(unit_ids), 0.8, 1.0)
            unit_colors[index] = np.array(color) * 255

    frames = []
    frame_indices = list(range(0, len(timestamps), FRAME_STEP))
    if frame_indices[-1] != len(timestamps) - 1:
        frame_indices.append(len(timestamps) - 1)
    for index in frame_indices:
        seconds = timestamps[index]
        unit_color_indices = np.searchsorted(unit_ids, unit_types[index])
        panels = [
            visibility_colors[visibility[index]],
            ownership_colors[ownership[index]],
            unit_colors[unit_color_indices],
        ]
        pixels = np.concatenate(panels, axis=1)
        maps = Image.fromarray(pixels).resize((768, 256), Image.Resampling.NEAREST)

        frame = Image.new("RGB", (768, 280), "black")
        frame.paste(maps, (0, 24))
        draw = ImageDraw.Draw(frame)
        for panel_index, title in enumerate(["Visibility", "Ownership", "Unit types"]):
            draw.text((panel_index * 256 + 8, 5), title, fill="white")
        game_time = f"{int(seconds) // 60}:{int(seconds) % 60:02d}"
        draw.text((710, 5), game_time, fill="white")
        frames.append(frame)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        disposal=2,
    )
    print(f"Saved {len(frames)} frames to {output_path}")

    if EXPORT_MP4:
        mp4_path = output_path.with_suffix(".mp4")
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(output_path),
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(mp4_path),
            ],
            check=True,
        )
        print(f"Saved MP4 to {mp4_path}")
