import argparse

import matplotlib.pyplot as plt
import numpy as np

from config import PROCESSED_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View processed minimap layers.")
    parser.add_argument(
        "--pov", choices=("single", "both", "fogless"), help="Omit for older exports"
    )
    parser.add_argument("--player", type=int, default=1, help="Focus player ID in a POV export")
    parser.add_argument("--frame", type=int, default=1510)
    args = parser.parse_args()

    game_paths = sorted(PROCESSED_DIR.glob("*/SpatialFeatures.npz"))
    if args.pov is not None:
        game_paths = sorted(
            (PROCESSED_DIR / args.pov).glob(f"*_player_{args.player}/SpatialFeatures.npz")
        )
    if not game_paths:
        parser.error("No matching output found. Extract this POV mode first.")
    game_path = game_paths[0]
    data = np.load(game_path, allow_pickle=False)

    frame_index = args.frame
    frame_count = len(data["visibility_map"])
    if not 0 <= frame_index < frame_count:
        parser.error(f"--frame must be between 0 and {frame_count - 1}")
    figure, axes = plt.subplots(1, 3, figsize=(14, 5))

    for axis, name in zip(axes, data.files):
        frame = data[name][frame_index]

        # Give each category a distinct position on the color scale.
        categories, color_indices = np.unique(frame, return_inverse=True)
        image = axis.imshow(
            color_indices.reshape(frame.shape),
            cmap="turbo",
            interpolation="nearest",
        )

        axis.set_title(name)
        colorbar = figure.colorbar(image, ax=axis, ticks=range(len(categories)))
        colorbar.ax.set_yticklabels(categories)
        colorbar.ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.show()
    data.close()
