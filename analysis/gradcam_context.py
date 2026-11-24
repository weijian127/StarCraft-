"""Metadata and minimap reference images for Grad-CAM explanations.

Everything here uses saved observations, the original deterministic sampler and
the original frame/seed/cutoff settings. No model is loaded and no timeline is
recomputed.
"""

import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data_processing.dataset import replay_collate_fn

OWNERSHIP_COLORS = (
    (25, 32, 40),      # 0 background
    (40, 120, 255),    # 1 self
    (0, 200, 0),       # 2 ally
    (150, 150, 150),   # 3 neutral
    (255, 60, 60),     # 4 enemy
)
VISIBILITY_COLORS = (
    (0, 0, 0),         # 0 hidden
    (63, 63, 63),      # 1 seen
    (200, 200, 200),   # 2 visible
    (255, 0, 255),     # 3 reserved / unknown
)
CONTEXT_LEGEND = {
    "ownership": {"0": "background", "1": "self", "2": "ally", "3": "neutral", "4": "enemy"},
    "ownership_colors": {str(index): "#%02x%02x%02x" % color
                         for index, color in enumerate(OWNERSHIP_COLORS)},
    "visibility": {"0": "hidden", "1": "seen", "2": "visible", "3": "reserved"},
    "visibility_colors": {str(index): "#%02x%02x%02x" % color
                          for index, color in enumerate(VISIBILITY_COLORS)},
    "note": "Model-input minimap visualization; not a game screenshot. "
            "Ownership colors follow the team OWNERSHIP palette.",
}


def sample_indices(count, frames, seed):
    """The same deterministic sampler used by viz.common.sample_indices."""
    if not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("count must be a positive integer")
    if not isinstance(frames, (int, np.integer)) or frames < 1:
        raise ValueError("frames must be a positive integer")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        batch = replay_collate_fn(
            [({"index": np.arange(count)[:, None, None]}, 0)],
            max_timesteps=int(frames), start_prob=0, length_prob=1,
        )
    return batch["inputs"]["index"][0, :, 0, 0].numpy().astype(np.int64)


def _load_times(sample_dir):
    sample_dir = Path(sample_dir)
    with np.load(sample_dir / "TimeInfo.npz", allow_pickle=False) as archive:
        seconds = archive["faster_seconds"].astype(float)
        loops = archive["game_loops"].astype(np.int64)
    if not (len(seconds) == len(loops) and len(seconds) >= 2
            and np.isfinite(seconds).all() and np.all(np.diff(seconds) > 0)
            and np.all(np.diff(loops) > 0)):
        raise ValueError("Saved observation timestamps are invalid")
    return seconds, loops


def sampling_metadata(sample_dir, cutoff, frames, seed, source):
    """Reconstruct the sampler result and record its provenance."""
    seconds, loops = _load_times(sample_dir)
    cutoff = float(cutoff)
    count = int(np.searchsorted(seconds[:-1], cutoff, side="right"))
    frames = int(frames)
    if count < frames:
        raise ValueError(f"Only {count} observations available at {cutoff:g}s; need {frames}")
    indices = sample_indices(count, frames, seed)
    if indices.ndim != 1 or len(indices) != frames or indices.min() < 0 or indices.max() >= count:
        raise ValueError("Reconstructed sample indices are invalid")
    selected_seconds = seconds[indices]
    selected_loops = loops[indices]
    last = int(indices[-1])
    return {
        "source": source,
        "method": "team viz.common.sample_indices (replay_collate_fn, "
                  "start_prob=0, length_prob=1)",
        "frames": frames,
        "seed": int(seed),
        "count": count,
        "frame_indices": [int(value) for value in indices],
        "frame_seconds": [float(value) for value in selected_seconds],
        "game_loops": [int(value) for value in selected_loops],
        "min_seconds": float(selected_seconds.min()),
        "max_seconds": float(selected_seconds.max()),
        "min_game_loop": int(selected_loops.min()),
        "max_game_loop": int(selected_loops.max()),
        "last_frame_index": last,
        "last_frame_seconds": float(seconds[last]),
        "last_game_loop": int(loops[last]),
    }


def render_minimap_frame(player_relative, visibility, output_path):
    """Render one saved ownership/visibility observation with the shared palette."""
    player_relative = np.asarray(player_relative)
    visibility = np.asarray(visibility)
    if player_relative.shape != (128, 128) or visibility.shape != (128, 128):
        raise ValueError("Expected 128x128 minimap layers")
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    visibility = np.clip(visibility, 0, len(VISIBILITY_COLORS) - 1).astype(np.int64)
    for index, color in enumerate(VISIBILITY_COLORS):
        image[visibility == index] = color
    player_relative = np.clip(player_relative, 0, len(OWNERSHIP_COLORS) - 1).astype(np.int64)
    for index, color in enumerate(OWNERSHIP_COLORS[1:], start=1):
        image[player_relative == index] = color
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").resize((512, 512), Image.Resampling.NEAREST).save(output_path)


def render_minimap_context(sample_dir, frame_index, output_path):
    """Render ownership/visibility at one saved observation to a square PNG."""
    sample_dir = Path(sample_dir)
    with np.load(sample_dir / "SpatialFeatures.npz", allow_pickle=False) as archive:
        player_relative = archive["player_relative"][int(frame_index)]
        visibility = archive["visibility_map"][int(frame_index)]
    render_minimap_frame(player_relative, visibility, output_path)


def render_minimap_frames(sample_dir, frame_indices, output_paths):
    """Render several saved observations while decompressing the NPZ only once.

    Dragging needs a minimap for every saved cutoff, and those frames repeat. The
    caller passes each unique frame once; this keeps pointer movement from
    decompressing the whole feature archive per cutoff.
    """
    sample_dir = Path(sample_dir)
    frame_indices = [int(value) for value in frame_indices]
    output_paths = [Path(value) for value in output_paths]
    if not frame_indices or len(frame_indices) != len(output_paths):
        raise ValueError("frame_indices and output_paths must be non-empty and aligned")
    with np.load(sample_dir / "SpatialFeatures.npz", allow_pickle=False) as archive:
        if "player_relative" not in archive.files or "visibility_map" not in archive.files:
            raise ValueError("SpatialFeatures.npz lacks the minimap layers")
        player_relative = archive["player_relative"]
        visibility = archive["visibility_map"]
        if len(player_relative) != len(visibility):
            raise ValueError("Minimap layers disagree on frame count")
        for frame, output in zip(frame_indices, output_paths):
            if frame < 0 or frame >= len(visibility):
                raise ValueError(f"frame index {frame} is outside the saved recording")
            render_minimap_frame(player_relative[frame], visibility[frame], output)



def observation_times(sample_dir):
    """Every saved TimeInfo timestamp for one POV (seconds and game loops)."""
    return _load_times(sample_dir)


def load_minimap_codes(sample_dir):
    """One uint8 code per minimap cell for all saved observations.

    code = visibility * 5 + ownership. Storing the two saved layers as one code
    array keeps the in-memory browse cache bounded (one byte per cell) while
    still rendering exactly the saved observation; no unit positions are
    interpolated and no frames are synthesised.
    """
    sample_dir = Path(sample_dir)
    with np.load(sample_dir / "SpatialFeatures.npz", allow_pickle=False) as archive:
        if "player_relative" not in archive.files or "visibility_map" not in archive.files:
            raise ValueError("SpatialFeatures.npz lacks the minimap layers")
        player_relative = archive["player_relative"]
        visibility = archive["visibility_map"]
        if len(player_relative) != len(visibility):
            raise ValueError("Minimap layers disagree on frame count")
        codes = (np.clip(visibility, 0, len(VISIBILITY_COLORS) - 1).astype(np.uint8) *
                 len(OWNERSHIP_COLORS) +
                 np.clip(player_relative, 0, len(OWNERSHIP_COLORS) - 1).astype(np.uint8))
    return codes


def _minimap_lut():
    """Ownership over visibility, matching render_minimap_frame()."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for visibility, color in enumerate(VISIBILITY_COLORS):
        lut[visibility * len(OWNERSHIP_COLORS):(visibility + 1) * len(OWNERSHIP_COLORS)] = color
    for ownership, color in enumerate(OWNERSHIP_COLORS[1:], start=1):
        for visibility in range(len(VISIBILITY_COLORS)):
            lut[visibility * len(OWNERSHIP_COLORS) + ownership] = color
    return lut


MINIMAP_LUT = _minimap_lut()


def minimap_frame_png(codes, frame_index, size=512):
    """PNG bytes for exactly one saved observation, using the shared palette."""
    codes = np.asarray(codes)
    if codes.ndim != 3 or codes.shape[1:] != (128, 128):
        raise ValueError("Expected saved 128x128 minimap code frames")
    frame = int(frame_index)
    if frame < 0 or frame >= len(codes):
        raise ValueError(f"frame index {frame} is outside the saved recording")
    image = MINIMAP_LUT[np.clip(codes[frame], 0, 255)]
    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").resize(
        (int(size), int(size)), Image.Resampling.NEAREST).save(
            buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def attribution_summary(data_npz):
    """Describe the raw heatmap peak without inventing a temporal frame mapping."""
    with np.load(Path(data_npz), allow_pickle=False) as archive:
        if "raw_heatmap" not in archive.files:
            raise ValueError("data.npz has no raw_heatmap")
        raw = np.asarray(archive["raw_heatmap"])
    if raw.ndim == 4:
        if raw.shape[0] != 1:
            raise ValueError("Expected one static Grad-CAM sample")
        raw = raw[0]
    if raw.ndim != 3:
        raise ValueError(f"Unexpected raw_heatmap shape {raw.shape}")
    maximum = float(raw.max())
    available = maximum > 0
    cells = []
    if available:
        for bin_index, row, col in np.argwhere(
                np.isclose(raw, maximum, rtol=0.0, atol=1e-12)):
            cells.append([int(bin_index), int(row), int(col)])
    return {
        "available": available,
        "max_value": maximum,
        "max_normalized": 1.0 if available else 0.0,
        "normalization": "per-clip divide by global maximum in viz.gradcam.snapshot",
        "raw_axis_order": "[batch, temporal_bin, spatial_row, spatial_col]",
        "display_orientation": "origin upper; row 0 top, column 0 left; team imshow order",
        "grid_shape": [int(raw.shape[1]), int(raw.shape[2])],
        "temporal_bins": int(raw.shape[0]),
        "bin_index": cells[0][0] if cells else None,
        "row": cells[0][1] if cells else None,
        "col": cells[0][2] if cells else None,
        "cells": cells,
        "tied": len(cells) > 1,
        "note": "Highest Grad-CAM region is a coarse model-attribution reference, "
                "not a decisive position or action.",
    }


def enrich_explanation(sample_dir, review_dir, cutoff, frames, seed, data_npz,
                       prefix, source, image_suffix=""):
    """Build sampling, context minimap and attribution fields for one explanation."""
    review_dir = Path(review_dir)
    sampling = sampling_metadata(sample_dir, cutoff, frames, seed, source)
    suffix = str(image_suffix or "")
    name = "minimap_%s_%ss%s.png" % (prefix, int(round(float(cutoff))), suffix)
    context_path = review_dir / name
    render_minimap_context(sample_dir, sampling["last_frame_index"], context_path)
    summary = attribution_summary(data_npz)
    return {
        "sampling": sampling,
        "context_image": name,
        "context_frame": {
            "frame_index": sampling["last_frame_index"],
            "frame_seconds": sampling["last_frame_seconds"],
            "game_loop": sampling["last_game_loop"],
            "source": source,
        },
        "context_legend": CONTEXT_LEGEND,
        "attribution": summary,
    }
