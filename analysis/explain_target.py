"""Compute the target class for a custom Grad-CAM cutoff.

The existing explanations explain class 1 when the model's probability of the
positive class is at least 0.5 and class 0 otherwise.  A custom browsing time has
no saved timeline probability, so this adapter runs one forward pass with the
job's own checkpoint, POV, frame count and seed to select the same policy target
before the Grad-CAM subprocess runs.  It writes metadata only; no explanation is
created here.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analysis.gradcam_context import sample_indices
from data_processing.dataset import SPATIAL_FEATURES
from training.train import choose_device, load_model


def _load_times(sample_dir):
    with np.load(Path(sample_dir) / "TimeInfo.npz", allow_pickle=False) as archive:
        seconds = archive["faster_seconds"].astype(float)
    if len(seconds) < 2 or not np.isfinite(seconds).all() or not np.all(np.diff(seconds) > 0):
        raise ValueError("Saved observation timestamps are invalid")
    return seconds


def probability_at(sample_dir, checkpoint, cutoff, frames, seed, device):
    """Return the model's class probabilities and the policy target."""
    sample_dir = Path(sample_dir)
    seconds = _load_times(sample_dir)
    frames = int(frames)
    if not 1 <= frames < len(seconds):
        raise ValueError("Not enough saved observations for the requested frame count")
    if not seconds[frames - 1] <= float(cutoff) <= seconds[-2]:
        raise ValueError(
            "Cutoff must contain enough sampled history and precede the terminal observation")
    count = int(np.searchsorted(seconds[:-1], float(cutoff), side="right"))
    indices = sample_indices(count, frames, int(seed))
    with np.load(sample_dir / "SpatialFeatures.npz", allow_pickle=False) as archive:
        inputs = {
            name: torch.as_tensor(archive[name][indices].astype(np.int64),
                                  device=device).unsqueeze(0)
            for name in SPATIAL_FEATURES
        }
    model = load_model(checkpoint, choose_device(device)).requires_grad_(False)
    torch.set_num_threads(4)
    with torch.no_grad():
        probabilities = model(inputs).softmax(-1)[0].detach().cpu().numpy()
    if probabilities.shape != (2,) or not np.isfinite(probabilities).all():
        raise ValueError("The checkpoint did not return two finite class probabilities")
    probability = float(probabilities[1])
    return {
        "cutoff_seconds": float(cutoff),
        "frames": frames,
        "seed": int(seed),
        "probability": probability,
        "class_probabilities": [float(value) for value in probabilities],
        "target_class": 1 if probability >= 0.5 else 0,
        "label_definition": model.label_definition,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cpu", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cutoff", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = probability_at(args.sample, args.checkpoint, args.cutoff,
                              args.frames, args.seed, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
