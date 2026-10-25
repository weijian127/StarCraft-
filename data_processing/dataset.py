"""PyTorch replay loading and temporal batching for schema-v2 exports.

Adapted from Insung Baek's StarCraft2/utils_sc/data.py. Retains the reference
feature dictionary and inputs/targets batch interface, using our manifest for
labels, lengths, and match-level splits instead of train/test directories.
"""

import argparse
import json
from collections import Counter
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import PROCESSED_DIR, resolve_path

SPATIAL_FEATURES = ("visibility_map", "player_relative", "unit_type")
POV_MODES = ("single", "both", "fogless")


def _validate_sampling(max_timesteps, weighting, start_prob, length_prob):
    if not isinstance(max_timesteps, int) or max_timesteps < 1:
        raise ValueError("max_timesteps must be a positive integer")
    if weighting not in ("log", "exp", "uniform"):
        raise ValueError("weighting must be log, exp, or uniform")
    if not 0 <= start_prob < length_prob <= 1:
        raise ValueError("Require 0 <= start_prob < length_prob <= 1")


def _sampling_window(frames, start_prob, length_prob):
    return int(frames * start_prob), int(frames * length_prob)


def _test_match_ids(match_labels, test_fraction, seed, stratify):
    """Assign whole matches; optionally retain both outcomes in both splits."""
    match_ids = sorted(match_labels)
    if len(match_ids) < 2:
        raise ValueError("At least two matches are needed for a train/test split")
    count = max(1, min(len(match_ids) - 1, round(len(match_ids) * test_fraction)))
    rng = np.random.default_rng(seed)
    if not stratify:
        return set(rng.permutation(match_ids)[:count])
    groups = [[rid for rid in match_ids if match_labels[rid] == label] for label in (0, 1)]
    if min(map(len, groups)) < 2 or count < 2 or len(match_ids) - count < 2:
        raise ValueError("Stratification needs two matches per outcome and two per split; "
                         "adjust test_fraction or disable stratify")
    low = max(1, count - (len(groups[1]) - 1))
    high = min(len(groups[0]) - 1, count - 1)
    count_zero = min(high, max(low, round(count * len(groups[0]) / len(match_ids))))
    return set(rng.permutation(groups[0])[:count_zero]) | set(
        rng.permutation(groups[1])[:count - count_zero]
    )


class SC2ReplayDataset(Dataset):
    """Load selected spatial layers and a Terran-win label for each POV.

    Args:
        root_dir: Parent of the single/both/fogless export directories.
        pov: Export mode; defaults to the checked-in both-player dataset.
        split: all, train, or test. Splits group both POVs by replay_id.
        include: Ordered feature names; this order is preserved in each batch.
        max_timesteps: Number of frames per sample after collation.
        weighting: log (reference default), exp, or uniform temporal sampling.
        start_prob: Beginning of the sampling window as a fraction of the replay.
        length_prob: Exclusive end of the window, not its length.
        test_fraction: Fraction of matches reserved for testing, rounded to the
            nearest integer, with at least one match in each split.
        seed: Seed for the match split. Use torch.manual_seed for frame sampling.
        stratify: Keep both outcomes in each split; requires enough matches.

    Splits are assigned before short-sample filtering, so changing the temporal
    window cannot move a match between training and testing. Samples with fewer
    than max_timesteps frames in that window are excluded. Use collate_fn from
    this instance to keep filtering and batching settings consistent.
    """

    def __init__(
        self,
        root_dir=PROCESSED_DIR,
        *,
        pov="both",
        split="all",
        include=SPATIAL_FEATURES,
        max_timesteps=50,
        weighting="log",
        start_prob=0.1,
        length_prob=0.8,
        test_fraction=0.2,
        seed=42,
        stratify=False,
    ):
        _validate_sampling(max_timesteps, weighting, start_prob, length_prob)
        if pov not in POV_MODES:
            raise ValueError(f"pov must be one of {POV_MODES}")
        if split not in ("all", "train", "test"):
            raise ValueError("split must be all, train, or test")
        if not 0 < test_fraction < 1:
            raise ValueError("test_fraction must be between 0 and 1")
        self.include = tuple(include)
        if (
            not self.include
            or len(set(self.include)) != len(self.include)
            or not set(self.include).issubset(SPATIAL_FEATURES)
        ):
            raise ValueError(f"include must contain unique names from {SPATIAL_FEATURES}")

        self.root_dir = resolve_path(root_dir) / pov
        self.manifest_path = self.root_dir / "manifest.json"
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 2:
            raise ValueError("Expected a schema_version 2 manifest")
        samples = manifest["samples"]
        match_labels = {}
        sample_ids = set()
        for sample in samples:
            if sample["sample_id"] in sample_ids:
                raise ValueError(f"Duplicate sample_id: {sample['sample_id']}")
            sample_ids.add(sample["sample_id"])
            if sample["pov_mode"] != pov:
                raise ValueError(f"Sample {sample['sample_id']} has the wrong POV mode")
            if sample["label"] not in (0, 1):
                raise ValueError("Labels must be 0 (Protoss wins) or 1 (Terran wins)")
            if not isinstance(sample["frames"], int) or sample["frames"] < 1:
                raise ValueError("Sample frame counts must be positive integers")
            previous = match_labels.setdefault(sample["replay_id"], sample["label"])
            if previous != sample["label"]:
                raise ValueError(f"Conflicting labels for match {sample['replay_id']}")

        if split != "all":
            test_ids = _test_match_ids(match_labels, test_fraction, seed, stratify)
            samples = [
                sample for sample in samples
                if (sample["replay_id"] in test_ids) == (split == "test")
            ]

        self.samples = []
        self.skipped_samples = 0
        for sample in samples:
            start, end = _sampling_window(sample["frames"], start_prob, length_prob)
            if end - start < max_timesteps:
                self.skipped_samples += 1
                continue
            self.samples.append(sample)
        if not self.samples:
            raise ValueError("No samples have enough frames in the requested split/window")

        self.replay_ids = frozenset(sample["replay_id"] for sample in self.samples)
        self.counts = Counter(sample["label"] for sample in self.samples)
        self.collate_fn = partial(
            replay_collate_fn,
            max_timesteps=max_timesteps,
            weighting=weighting,
            start_prob=start_prob,
            length_prob=length_prob,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        source_path = (self.root_dir / sample["source_path"]).resolve()
        if not source_path.is_relative_to(self.root_dir):
            raise ValueError("Feature path must stay inside the POV export directory")
        with np.load(source_path, allow_pickle=False) as archive:
            features = {name: archive[name] for name in self.include}
        shape = next(iter(features.values())).shape
        for name, layer in features.items():
            if layer.ndim != 3 or layer.shape != shape or len(layer) != sample["frames"]:
                raise ValueError(f"Invalid shape/frame count for {sample['sample_id']}/{name}")
            if not np.issubdtype(layer.dtype, np.integer):
                raise ValueError(f"Expected categorical integer data for {name}")
        return features, sample["label"]


def replay_collate_fn(
    samples, max_timesteps=50, weighting="log", start_prob=0.1, length_prob=0.8
):
    """Sample aligned frames without replacement and return inputs/targets.

    Inputs are int64 tensors [batch, time, height, width], even for one frame.
    Targets are int64 class indices. Unit IDs are preserved, not clipped to a
    PySC2-version-dependent vocabulary. All features use the same sorted frame
    indices for each sample. Randomness follows PyTorch's seed/worker seeding.
    """
    _validate_sampling(max_timesteps, weighting, start_prob, length_prob)
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    names = tuple(samples[0][0])
    if not names:
        raise ValueError("Samples must contain at least one feature")
    batched_inputs = {name: [] for name in names}
    targets = []
    for features, target in samples:
        if tuple(features) != names:
            raise ValueError("All samples must have the same ordered feature names")
        shape = features[names[0]].shape
        if len(shape) != 3 or any(layer.shape != shape for layer in features.values()):
            raise ValueError("Features must have aligned [time, height, width] shapes")
        start, end = _sampling_window(shape[0], start_prob, length_prob)
        if end - start < max_timesteps:
            raise ValueError(f"Sampling window has {end - start} frames; need {max_timesteps}")

        times = torch.arange(start, end, dtype=torch.float64)
        if weighting == "log":
            # Shift by one so frame zero has positive weight, even in exact-fit windows.
            log_weights = torch.log(torch.log1p(times + 1))
        elif weighting == "exp":
            log_weights = times
        else:
            log_weights = torch.zeros_like(times)
        # Gumbel top-k gives weighted sampling without replacement in log space.
        # Unlike exp(t)/sum(exp(t)), it remains finite for long replays.
        uniform = torch.rand(len(times), dtype=torch.float64).clamp_min(torch.finfo(torch.float64).tiny)
        scores = log_weights - torch.log(-torch.log(uniform))
        indices = (torch.topk(scores, max_timesteps).indices + start).sort().values.numpy()
        for name in names:
            selected = features[name][indices].astype(np.int64, copy=False)
            batched_inputs[name].append(torch.from_numpy(selected))
        targets.append(target)

    return {
        "inputs": {name: torch.stack(layers) for name, layers in batched_inputs.items()},
        "targets": torch.tensor(targets, dtype=torch.long),
    }


def main():
    parser = argparse.ArgumentParser(description="Load and inspect one replay batch.")
    parser.add_argument("--root", type=resolve_path, default=PROCESSED_DIR)
    parser.add_argument("--pov", choices=POV_MODES, default="both")
    parser.add_argument("--split", choices=("all", "train", "test"), default="all")
    parser.add_argument("--max-timesteps", type=int, default=50)
    parser.add_argument("--weighting", choices=("log", "exp", "uniform"), default="log")
    parser.add_argument("--start-prob", type=float, default=0.1)
    parser.add_argument("--length-prob", type=float, default=0.8)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size must be positive and num-workers must be nonnegative")
    torch.manual_seed(args.seed)
    dataset = SC2ReplayDataset(
        args.root, pov=args.pov, split=args.split, max_timesteps=args.max_timesteps,
        weighting=args.weighting, start_prob=args.start_prob, length_prob=args.length_prob,
        test_fraction=args.test_fraction, seed=args.seed,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=args.split == "train",
        collate_fn=dataset.collate_fn, num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.seed),
    )
    batch = next(iter(loader))
    print(f"{len(dataset)} samples / {len(dataset.replay_ids)} matches; skipped {dataset.skipped_samples} short samples")
    print(f"Labels (0=Protoss, 1=Terran): {dict(dataset.counts)}")
    for name, tensor in batch["inputs"].items():
        print(f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
    print(f"targets: {batch['targets'].tolist()}")


if __name__ == "__main__":
    main()
