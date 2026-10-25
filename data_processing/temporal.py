"""Elapsed-time examples containing only observations available by the cutoff."""

from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset, default_collate

from data_processing.dataset import SC2ReplayDataset


def prefix_indices(seconds, cutoff, *, frames=50, history_seconds=0, random=False):
    """Select a chronological clip ending at the latest observation <= cutoff.

    Cutoffs are absolute elapsed seconds, never a fraction of the final duration.
    Zero history_seconds uses the entire observed prefix. Evaluation uses evenly
    spaced indices; training randomly samples earlier frames plus the last one.
    """
    if not np.isfinite(cutoff) or cutoff < 0 or frames < 1 or history_seconds < 0:
        raise ValueError("Invalid cutoff, frame count, or history length")
    if not np.isfinite(history_seconds):
        raise ValueError("history_seconds must be finite")
    end = int(np.searchsorted(seconds, cutoff, side="right"))
    start = int(np.searchsorted(seconds, max(0, cutoff - history_seconds), side="left")) if history_seconds else 0
    count = end - start
    if count < frames:
        raise ValueError(f"Only {count} observations available at {cutoff:g}s; need {frames}")
    if frames == 1:
        return np.array([end - 1], dtype=np.int64)
    if random:
        weights = torch.log1p(torch.as_tensor(seconds[start:end - 1], dtype=torch.float64) + 1)
        chosen = torch.multinomial(weights, frames - 1, replacement=False).numpy() + start
        return np.sort(np.append(chosen, end - 1))
    return np.linspace(start, end - 1, frames, dtype=np.int64)


class PrefixReplayDataset(Dataset):
    """Expand a grouped match split into fixed elapsed-time prediction examples.

    Terminal observations are excluded. Each example preserves its frame indices,
    elapsed times, cutoff and match/POV identity for explanations and evaluation.
    Caches at most cache_samples decoded replays per worker (0 disables caching).
    """

    collate_fn = staticmethod(default_collate)

    def __init__(self, root_dir, *, pov="both", split="train", frames=50,
                 min_seconds=60, step_seconds=60, max_seconds=1200,
                 history_seconds=0, seed=42, test_fraction=0.2,
                 stratify=True, random_sampling=None, cache_samples=2):
        if frames < 1 or min_seconds < 0 or step_seconds <= 0 or max_seconds < min_seconds:
            raise ValueError("Invalid frame count or elapsed-time grid")
        if not all(np.isfinite(v) for v in (min_seconds, step_seconds, max_seconds, history_seconds)):
            raise ValueError("Time settings must be finite")
        if history_seconds < 0 or cache_samples < 0:
            raise ValueError("History length and cache size must be nonnegative")
        self.base = SC2ReplayDataset(root_dir, pov=pov, split=split, max_timesteps=frames,
                                    start_prob=0, length_prob=1, seed=seed,
                                    test_fraction=test_fraction, stratify=stratify)
        self.frames = frames
        self.history_seconds = history_seconds
        self.random_sampling = split == "train" if random_sampling is None else random_sampling
        self.cache_samples = cache_samples
        self._cache = OrderedDict()
        self.timings = []
        self.examples = []
        grid = np.arange(min_seconds, max_seconds + step_seconds * 1e-9, step_seconds)
        for index, sample in enumerate(self.base.samples):
            timing_path = (self.base.root_dir / sample["timing_path"]).resolve()
            if not timing_path.is_relative_to(self.base.root_dir):
                raise ValueError("Timing path must stay inside the export directory")
            with np.load(timing_path, allow_pickle=False) as timing:
                seconds = timing["faster_seconds"].astype(np.float64)
            if (seconds.shape != (sample["frames"],) or not np.isfinite(seconds).all()
                    or seconds[0] < 0 or not np.all(np.diff(seconds) > 0)):
                raise ValueError(f"Invalid timestamps: {sample['sample_id']}")
            self.timings.append(seconds)
            for cutoff in grid:
                if cutoff >= seconds[-1]:
                    continue  # Do not train on the observation that announces the result.
                try:
                    prefix_indices(seconds, cutoff, frames=frames, history_seconds=history_seconds)
                except ValueError:
                    continue
                self.examples.append((index, float(cutoff)))
        if not self.examples:
            raise ValueError("No eligible prefixes; adjust elapsed-time grid/frame count")
        active = sorted({index for index, _ in self.examples})
        self.samples = [self.base.samples[index] for index in active]
        self.replay_ids = frozenset(sample["replay_id"] for sample in self.samples)

    def __len__(self):
        return len(self.examples)

    def arrays(self, base_index):
        if base_index in self._cache:
            self._cache.move_to_end(base_index)
            return self._cache[base_index]
        features, _ = self.base[base_index]
        if self.cache_samples:
            self._cache[base_index] = features
            while len(self._cache) > self.cache_samples:
                self._cache.popitem(last=False)
        return features

    def __getitem__(self, index):
        base_index, cutoff = self.examples[index]
        metadata = self.base.samples[base_index]
        seconds = self.timings[base_index]
        indices = prefix_indices(seconds, cutoff, frames=self.frames,
                                 history_seconds=self.history_seconds, random=self.random_sampling)
        features = self.arrays(base_index)
        return {
            "inputs": {name: torch.from_numpy(layer[indices].astype(np.int64)) for name, layer in features.items()},
            "targets": torch.tensor(metadata["label"], dtype=torch.long),
            "frame_indices": torch.from_numpy(indices.copy()),
            "frame_seconds": torch.from_numpy(seconds[indices].copy()),
            "cutoff_seconds": cutoff, "replay_id": metadata["replay_id"],
            "sample_id": metadata["sample_id"], "focus_player_id": metadata["focus_player_id"],
        }
