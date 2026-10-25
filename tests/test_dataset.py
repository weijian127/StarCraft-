import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from data_processing.dataset import SC2ReplayDataset, SPATIAL_FEATURES, replay_collate_fn


def make_export(root, *, pov="both", lengths=(80, 90, 100, 110, 120, 130)):
    """Frame-coded recordings: channel offsets expose alignment errors."""
    output = root / pov
    output.mkdir()
    samples = []
    for match, frames in enumerate(lengths):
        for player in ((1, 2) if pov == "both" else (1,)):
            sample_id = f"match_{match}_{pov}_player_{player}"
            folder = output / sample_id
            folder.mkdir()
            timeline = np.broadcast_to(np.arange(frames)[:, None, None], (frames, 2, 2))
            np.savez_compressed(
                folder / "SpatialFeatures.npz",
                visibility_map=timeline.astype(np.uint8),
                player_relative=(timeline + 10).astype(np.uint8),
                unit_type=(timeline + 5000).astype(np.uint16),
            )
            samples.append({
                "replay_id": f"match_{match}", "sample_id": sample_id,
                "focus_player_id": player, "pov_mode": pov,
                "label": match % 2, "frames": frames,
                "source_path": f"{sample_id}/SpatialFeatures.npz",
            })
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 2, "samples": samples}))
    return manifest


@pytest.mark.parametrize("pov", ["single", "both", "fogless"])
def test_manifest_labels_and_selected_features(tmp_path, pov):
    make_export(tmp_path, pov=pov)
    dataset = SC2ReplayDataset(tmp_path, pov=pov, include=("unit_type", "visibility_map"))
    features, label = dataset[0]
    assert tuple(features) == ("unit_type", "visibility_map")
    assert features["unit_type"].dtype == np.uint16
    assert label == 0
    if pov == "both":
        assert dataset[1][1] == label  # Changing the observed player cannot flip the target.
    assert len(dataset) == (12 if pov == "both" else 6)


def test_split_keeps_matches_together_and_is_order_independent(tmp_path):
    manifest = make_export(tmp_path)
    train = SC2ReplayDataset(tmp_path, split="train", seed=7)
    test = SC2ReplayDataset(tmp_path, split="test", seed=7)
    assert train.replay_ids.isdisjoint(test.replay_ids)
    assert train.replay_ids | test.replay_ids == {f"match_{i}" for i in range(6)}
    assert len(train) + len(test) == 12
    assert all(sum(s["replay_id"] == rid for s in train.samples) == 2 for rid in train.replay_ids)
    data = json.loads(manifest.read_text())
    data["samples"].reverse()
    manifest.write_text(json.dumps(data))
    assert SC2ReplayDataset(tmp_path, split="test", seed=7).replay_ids == test.replay_ids


def test_exact_fit_and_filtering_does_not_change_split(tmp_path):
    make_export(tmp_path, lengths=(49, 50, 80, 100, 120, 140))
    dataset = SC2ReplayDataset(tmp_path, start_prob=0, length_prob=1)
    assert dataset.skipped_samples == 2
    assert "match_1" in dataset.replay_ids
    batch = dataset.collate_fn([dataset[0], dataset[1]])
    assert batch["inputs"]["visibility_map"].shape == (2, 50, 2, 2)
    for split in ("train", "test"):
        full = SC2ReplayDataset(tmp_path, split=split, max_timesteps=1, start_prob=0, length_prob=1)
        shorter = SC2ReplayDataset(tmp_path, split=split, start_prob=0, length_prob=0.75)
        assert shorter.replay_ids <= full.replay_ids


@pytest.mark.parametrize("weighting", ["log", "exp", "uniform"])
def test_sampling_is_aligned_sorted_reproducible_and_within_window(tmp_path, weighting):
    make_export(tmp_path)
    dataset = SC2ReplayDataset(tmp_path, max_timesteps=20, weighting=weighting, start_prob=0.25, length_prob=0.75)
    torch.manual_seed(123)
    batch = dataset.collate_fn([dataset[0], dataset[2]])
    frames = batch["inputs"]["visibility_map"]
    assert frames.shape == (2, 20, 2, 2)
    for row, length in enumerate((80, 90)):
        indices = frames[row, :, 0, 0]
        assert torch.all(indices[1:] > indices[:-1])
        assert int(indices.min()) >= int(length * 0.25)
        assert int(indices.max()) < int(length * 0.75)
    assert torch.equal(batch["inputs"]["player_relative"], frames + 10)
    assert torch.equal(batch["inputs"]["unit_type"], frames + 5000)
    assert all(t.dtype == torch.int64 for t in batch["inputs"].values())
    assert batch["targets"].dtype == torch.int64
    assert batch["targets"].tolist() == [0, 1]
    torch.manual_seed(123)
    repeated = dataset.collate_fn([dataset[0], dataset[2]])
    assert torch.equal(frames, repeated["inputs"]["visibility_map"])


def test_exp_sampling_stays_finite_for_long_replays():
    features = {"unit_type": np.arange(10000, dtype=np.uint16).reshape(-1, 1, 1)}
    batch = replay_collate_fn([(features, 1)], weighting="exp", start_prob=0, length_prob=1)
    indices = batch["inputs"]["unit_type"].flatten()
    assert len(indices.unique()) == 50
    assert torch.all(indices[1:] > indices[:-1])
    assert indices[-1] <= 9999


def test_one_frame_keeps_time_dimension_and_uint16_ids():
    features = {"unit_type": np.full((1, 2, 2), 60000, dtype=np.uint16)}
    batch = replay_collate_fn([(features, 1)], max_timesteps=1, start_prob=0, length_prob=1)
    assert batch["inputs"]["unit_type"].shape == (1, 1, 2, 2)
    assert torch.all(batch["inputs"]["unit_type"] == 60000)


def test_short_sample_does_not_silently_resize_batch():
    short = {"unit_type": np.zeros((49, 2, 2), dtype=np.uint16)}
    with pytest.raises(ValueError, match="49 frames; need 50"):
        replay_collate_fn([(short, 0)], start_prob=0, length_prob=1)


@pytest.mark.parametrize("options", [
    {"max_timesteps": 0}, {"weighting": "invalid"},
    {"start_prob": 0.8, "length_prob": 0.2}, {"length_prob": 1.1},
    {"split": "invalid"}, {"test_fraction": 0},
    {"include": ()}, {"include": ("unit_type", "unit_type")},
])
def test_invalid_configuration_is_rejected(tmp_path, options):
    with pytest.raises(ValueError):
        SC2ReplayDataset(tmp_path, **options)


@pytest.mark.parametrize("field,value,message", [
    ("label", 1, "Conflicting labels"),
    ("sample_id", "match_0_both_player_2", "Duplicate sample_id"),
    ("pov_mode", "fogless", "wrong POV"),
])
def test_invalid_manifest_is_rejected(tmp_path, field, value, message):
    manifest = make_export(tmp_path)
    data = json.loads(manifest.read_text())
    data["samples"][0][field] = value
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        SC2ReplayDataset(tmp_path)


@pytest.mark.parametrize("field,value,message", [
    ("frames", 81, "frame count"),
    ("source_path", "../outside.npz", "inside"),
])
def test_invalid_sample_is_reported(tmp_path, field, value, message):
    manifest = make_export(tmp_path)
    data = json.loads(manifest.read_text())
    data["samples"][0][field] = value
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        SC2ReplayDataset(tmp_path)[0]


def test_spawn_worker_loading_is_reproducible(tmp_path):
    make_export(tmp_path)
    dataset = SC2ReplayDataset(tmp_path)

    def read_batches():
        loader = DataLoader(
            dataset, batch_size=2, collate_fn=dataset.collate_fn,
            num_workers=1, multiprocessing_context="spawn", shuffle=True,
            generator=torch.Generator().manual_seed(81),
        )
        return list(loader)

    first, second = read_batches(), read_batches()
    assert len(first) == len(second) == 6
    for left, right in zip(first, second):
        assert torch.equal(left["targets"], right["targets"])
        for name in SPATIAL_FEATURES:
            assert torch.equal(left["inputs"][name], right["inputs"][name])


def test_batch_feeds_embeddings_and_classification_loss(tmp_path):
    make_export(tmp_path)
    dataset = SC2ReplayDataset(tmp_path)
    batch = dataset.collate_fn([dataset[0], dataset[2]])
    embeddings = torch.nn.Embedding(65536, 2)
    classifier = torch.nn.Linear(2, 2)
    encoded = embeddings(batch["inputs"]["unit_type"]).mean(dim=(1, 2, 3))
    loss = torch.nn.functional.cross_entropy(classifier(encoded), batch["targets"])
    loss.backward()
    assert torch.isfinite(loss)
    assert embeddings.weight.grad is not None
