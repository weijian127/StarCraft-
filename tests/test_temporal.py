import json

import numpy as np
import pytest
import torch
from torch import nn

from analysis.attribution import grad_cam, pivotal_intervals
from analysis.review import masking_check
from data_processing.temporal import PrefixReplayDataset, prefix_indices
from training import train_temporal


def make_timed_export(root):
    directory = root / "both"
    directory.mkdir()
    samples = []
    for match in range(10):
        for player in (1, 2):
            sample_id = f"match{match}_p{player}"
            folder = directory / sample_id
            folder.mkdir()
            frames = np.broadcast_to(np.arange(41)[:, None, None], (41, 2, 2))
            np.savez_compressed(folder / "SpatialFeatures.npz", visibility_map=frames.astype(np.uint8),
                                player_relative=(frames + 10).astype(np.uint8),
                                unit_type=(frames + 5000).astype(np.uint16))
            np.savez(folder / "TimeInfo.npz", faster_seconds=np.arange(41, dtype=float))
            samples.append(dict(replay_id=f"match{match}", sample_id=sample_id, focus_player_id=player,
                                pov_mode="both", label=match % 2, frames=41,
                                source_path=f"{sample_id}/SpatialFeatures.npz", timing_path=f"{sample_id}/TimeInfo.npz"))
    (directory / "manifest.json").write_text(json.dumps(dict(schema_version=2, samples=samples)))


def test_cutoff_does_not_depend_on_unobserved_future_or_game_length():
    seconds = np.arange(100, dtype=float)
    prefix = prefix_indices(seconds, 42.5, frames=9)
    extended = prefix_indices(np.concatenate((seconds, np.arange(100, 400))), 42.5, frames=9)
    assert np.array_equal(prefix, extended)
    assert prefix[-1] == 42 and np.all(np.diff(prefix) > 0)
    torch.manual_seed(11)
    randomized = prefix_indices(seconds, 42.5, frames=9, history_seconds=20, random=True)
    assert randomized[0] >= 23 and randomized[-1] == 42
    assert len(np.unique(randomized)) == 9
    with pytest.raises(ValueError, match="observations"):
        prefix_indices(seconds, 5, frames=9)


def test_prefix_provenance_determinism_and_match_isolation(tmp_path):
    make_timed_export(tmp_path)
    settings = dict(root_dir=tmp_path, frames=9, min_seconds=10, step_seconds=10, max_seconds=50)
    train = PrefixReplayDataset(split="train", **settings)
    validation = PrefixReplayDataset(split="test", **settings)
    assert train.replay_ids.isdisjoint(validation.replay_ids)
    assert len(train) + len(validation) == 20 * 3  # terminal frame at 40s never selected
    first = validation[0]
    torch.rand(17)
    repeated = validation[0]
    assert torch.equal(first["frame_indices"], repeated["frame_indices"])
    assert torch.equal(first["inputs"]["visibility_map"][:, 0, 0], first["frame_indices"])
    assert torch.equal(first["inputs"]["unit_type"], first["inputs"]["visibility_map"] + 5000)
    assert torch.all(first["frame_seconds"] <= first["cutoff_seconds"])
    # Poison all future observations; prediction inputs must stay identical.
    base_index = validation.examples[0][0]
    for layer in validation.arrays(base_index).values():
        layer[11:] = 0
    assert all(torch.equal(value, validation[0]["inputs"][key]) for key, value in first["inputs"].items())
    for item in (train[0], validation[0]):
        assert item["frame_indices"][-1] == 10


def test_invalid_timing_is_rejected(tmp_path):
    make_timed_export(tmp_path)
    np.savez(tmp_path / "both/match0_p1/TimeInfo.npz", faster_seconds=np.zeros(41))
    with pytest.raises(ValueError, match="Invalid timestamps"):
        PrefixReplayDataset(tmp_path, split="all", frames=9, min_seconds=10, max_seconds=30)


class AnalyticModel(nn.Module):
    """A known nonlinear head: d(s^2)/ds makes attribution depend on the input."""
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.pool3 = nn.AdaptiveAvgPool3d(1)
        self.drop_out = nn.Identity()
        self.linear = SquareHead()

    def forward_features(self, inputs):
        return inputs["x"].float().unsqueeze(1) * self.scale

    def forward(self, inputs):
        return self.linear(self.pool3(self.forward_features(inputs)).flatten(1))


class SquareHead(nn.Module):
    def forward(self, features):
        return torch.cat((torch.zeros_like(features), features.square()), dim=1)


def test_gradcam_uses_autograd_through_nonlinear_head_and_preserves_state():
    model = AnalyticModel().train()
    model.scale.grad = torch.tensor(7.)
    patch = torch.zeros((1, 2, 4, 4))
    patch[:, :, :2, :2] = 4
    with torch.no_grad():
        result = grad_cam(model, {"x": patch}, target_class=1)
    expected = (patch > 0).float()
    torch.testing.assert_close(result["heatmap"], expected)
    assert model.training and model.scale.grad.item() == 7
    assert result["heatmap"].requires_grad is False
    opposite = grad_cam(model, {"x": patch}, target_class=0)["heatmap"]
    assert torch.isfinite(opposite).all() and torch.count_nonzero(opposite) == 0
    model.requires_grad_(False)
    torch.testing.assert_close(grad_cam(model, {"x": patch})["heatmap"], expected)


def test_masking_highlighted_cells_removes_known_signal_and_preserves_inputs():
    model = AnalyticModel().train()
    patch = torch.zeros((1, 2, 4, 4))
    patch[:, :, :2, :2] = 4
    original = patch.clone()
    heatmap = grad_cam(model, {"x": patch})["heatmap"]
    result = masking_check(model, {"x": patch.transpose(2, 3)}, heatmap.transpose(2, 3), fraction=0.25)
    assert result["highlighted_probability_drop"] > result["random_probability_drop"]
    assert result["masked_cells"] == 8 and model.training
    torch.testing.assert_close(patch, original)
    assert masking_check(model, {"x": patch}, torch.zeros_like(patch))["available"] is False


def test_pivotal_intervals_do_not_merge_reversals_or_skipped_changes():
    events = pivotal_intervals([0, 60, 120, 180, 300, 360], [.1, .5, .3, .31, .31, .8],
                              threshold=.15, merge_seconds=60)
    assert len(events) == 3
    assert events[0]["start_seconds"] == 0 and events[0]["end_seconds"] == 60
    assert events[0]["net_change"] == pytest.approx(.4)
    assert events[1]["start_seconds"] == 60 and events[1]["end_seconds"] == 120
    assert events[1]["net_change"] == pytest.approx(-.2)
    assert events[1]["changes"] == 1
    assert events[2]["start_seconds"] == 300 and events[2]["end_seconds"] == 360
    assert pivotal_intervals([0, 60], [.5, .5]) == []
    with pytest.raises(ValueError):
        pivotal_intervals([60, 60], [.2, .6])
    with pytest.raises(ValueError):
        pivotal_intervals([0, 60], [.2, float("nan")])


def test_pivotal_intervals_merge_only_consecutive_same_direction_within_cap():
    events = pivotal_intervals([0, 60, 120, 180, 240, 300],
                              [.1, .2, .3, .4, .5, .35],
                              threshold=.1, merge_seconds=120)
    assert len(events) == 3
    assert (events[0]["start_seconds"], events[0]["end_seconds"]) == (0, 120)
    assert events[0]["changes"] == 2 and events[0]["net_change"] == pytest.approx(.2)
    assert (events[1]["start_seconds"], events[1]["end_seconds"]) == (120, 240)
    assert events[1]["changes"] == 2 and events[1]["net_change"] == pytest.approx(.2)
    assert events[2]["net_change"] == pytest.approx(-.15)


def test_pivotal_intervals_do_not_cross_unqualified_step():
    events = pivotal_intervals([0, 60, 120, 180], [.1, .2, .21, .35],
                              threshold=.1, merge_seconds=120)
    assert len(events) == 2
    assert (events[0]["start_seconds"], events[0]["end_seconds"]) == (0, 60)
    assert (events[1]["start_seconds"], events[1]["end_seconds"]) == (120, 180)
    assert events[1]["changes"] == 1


def test_metrics_and_per_cutoff_loss_are_sample_weighted():
    known = train_temporal.metrics_from_predictions([0, 1], [.25, .75], [1., 2.])
    assert known["brier"] == .0625 and known["accuracy"] == 1
    assert known["loss"] == 1.5
    model = AnalyticModel()
    batches = [dict(inputs={"x": torch.zeros(n, 1, 1, 1)}, targets=torch.tensor(labels),
                    cutoff_seconds=torch.tensor(times))
               for n, labels, times in [(2, [0, 1], [60., 120.]), (1, [0], [60.])]]
    metrics = train_temporal.run_prefix_epoch(model, batches, torch.device("cpu"))
    assert metrics["samples"] == 3 and metrics["brier"] == .25
    assert metrics["by_cutoff_seconds"]["60"]["samples"] == 2
    assert metrics["by_cutoff_seconds"]["120"]["samples"] == 1


class TinyTrainingModel(nn.Module):
    config = {}

    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Dropout(.3), nn.Linear(1, 2))

    def forward(self, inputs):
        return self.head(inputs["visibility_map"].float().mean((1, 2, 3)).unsqueeze(1) / 40)


def test_resume_reproduces_uninterrupted_training_and_restores_rng(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    make_timed_export(data)
    monkeypatch.setattr(train_temporal, "BaekResNet3D", TinyTrainingModel)
    common = ["--root", str(data), "--device", "cpu", "--frames", "9", "--min-seconds", "10",
              "--step-seconds", "10", "--max-seconds", "30", "--threads", "1", "--log-every", "0"]
    train_temporal.main(common + ["--epochs", "2", "--output", str(tmp_path / "whole")])
    train_temporal.main(common + ["--epochs", "1", "--output", str(tmp_path / "part")])
    train_temporal.main(common + ["--epochs", "2", "--output", str(tmp_path / "resumed"),
                                  "--resume", str(tmp_path / "part/last.pt")])
    whole = torch.load(tmp_path / "whole/last.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resumed/last.pt", weights_only=True)
    for key, value in whole["model_state_dict"].items():
        torch.testing.assert_close(value, resumed["model_state_dict"][key], rtol=0, atol=0)
    assert whole["history"][-1]["validation"] == resumed["history"][-1]["validation"]
    assert torch.equal(whole["rng_cpu"], resumed["rng_cpu"])
    assert torch.equal(whole["rng_loader"], resumed["rng_loader"])
    # The selected best can originate in an earlier run of a resume chain.
    best_path = tmp_path / "resumed/best.pt"
    best = torch.load(best_path, weights_only=True)
    best["run_config"]["output"] = str(tmp_path / "earlier-run")
    torch.save(best, best_path)
    train_temporal.main(common + ["--epochs", "3", "--output", str(tmp_path / "resumed-again"),
                                  "--resume", str(tmp_path / "resumed/last.pt")])
    assert json.loads((tmp_path / "resumed-again/status.json").read_text())["epoch"] == 3
    train_temporal.main(common + ["--epochs", "10", "--max-hours", "0.000000001", "--output", str(tmp_path / "budget")])
    status = json.loads((tmp_path / "budget/status.json").read_text())
    assert status["epoch"] == 1 and status["reason"] == "time_budget" and status["state"] == "completed"
    assert (tmp_path / "budget/last.pt").is_file()
