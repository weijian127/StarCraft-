import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from data_processing.dataset import _test_match_ids
from models.baek import BaekResNet3D, EmbeddedGaussianNonLocal3D, FEATURES
from training.train import evaluate, load_model, run_epoch, save_checkpoint


@pytest.fixture(autouse=True)
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def inputs(unit_id=7):
    return {name: torch.full((1, 10, 64, 64), value, dtype=torch.long)
            for name, value in zip(FEATURES, (2, 1, unit_id))}


def test_nonlocal_identity_and_attention_gradients():
    block = EmbeddedGaussianNonLocal3D(8)
    x = torch.randn(2, 8, 3, 4, 4, requires_grad=True)
    torch.testing.assert_close(block(x), x)
    # Once the zero-initialized output gate opens, all attention projections learn.
    with torch.no_grad():
        block.W[1].weight.fill_(1)
    block(x).square().mean().backward()
    for projection in (block.theta, block.phi[0], block.g[0], block.W[0]):
        assert torch.isfinite(projection.weight.grad).all()
        assert projection.weight.grad.abs().sum() > 0


def test_full_model_raw_ids_backward_and_input_order():
    torch.manual_seed(4)
    model = BaekResNet3D().eval()
    batch = inputs(65535)
    logits = model(batch)
    assert logits.shape == (1, 2)
    with torch.no_grad():
        reordered = model(dict(reversed(list(batch.items()))))
    torch.testing.assert_close(logits, reordered)
    loss = nn.functional.cross_entropy(logits, torch.tensor([1]))
    loss.backward()
    assert model.conv1.weight.grad.abs().sum() > 0
    embedding = model.embeddings["unit_type"].embed_fn.weight
    assert embedding.grad[65535].abs().sum() > 0
    assert torch.count_nonzero(embedding.grad[0]) == 0
    before = model.linear.weight.detach().clone()
    torch.optim.SGD(model.parameters(), lr=0.01).step()
    assert not torch.equal(before, model.linear.weight)


@pytest.mark.parametrize("feature,value", [("unit_type", -1), ("visibility_map", 4), ("player_relative", 5)])
def test_bad_categories_fail_with_feature_name(feature, value):
    model = BaekResNet3D(unit_type_vocab_size=8)
    batch = inputs()
    batch[feature][0, 0, 0, 0] = value
    with pytest.raises(ValueError, match=feature):
        model(batch)


def test_checkpoint_roundtrip_and_frozen_embeddings(tmp_path):
    model = BaekResNet3D(unit_type_vocab_size=8, freeze_embeddings=True).eval()
    assert not any(p.requires_grad for p in model.embeddings.parameters())
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad))
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, optimizer, scaler, epoch=1, metrics={}, run_config={})
    restored = load_model(path)
    with torch.no_grad():
        torch.testing.assert_close(model(inputs()), restored(inputs()))
    assert restored.config == model.config
    assert not restored.training


def test_stratified_split_keeps_outcomes_and_is_order_independent():
    labels = {str(index): int(index >= 6) for index in range(11)}
    held_out = _test_match_ids(labels, 0.2, 42, True)
    assert len(held_out) == 2
    assert {labels[rid] for rid in held_out} == {0, 1}
    assert {labels[rid] for rid in labels if rid not in held_out} == {0, 1}
    assert held_out == _test_match_ids(dict(reversed(list(labels.items()))), 0.2, 42, True)
    with pytest.raises(ValueError, match="Stratification"):
        _test_match_ids({"one": 0, "two": 1}, 0.2, 42, True)


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 2)

    def forward(self, data):
        return self.linear(data["x"].float().reshape(-1, 1))


def test_epoch_weights_loss_by_samples_and_evaluation_does_not_train():
    model = TinyClassifier()
    batches = [
        {"inputs": {"x": torch.tensor([0.0, 1.0])}, "targets": torch.tensor([0, 1])},
        {"inputs": {"x": torch.tensor([2.0])}, "targets": torch.tensor([0])},
    ]
    state = {key: value.clone() for key, value in model.state_dict().items()}
    expected = nn.functional.cross_entropy(model({"x": torch.tensor([0., 1., 2.])}), torch.tensor([0, 1, 0]))
    metrics = run_epoch(model, batches, torch.device("cpu"))
    assert metrics["loss"] == pytest.approx(expected.item())
    assert metrics["samples"] == 3
    assert sum(map(sum, metrics["confusion_matrix"])) == 3
    assert all(torch.equal(state[key], value) for key, value in model.state_dict().items())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    run_epoch(model, batches, torch.device("cpu"), optimizer=optimizer)
    assert not torch.equal(state["linear.weight"], model.linear.weight)


def random_collate(batch):
    return {"inputs": {"x": torch.rand(len(batch))}, "targets": torch.tensor(batch)}


def test_validation_sampling_is_repeatable_and_preserves_training_rng():
    loader = DataLoader([0, 1, 0], batch_size=2, collate_fn=random_collate,
                        generator=torch.Generator().manual_seed(1))
    model = TinyClassifier()
    state = torch.get_rng_state().clone()
    first = evaluate(model, loader, torch.device("cpu"), seed=123)
    assert torch.equal(torch.get_rng_state(), state)
    torch.rand(10)
    second = evaluate(model, loader, torch.device("cpu"), seed=123)
    assert first == second
