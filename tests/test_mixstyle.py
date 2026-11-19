"""Test event-aware feature transformation independently of any encoder."""

import pytest

torch = pytest.importorskip("torch")
from src.geoid.dg.mixstyle import MixStyle  # noqa: E402


def test_statistics_formula_and_gradients():
    x = torch.arange(48.0).reshape(2, 3, 2, 4).requires_grad_()
    module = MixStyle(p=1)
    # Two singleton domains force donor mapping; reproduce RNG draws and Beta.
    torch.manual_seed(9)
    torch.randint(1, ())
    torch.randint(1, ())
    weight = torch.distributions.Beta(torch.tensor(0.1), torch.tensor(0.1)).sample((2, 1, 1, 1))
    mu = x.mean((2, 3), keepdim=True).detach()
    sd = (x.var((2, 3), keepdim=True) + 1e-6).sqrt().detach()
    expected = (x - mu) / sd * (sd * weight + sd.flip(0) * (1 - weight))
    expected = expected + mu * weight + mu.flip(0) * (1 - weight)
    torch.manual_seed(9)
    y = module(x, ["a", "b"])
    torch.testing.assert_close(y, expected)
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    torch.manual_seed(9)
    torch.testing.assert_close(module(x, ["a", "b"]), y)


def test_imbalanced_events_and_constant_features():
    module = MixStyle(1)
    x = torch.ones(4, 2, 3, 3, requires_grad=True)
    events = ["a", "a", "a", "b"]
    y = module(x, events)
    assert all(events[i] != events[j] for i, j in enumerate(module.last_info["donor_indices"]))
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("kind", ["eval", "p0", "one_event", "small"])
def test_identity_without_rng(kind):
    x = torch.randn(2, 3, 1, 1) if kind == "small" else torch.randn(2, 3, 2, 2)
    module = MixStyle(0 if kind == "p0" else 1)
    if kind == "eval":
        module.eval()
    events = ["a", "a"] if kind == "one_event" else ["a", "b"]
    state = torch.get_rng_state()
    assert module(x, events) is x
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision(dtype):
    x = torch.randn(2, 3, 4, 4).to(dtype).requires_grad_()
    y = MixStyle(1)(x, ["a", "b"])
    assert y.dtype == dtype and torch.isfinite(y).all()
    y.float().square().mean().backward()
    assert torch.isfinite(x.grad).all()


def test_validation():
    with pytest.raises(ValueError):
        MixStyle(alpha=0)
    with pytest.raises(ValueError):
        MixStyle(1)(torch.zeros(2, 5, 3), ["a", "b"])
