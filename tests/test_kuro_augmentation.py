"""Pixel provenance and isolation tests, no downloaded data."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from kuro_augmentation import ClassMix  # noqa: E402


def batch():
    labels = torch.tensor([[[0, 1], [2, 255]], [[2, 0], [1, 255]]])
    return dict(
        sar=torch.arange(48.0).reshape(2, 3, 2, 2, 2),
        dem=torch.arange(8.0).reshape(2, 1, 2, 2),
        mask=labels,
        input_valid=torch.ones_like(labels, dtype=torch.bool),
        valid_mask=labels != 255,
        event_id=["a", "b"],
        split=["train", "train"],
        sample_id=["1", "2"],
    )


def test_provenance_repeatability_and_no_mutation():
    source = batch()
    before = {k: v.clone() for k, v in source.items() if isinstance(v, torch.Tensor)}
    torch.manual_seed(42)
    out, info = ClassMix(1)(source, training=True)
    assert info["applied"] == 2
    for r in info["records"]:
        i, j = r["recipient_index"], r["donor_index"]
        m = info["mix_mask"][i]
        assert r["recipient_event"] != r["donor_event"]
        assert len(r["classes"]) == 1
        for key in before:
            assert torch.equal(out[key][i], torch.where(m, source[key][j], source[key][i]))
    assert torch.equal(out["valid_mask"], out["mask"] != 255)
    for key in before:
        assert torch.equal(source[key], before[key])
    torch.manual_seed(42)
    again, _ = ClassMix(1)(source, training=True)
    assert torch.equal(out["sar"], again["sar"])


def test_disabled_rng_and_split_guard():
    source = batch()
    state = torch.get_rng_state()
    assert ClassMix(0)(source, training=True)[0] is source
    assert ClassMix(1)(source, training=False)[0] is source
    assert torch.equal(state, torch.get_rng_state())
    source["split"][1] = "test"
    with pytest.raises(ValueError, match="training"):
        ClassMix(1)(source, training=True)


@pytest.mark.parametrize("case", ["same_event", "one_class", "ignore"])
def test_skips(case):
    source = batch()
    if case == "same_event":
        source["event_id"] = ["a", "a"]
    else:
        source["mask"].fill_(0 if case == "one_class" else 255)
        source["valid_mask"] = source["mask"] != 255
    out, info = ClassMix(1)(source, training=True)
    assert info["applied"] == 0
    assert torch.equal(out["sar"], source["sar"])


def test_invalid_validity_rejected():
    source = batch()
    source["input_valid"].zero_()
    with pytest.raises(ValueError, match="Supervision"):
        ClassMix(1)(source, training=True)


def test_eligible_donor_selected_before_sampling():
    source = batch()
    for key, value in list(source.items()):
        if isinstance(value, torch.Tensor):
            source[key] = torch.cat([value, value[:1].clone()], dim=0)
    source["event_id"] = ["a", "b", "c"]
    source["sample_id"] = ["1", "2", "3"]
    source["split"] = ["train"] * 3
    source["mask"][1].fill_(0)
    source["mask"][1, 0, 0] = 255  # ignore is not a second class
    source["valid_mask"] = source["mask"] != 255
    for seed in range(10):
        torch.manual_seed(seed)
        _, info = ClassMix(1)(source, training=True)
        assert info["applied"] == 3
        assert all(r["donor_index"] != 1 for r in info["records"])
        assert info["records"][0]["donor_index"] == 2
        assert info["records"][2]["donor_index"] == 0


def test_eligible_same_event_cannot_be_donor():
    source = batch()
    source["mask"][1].fill_(0)
    source["valid_mask"] = source["mask"] != 255
    _, info = ClassMix(1)(source, training=True)
    assert info["records"][0]["reason"] == "no_eligible_donor"
    assert info["records"][1]["reason"] == "applied"
