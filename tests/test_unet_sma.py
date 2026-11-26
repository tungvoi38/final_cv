"""Tests for the A3 SMA recurrence, continuation and logit ensemble."""

import importlib
import json
import random
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from scripts import unet_sma  # noqa: E402


def test_official_count_and_batchnorm_update():
    online = torch.nn.BatchNorm2d(2)
    average = unet_sma.SimpleMovingAverage(online)
    with torch.no_grad():
        for step in range(1, 102):
            online.weight.fill_(step)
            online.running_mean.fill_(2 * step)
            online.running_var.fill_(3 * step)
            online.num_batches_tracked.fill_(step)
            average.update()
            if step == 99:
                assert average.count == 0
                assert average.model.weight[0].item() == 99
            if step == 100:
                assert average.count == 1
                assert average.model.weight[0].item() == pytest.approx(99.5)
    assert average.steps == 101 and average.count == 2
    torch.testing.assert_close(average.model.weight, torch.full((2,), 100.0))
    torch.testing.assert_close(average.model.running_mean, torch.full((2,), 200.0))
    torch.testing.assert_close(average.model.running_var, torch.full((2,), 300.0))
    assert average.model.num_batches_tracked.item() == 0
    assert not average.model.training and not average.model.weight.requires_grad
    x = torch.zeros(1, 2, 2, 2)
    before = average.model.running_mean.clone()
    assert torch.isfinite(average.model(x)).all()
    torch.testing.assert_close(average.model.running_mean, before)
    broken = dict(average.state_dict(), count=3)
    with pytest.raises(ValueError, match="averaging count"):
        average.load_state_dict(broken)


def test_overflow_skips_sma_and_scheduler():
    pytest.importorskip("tqdm")
    from scripts.train_unet import step_optimizer

    model = torch.nn.Linear(1, 1)
    averaging = unet_sma.SimpleMovingAverage(model)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 10)
    scaler = torch.amp.GradScaler("cpu", init_scale=8)
    loss = model(torch.ones(1, 1)).sum()
    scaler.scale(loss).backward()
    if step_optimizer(model, opt, scaler, sched):
        averaging.update()
    opt.zero_grad(set_to_none=True)
    scaler.scale(model(torch.ones(1, 1)).sum() * float("inf")).backward()
    if step_optimizer(model, opt, scaler, sched):
        averaging.update()
    assert averaging.steps == sched.last_epoch == 1
    assert scaler.get_scale() == 4


def test_sma_resume_at_epoch_boundary(tmp_path):
    pytest.importorskip("tqdm")
    from scripts.train_unet import step_optimizer

    config = {"epochs": 2, "run_id": "synthetic-sma"}
    logs = Mock()
    logs.start_run.side_effect = lambda **kwargs: nullcontext(
        SimpleNamespace(info=SimpleNamespace(run_id="fake-id"))
    )

    def run(directory, *, resume=False, interrupt=False):
        if not resume:
            directory.mkdir()
            (directory / "checkpoints").mkdir()
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 101)
        scaler = torch.amp.GradScaler("cpu", init_scale=8, growth_interval=1000)
        generators = {
            name: torch.Generator().manual_seed(seed)
            for seed, name in enumerate(("train_loader", "validation_loader", "train_sampler"))
        }
        draws = []

        def train(epoch, average):
            if epoch == 2 and interrupt:
                raise InterruptedError
            for _ in range(99 if epoch == 1 else 2):
                rnd = random.random() + np.random.rand() + torch.rand(()).item()
                sample = torch.rand((1, 2), generator=generators["train_sampler"])
                worker = torch.rand((), generator=generators["train_loader"]).item()
                draws.append((rnd, sample.tolist(), worker))
                optimizer.zero_grad(set_to_none=True)
                loss = model(sample).sum() * (rnd + worker)
                scaler.scale(loss).backward()
                if step_optimizer(model, optimizer, scaler, scheduler):
                    average.update()
            return loss.item()

        def validate(m):
            torch.rand((), generator=generators["validation_loader"])
            score = torch.sigmoid(m.weight.mean()).item()
            return {"pooled_miou": score}, {"1": {"flood_iou": score}}

        unet_sma.train_sma(
            directory,
            config,
            model,
            train,
            validate,
            optimizer,
            scheduler,
            scaler,
            generators,
            logs,
            resume=resume,
        )
        checkpoint = torch.load(
            directory / "checkpoints/epoch_002.pt", weights_only=True, map_location="cpu"
        )
        return draws[-2:], checkpoint

    uninterrupted, a = run(tmp_path / "whole")
    directory = tmp_path / "restart"
    with pytest.raises(InterruptedError):
        run(directory, interrupt=True)
    assert not (directory / "best.pt").exists()
    (directory / "history.csv").write_text("interrupted publication")
    resumed, b = run(directory, resume=True)
    assert resumed == uninterrupted
    assert a["journal"] == b["journal"]
    assert a["sma"]["steps"] == b["sma"]["steps"] == 101
    assert a["sma"]["count"] == b["sma"]["count"] == 2
    for kind in ("model", "online_model"):
        for name in a[kind]:
            torch.testing.assert_close(a[kind][name], b[kind][name], rtol=0, atol=0)
    for name in ("scheduler", "scaler"):
        assert a["training_state"][name] == b["training_state"][name]
    for key, state in a["training_state"]["optimizer"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, b["training_state"]["optimizer"]["state"][key][name])
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["best_epoch"] == 2 and (directory / "best_online.pt").is_file()
    assert "interrupted" not in (directory / "history.csv").read_text()


def evaluator(monkeypatch):
    for module in ("tqdm", "pyarrow", "rasterio"):
        pytest.importorskip(module)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("evaluate_unet_eoa")


def test_pixel_logits_are_averaged_before_argmax(monkeypatch):
    module = evaluator(monkeypatch)

    class Fixed(torch.nn.Module):
        def __init__(self, logits):
            super().__init__()
            self.logits = torch.tensor(logits, dtype=torch.float16)[None, :, None, :]

        def forward(self, inputs):
            assert inputs.shape == (1, 7, 1, 2)
            return self.logits

    members = [Fixed([[8, 0], [0, 0], [0, 7]])]
    members += [Fixed([[0, 1], [1, 0], [0, 0]]) for _ in range(2)]
    input_tensor = torch.zeros(1, 7, 1, 2)
    out = module.LogitEnsemble(members)(input_tensor)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, torch.stack([m.logits.float() for m in members]).mean(0))
    assert out.argmax(1).tolist() == [[[0, 2]]]
    with pytest.raises(ValueError, match="Misaligned"):
        module.LogitEnsemble([torch.nn.Identity()])(input_tensor)


def test_fixed_flood_bias_preserves_non_flood_tie_breaking(monkeypatch):
    module = evaluator(monkeypatch)
    logits = torch.tensor([[[[1, 0, 1, 0]], [[0, 1, 1, 1]], [[0, 0.2, 0, 1]]]])
    assert module.predict_with_flood_bias(logits, 0.95).tolist() == [[[0, 2, 0, 2]]]
    assert module.predict_with_flood_bias(logits, 1.0).tolist() == [[[0, 2, 0, 2]]]
    torch.testing.assert_close(module.predict_with_flood_bias(logits, 0), logits.argmax(1))


@pytest.mark.parametrize("legacy", (False, True))
def test_calibrated_cli_defaults_to_validation(monkeypatch, tmp_path, legacy):
    import sys

    module = evaluator(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    argv = ["evaluate_unet_eoa.py", "calibrated", "--out", str(tmp_path)]
    if legacy:
        argv.append("--no-tta")
    monkeypatch.setattr(sys, "argv", argv)

    def check(args, device):
        assert args.evaluation_split == "validation"
        assert args.out == tmp_path
        assert device.type == "cuda"
        config = module.load_json(args.calibration)
        assert config["flood_logit_bias"] == 0.95
        assert ("flip_axes" in config) == (not legacy)

    monkeypatch.setattr(module, "evaluate_calibrated", check)
    module.main()


def test_flip_tta_ensemble_undoes_flips_before_mean(monkeypatch):
    evaluator(monkeypatch)
    module = importlib.import_module("evaluate_unet_eoa_flip_tta")
    inputs = torch.arange(12).reshape(1, 1, 3, 4).float()

    class SpatialModel(torch.nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale

        def forward(self, x):
            return (self.scale * x + inputs).expand(-1, 3, -1, -1)

    models = [SpatialModel(scale) for scale in (1, 2, 6)]
    expected = (3 * inputs + inputs.mean()).expand(-1, 3, -1, -1)
    torch.testing.assert_close(module.FlipTTAEnsemble(models)(inputs), expected)


def test_pre_variability_eoa_uses_frozen_eleven_channel_layout(monkeypatch, tmp_path):
    module = evaluator(monkeypatch)
    protocol = module.load_json(module.REPO / "configs/experiments/unet_a3_eoa.json")
    protocol["member_training_overrides"] = dict(module.PRE_VARIABILITY_MEMBER_TRAINING)
    protocol["member_input_order"] = [*protocol["input_order"], *module.PRE_VARIABILITY_CHANNELS]
    module.check_protocol(protocol)
    protocol["member_input_order"][-1] = "VV"
    with pytest.raises(ValueError, match="member protocol"):
        module.check_protocol(protocol)

    def make_model(head, *, encoder_weights, sar_delta, pre_variability):
        assert head == "a3" and encoder_weights is None and sar_delta and pre_variability
        return torch.nn.Conv2d(11, 3, 1)

    monkeypatch.setattr(module, "build_model", make_model)
    checkpoint = tmp_path / "member.pt"
    model = torch.nn.Conv2d(11, 3, 1)
    torch.save({
        "epoch": 1, "config": module.PRE_VARIABILITY_MEMBER_TRAINING,
        "model": model.state_dict(),
    }, checkpoint)
    member = {"epoch": 1, "checkpoint": str(checkpoint), "head": "a3",
              "weights_key": "model", "input_channels": 11}
    loaded = module.load_model(member, "cpu")
    batch = {"sar": torch.rand(1, 3, 2, 224, 224), "dem": torch.rand(1, 1, 224, 224)}
    inputs, _ = module.pre_variability_inputs(batch)
    assert inputs.shape == (1, 11, 224, 224)
    ensemble = module.LogitEnsemble([loaded] * 3)
    torch.testing.assert_close(ensemble(inputs), model(inputs))


def test_freeze_requires_complete_validation_and_detects_changed_checkpoint(monkeypatch, tmp_path):
    module = evaluator(monkeypatch)
    protocol = module.load_json(module.REPO / "configs/experiments/unet_a3_eoa.json")
    reference = Path(protocol["reference_run"])
    if reference.is_dir():
        cfg, summary, _ = module.check_run(reference, protocol, averaged=False)
        original = module.freeze_member(reference, cfg, summary, kind="reference")
        assert original["epoch"] == summary["best_epoch"]

    # Use a local checkpoint to test the digest gate without modifying a real run.
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixed")
    member = {"checkpoint": str(checkpoint), "sha256": module.digest(checkpoint)}
    selection = {
        "protocol": protocol,
        "evaluation_code_sha256": module.evaluation_code_hashes(),
        "original_erm": member,
        "online": [member] * 3,
        "sma": [member] * 3,
    }
    module.verify_frozen(selection)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frozen checkpoint changed"):
        module.verify_frozen(selection)


def test_test_command_requires_prior_validation(monkeypatch, tmp_path):
    import sys

    module = evaluator(monkeypatch)
    (tmp_path / "selection.json").write_text("{}")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sys, "argv", ["evaluate_unet_eoa.py", "test", "--out", str(tmp_path)])
    with pytest.raises(FileNotFoundError, match="validation_metrics.json"):
        module.main()
    assert not (tmp_path / "test").exists()
