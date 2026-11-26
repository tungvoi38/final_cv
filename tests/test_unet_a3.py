"""CPU checks for the standalone A3 training and evaluation path."""

import math
from collections import Counter
from contextlib import nullcontext

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tqdm")
from scripts import train_unet  # noqa: E402


def test_preprocessing_masks_channel_order_and_d4():
    pytest.importorskip("pyarrow")
    pytest.importorskip("rasterio")
    from scripts import kuro_dataset as data

    sar_db = np.array([[-10, -20], [-30, -40], [-50, -60]], dtype=np.float32)
    linear = 10 ** (sar_db / 10)
    arrays = {
        time: np.broadcast_to(linear[i, :, None, None], (2, 224, 224)).copy()
        for i, time in enumerate(data.TIMES)
    }
    arrays["dem"] = np.zeros((1, 224, 224), dtype=np.float32)
    arrays["mask"] = np.zeros((1, 224, 224), dtype=np.uint8)
    arrays["invalid_data"] = np.ones_like(arrays["mask"])
    arrays["invalid_data"][0, 0, 1] = 0
    arrays["dem"][0, 0, 3] = 1e30
    arrays["mask"][0, 0, 4:7] = [3, 1, 2]
    flags = {time: np.ones((2, 224, 224), dtype=bool) for time in data.TIMES}
    flags[data.TIMES[1]][0, 0, 2] = False

    sample = data.preprocess(arrays, flags, {"dem": {"nodata": 0}})
    data.validate_arrays(sample)
    np.testing.assert_allclose(sample["sar_db"][:, :, 0, 0], sar_db)
    np.testing.assert_array_equal(sample["mask"][0, :7], [0, 255, 255, 255, 255, 1, 2])
    assert sample["input_valid"][0, 4] and not sample["valid_mask"][0, 4]
    transformed = data.transform_sample(sample)
    expected = ((sar_db - [-12.599, -20.293]) / [5.195, 5.89]).ravel()
    batch = {name: torch.from_numpy(transformed[name])[None] for name in ("sar", "dem")}
    inputs, _ = data.model_inputs(batch)
    assert inputs.shape == (1, 7, 224, 224)
    np.testing.assert_allclose(
        inputs[0, :, 0, 0], np.r_[expected, -670.665 / 951.272], rtol=1e-6, atol=1e-6
    )
    assert torch.count_nonzero(inputs[0, :, 0, 1:4]) == 0
    with_delta = train_unet.append_sar_delta(inputs)
    assert with_delta.shape == (1, 9, 224, 224)
    torch.testing.assert_close(with_delta[:, :7], inputs)
    np.testing.assert_allclose(with_delta[0, 7:, 0, 0], [-30 / 5.195, -30 / 5.890],
                               rtol=1e-6, atol=1e-6)
    assert torch.count_nonzero(with_delta[0, :, 0, 1:4]) == 0

    for d4 in range(8):
        actual = data.transform_sample(sample, d4)
        for name in ("sar", "dem", "mask", "input_valid", "valid_mask"):
            rotated = np.rot90(transformed[name], d4 % 4, axes=(-2, -1))
            expected = np.flip(rotated, axis=-1) if d4 >= 4 else rotated
            np.testing.assert_array_equal(actual[name], expected)
    for split in ("validation", "test"):
        with pytest.raises(ValueError, match="Augmentation is allowed only for train"):
            data.KuroDataset(None, split, augment=True)


def test_a3_head_is_one_convolution_with_full_resolution_logits():
    pytest.importorskip("segmentation_models_pytorch")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = train_unet.build_model("a3", encoder_weights=None).eval()
        head = model.segmentation_head
        assert isinstance(head, torch.nn.Conv2d)
        assert (head.in_channels, head.out_channels) == (16, 3)
        assert head.kernel_size == (1, 1) and head.padding == (0, 0)
        assert not list(head.children())
        assert tuple(model.encoder.conv1.weight.shape) == (64, 7, 7, 7)
        assert tuple(model.state_dict()["segmentation_head.weight"].shape) == (3, 16, 1, 1)
        with torch.inference_mode():
            logits = model(torch.zeros(1, 7, 224, 224))
        assert logits.shape == (1, 3, 224, 224) and torch.isfinite(logits).all()
        legacy = train_unet.build_model("standard", encoder_weights=None)
        assert legacy.segmentation_head[0].kernel_size == (3, 3)
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize("variability,channels", [(False, 9), (True, 11)])
def test_sar_delta_stem_preserves_initialization_and_learns_extra_kernels(variability, channels):
    pytest.importorskip("segmentation_models_pytorch")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        original = train_unet.build_model("a3", encoder_weights=None).eval()
        expected_rng = torch.get_rng_state()
        torch.manual_seed(42)
        expanded = train_unet.build_model(
            "a3", encoder_weights=None, sar_delta=True, pre_variability=variability
        ).eval()
        assert torch.equal(torch.get_rng_state(), expected_rng)
        for key, tensor in original.state_dict().items():
            if key == "encoder.conv1.weight":
                torch.testing.assert_close(expanded.state_dict()[key][:, :7], tensor, rtol=0, atol=0)
            else:
                torch.testing.assert_close(expanded.state_dict()[key], tensor, rtol=0, atol=0)
        stem = expanded.encoder.conv1
        assert stem.in_channels == expanded.encoder.out_channels[0] == channels
        assert torch.count_nonzero(stem.weight[:, 7:]) == 0
        inputs = torch.rand(2, 7, 64, 64)
        with_delta = train_unet.append_sar_delta(inputs, pre_variability=variability)
        with torch.inference_mode():
            expected = original(inputs)
            actual = expanded(with_delta)
        assert actual.shape == (2, 3, 64, 64)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        stem(with_delta).square().mean().backward()
        assert torch.count_nonzero(stem.weight.grad[:, 7:]) > 0
        with pytest.raises(ValueError, match="A3 head"):
            train_unet.build_model("standard", encoder_weights=None, sar_delta=True)


def test_pre_variability_appends_absolute_differences_in_vv_vh_order():
    inputs = torch.tensor([1, 7, 3, 2, 10, 12, 20], dtype=torch.float32).view(1, 7, 1, 1)
    expanded = train_unet.append_sar_delta(inputs, pre_variability=True)
    torch.testing.assert_close(expanded[:, :9], train_unet.append_sar_delta(inputs), rtol=0, atol=0)
    assert expanded[0, 9:, 0, 0].tolist() == [2, 5]
    neutral = train_unet.append_sar_delta(torch.zeros_like(inputs), pre_variability=True)
    assert torch.count_nonzero(neutral) == 0


def test_event_sampler_equal_mass_4000_draws_with_replacement():
    events = ["a"] * 2500 + ["b"] * 1000 + ["c"] * 500
    sampler = train_unet.event_balanced_sampler(events, 42)
    assert sampler.replacement and len(sampler) == 4000
    for start, end in ((0, 2500), (2500, 3500), (3500, 4000)):
        assert sampler.weights[start:end].sum().item() == pytest.approx(1)
        assert sampler.weights[start].item() == pytest.approx(1 / (end - start))

    rng_before = torch.random.get_rng_state()
    first = list(sampler)
    assert len(first) == 4000 and len(set(first)) < 4000
    assert list(train_unet.event_balanced_sampler(events, 42)) == first
    assert list(sampler) != first
    assert torch.equal(rng_before, torch.random.get_rng_state())
    draws = Counter(events[index] for index in first)
    assert all(abs(count - 4000 / 3) < 150 for count in draws.values())


def test_square_root_sampling_preserves_draw_count_and_increases_diversity():
    events = ["large"] * 1542 + ["medium"] * 56 + ["small"] * 2
    exponent = train_unet.EVENT_SAMPLING_EXPONENTS["event_sqrt"]
    sampler = train_unet.event_balanced_sampler(events, 42, exponent=exponent)
    assert sampler.replacement and len(sampler) == len(events)
    for start, end in ((0, 1542), (1542, 1598), (1598, 1600)):
        assert sampler.weights[start].item() == pytest.approx(1 / math.sqrt(end - start))
        assert sampler.weights[start:end].sum().item() == pytest.approx(math.sqrt(end - start))
    draws = list(sampler)
    assert draws == list(train_unet.event_balanced_sampler(events, 42, exponent=exponent))
    baseline_draws = list(train_unet.event_balanced_sampler(events, 42))
    assert len(set(draws)) > len(set(baseline_draws))


def test_ce_dice_ignore_and_absent_class():
    logits = torch.zeros(1, 3, 1, 3, requires_grad=True)
    labels = torch.tensor([[[0, 2, 255]]])
    loss = train_unet.loss_fn(logits, labels)
    expected = math.log(3) + 1 - (5 / 8 + 3 / 5 + 5 / 8) / 3
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert torch.count_nonzero(logits.grad[..., 2]) == 0

    logits.grad.zero_()
    empty = train_unet.loss_fn(logits, torch.full_like(labels, 255))
    assert empty.item() == 0
    empty.backward()
    assert torch.count_nonzero(logits.grad) == 0


def test_scheduler_and_clipping_follow_successful_amp_steps():
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    batches = len(torch.utils.data.DataLoader(range(4000), batch_size=2))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 15 * batches)
    scaler = torch.amp.GradScaler("cpu", init_scale=8, growth_interval=1)
    assert scheduler.T_max == 30_000

    scaler.scale(100 * model.weight.sum()).backward()
    train_unet.step_optimizer(model, optimizer, scaler, scheduler)
    assert model.weight.grad.norm().item() == pytest.approx(1)
    assert scaler.get_scale() == 16 and scheduler.last_epoch == 1
    lr = optimizer.param_groups[0]["lr"]
    expected = 1e-4 * (1 + math.cos(math.pi / 30_000)) / 2
    assert lr == pytest.approx(expected, rel=1e-12)

    optimizer.zero_grad(set_to_none=True)
    weight = model.weight.detach().clone()
    scaler.scale(float("inf") * model.weight.sum()).backward()
    train_unet.step_optimizer(model, optimizer, scaler, scheduler)
    assert scaler.get_scale() == 8 and scheduler.last_epoch == 1
    assert optimizer.param_groups[0]["lr"] == lr
    torch.testing.assert_close(model.weight, weight)


def test_validation_metrics_ignore_pixels_and_pool_counts(monkeypatch):
    y = torch.tensor([[[0, 1, 2, 2, 255, 0]], [[0, 0, 1, 2, 2, 0]]])
    pred = torch.tensor([[[2, 2, 2, 1, 2, 2]], [[1, 0, 1, 2, 2, 0]]])
    mask = torch.ones_like(y, dtype=torch.bool)
    mask[0, 0, 5] = False
    batch = {
        "sar": torch.nn.functional.one_hot(pred, 3).permute(0, 3, 1, 2).float(),
        "dem": torch.empty(2, 1, 1, 6),
        "mask": y,
        "valid_mask": mask,
        "event_id": ["a", "b"],
    }
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    metrics, events = train_unet.validate_with_events(
        torch.nn.Identity(), [batch], torch.device("cpu"), lambda b: (b["sar"], None)
    )
    assert metrics["pooled_miou"] == pytest.approx((1 / 2 + 1 / 4 + 1 / 2) / 3)
    assert metrics["pooled_flood_precision"] == pytest.approx(3 / 5)
    assert metrics["pooled_flood_recall"] == pytest.approx(3 / 4)
    assert metrics["pooled_flood_f1"] == pytest.approx(2 * 3 / 9)
    assert metrics["event_macro_flood_iou"] == pytest.approx((1 / 4 + 1) / 2)
    assert (events["a"]["flood_tp"], events["a"]["flood_fp"], events["a"]["flood_fn"]) == (1, 2, 1)
    assert "permanent_water_as_flood_pixels" not in metrics


def test_water_confusion_uses_true_rows_and_ignores_unlabelled_pixels(monkeypatch):
    y = torch.tensor([[[1, 1, 2, 255, 1]]])
    pred = torch.tensor([[[2, 2, 1, 2, 2]]])
    batch = {
        "sar": torch.nn.functional.one_hot(pred, 3).permute(0, 3, 1, 2).float(),
        "dem": torch.empty(1, 1, 1, 5),
        "mask": y,
        "valid_mask": torch.tensor([[[True, True, True, True, False]]]),
        "event_id": ["a"],
    }
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    metrics, _ = train_unet.validate_with_events(
        torch.nn.Identity(), [batch], torch.device("cpu"), lambda b: (b["sar"], None),
        report_water_confusion=True,
    )
    assert metrics["permanent_water_as_flood_pixels"] == 2
    assert metrics["flood_as_permanent_water_pixels"] == 1
