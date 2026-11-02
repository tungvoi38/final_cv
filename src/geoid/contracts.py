"""Executable NumPy boundary checks for the proposed contract, not research code."""

import numpy as np

from geoid.config import require


def array(value, dtype, name):
    require(
        isinstance(value, np.ndarray) and value.dtype == np.dtype(dtype),
        f"{name}: expected {dtype} array",
    )
    require(np.isfinite(value).all(), f"{name}: nonfinite values")
    return value


def validate_sample(sample):
    mask = array(sample.get("mask"), "int64", "mask")
    require(mask.ndim == 2 and all(d > 0 for d in mask.shape), "mask: expected positive H,W")
    require(np.isin(mask, [0, 1, 2, 255]).all(), "mask: invalid class")
    for name, channels in (("sar", 4), ("optical", 12), ("dem", 1)):
        value = array(sample.get(name), "float32", name)
        require(value.shape == (channels, *mask.shape), f"{name}: shape mismatch")
    for key in ("event_id", "aoi_id", "tile_id"):
        require(isinstance(sample.get(key), str) and bool(sample[key].strip()), f"missing {key}")


def valid_target_pixels(mask):
    mask = array(mask, "int64", "mask")
    require(np.isin(mask, [0, 1, 2, 255]).all(), "mask: invalid class")
    return mask != 255


def validate_model_output(output, *, batch_size, height, width):
    logits = array(output.get("logits"), "float32", "logits")
    require(logits.shape == (batch_size, 3, height, width), "logits: shape mismatch")
    features = output.get("features")
    require(isinstance(features, dict), "missing features mapping")
    alignment = array(features.get("alignment"), "float32", "alignment")
    require(
        alignment.ndim == 4
        and alignment.shape[0] == batch_size
        and all(d > 0 for d in alignment.shape),
        "alignment: expected B,D,h,w",
    )


def validate_dg_inputs(features, masks, event_ids, *, split):
    require(split == "train", "DG is training-only")
    features = array(features, "float32", "features")
    masks = array(masks, "int64", "masks")
    require(features.ndim == 4 and all(d > 0 for d in features.shape), "expected B,D,h,w")
    require(masks.shape == (features.shape[0], *features.shape[2:]), "DG masks need feature grid")
    require(np.isin(masks, [0, 1, 2, 255]).all(), "DG invalid mask class")
    require(
        len(event_ids) == features.shape[0]
        and all(isinstance(e, str) and e.strip() for e in event_ids),
        "DG event metadata",
    )
    require(len(set(event_ids)) >= 2, "DG requires distinct source events")


def validate_dg_output(result):
    loss = array(result.get("loss"), "float32", "loss")
    require(loss.shape == (), "DG loss must be scalar")
    metrics = result.get("metrics")
    require(isinstance(metrics, dict), "DG metrics must be a mapping")
    require(
        all(
            isinstance(k, str) and isinstance(v, (int, float)) and np.isfinite(v)
            for k, v in metrics.items()
        ),
        "DG metrics must be finite numbers",
    )
