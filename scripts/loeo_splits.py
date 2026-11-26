"""Deterministic, event-disjoint development folds with geographic overlap purging."""
import hashlib

import numpy as np

from kuro_dataset import RAW_SHA256, REVISION


SETTINGS = {
    "epochs": 20, "batch_size": 2, "workers": 2, "encoder_lr": 3e-5, "decoder_lr": 1e-4,
    "weight_decay": .01, "optimizer": "adamw", "loss": "ce_dice", "head": "a3",
    "sar_delta": True, "sar_pre_variability": True, "sampling": "event_sqrt",
    "averaging": "sma", "selection": "pooled_miou",
}
SEEDS = (42, 43, 44)
DRAWS = 4000


def development(frame):
    column = "tortilla:data_split"
    events = {split: set(frame.loc[frame[column] == split, "actid"].astype(int))
              for split in ("train", "validation", "test")}
    if tuple(map(len, events.values())) != (27, 6, 10) or len(set.union(*events.values())) != 43:
        raise ValueError("Expected disjoint 27 train, 6 validation and 10 test events")
    rows = frame[frame[column].isin(("train", "validation"))].reset_index(drop=True)
    if len(rows) != 5000 or rows["tortilla:id"].astype(str).duplicated().any():
        raise ValueError("Expected 5000 unique development patches")
    return rows, events


def footprint_neighbors(rows):
    affine = np.stack(rows["stac:geotransform"])
    if (not rows["stac:crs"].eq("EPSG:3857").all()
            or not np.all(affine[:, [1, 2, 4, 5]] == [10, 0, 0, -10])
            or any(tuple(shape) != (224, 224) for shape in rows["stac:raster_shape"])):
        raise ValueError("Unexpected patch grid for the footprint audit")
    corner = affine[:, [0, 3]]
    event = rows.actid.to_numpy()
    return [np.flatnonzero((event != event[i]) & (np.abs(corner - xy) < 2240 - 1e-6).all(1))
            for i, xy in enumerate(corner)]


def replacement_key(target, candidate):
    return hashlib.sha256(f"kuro-loeo-inner-v1:{target}:{candidate}".encode()).digest()


def make_fold(rows, events, neighbors, target):
    if target not in events["train"] | events["validation"]:
        raise ValueError(f"Target {target} is not a development event")
    inner = events["validation"] - {target}
    if len(inner) < 6:
        inner.add(min(events["train"], key=lambda e: replacement_key(target, e)))
    train = (events["train"] | events["validation"]) - inner - {target}
    target_idx = set(np.flatnonzero(rows.actid.to_numpy() == target))
    target_overlap = set().union(*(set(neighbors[i]) for i in target_idx))
    inner_idx = set(np.flatnonzero(rows.actid.isin(inner).to_numpy()))
    retained_inner = inner_idx - target_overlap
    inner_overlap = set().union(*(set(neighbors[i]) for i in retained_inner))
    train_idx = set(np.flatnonzero(rows.actid.isin(train).to_numpy()))
    retained_train = train_idx - target_overlap - inner_overlap
    masks = {"source_train": retained_train, "inner_validation": retained_inner, "target": target_idx,
             "excluded_target_overlap": (train_idx | inner_idx) & target_overlap,
             "excluded_inner_overlap": (train_idx - target_overlap) & inner_overlap}
    if [len(set(rows.iloc[sorted(ids)].actid)) for ids in (retained_train, retained_inner, target_idx)] != [26, 6, 1]:
        raise ValueError("Footprint purge removed an entire event")
    assert sum(map(len, masks.values())) == len(rows) == len(set.union(*masks.values()))
    for first, second in ((retained_train, retained_inner), (retained_train, target_idx),
                          (retained_inner, target_idx)):
        if any(second.intersection(neighbors[i]) for i in first):
            raise ValueError("Cross-role footprint leakage")
    patch_ids = rows["tortilla:id"].astype(str).to_numpy()
    return {
        "schema": 1, "dataset_revision": REVISION, "dataset_sha256": RAW_SHA256,
        "target_event": int(target), "source_train_events": sorted(train),
        "inner_validation_events": sorted(inner), "original_test_events_excluded": sorted(events["test"]),
        "sample_ids": {name: patch_ids[sorted(ids)].tolist() for name, ids in masks.items()},
        "train_draws_per_epoch": DRAWS,
    }


def verified_fold(frame, manifest):
    rows, events = development(frame)
    expected = make_fold(rows, events, footprint_neighbors(rows), manifest["target_event"])
    if manifest != expected:
        raise ValueError("Fold manifest disagrees with the deterministic leakage audit")
    return rows
