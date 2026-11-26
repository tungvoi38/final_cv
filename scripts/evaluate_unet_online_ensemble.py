#!/usr/bin/env python3
"""Evaluate the frozen online members of the 11-channel A3 comparison."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from evaluate_unet_eoa import (
    FLIP_TTA_BASELINE, METRICS, KuroDataset, RawArchive, digest, evaluation_code_hashes,
    load_calibrated_selection, load_json, load_model, seed_everything, write_json,
)
from evaluate_unet_eoa_flip_tta import BIASES, FLIPS, ZERO, best_bias, collect, metrics, save_csv
from torch.utils.data import DataLoader


def freeze(selection_path):
    recipe = load_json(FLIP_TTA_BASELINE)
    selection = load_calibrated_selection(recipe, selection_path)
    members = selection["online"]
    if [member["seed"] for member in members] != [42, 43, 44]:
        raise ValueError("Expected the three validation-selected online seeds")
    for member in members:
        if (member["input_channels"], member["weights_key"], member["head"]) != (11, "online_model", "a3"):
            raise ValueError("Expected online weights for an 11-channel A3 member")
        if digest(member["checkpoint"]) != member["sha256"]:
            raise ValueError(f"Frozen online checkpoint changed: {member['checkpoint']}")
    return selection, {
        "selection_sha256": digest(selection_path), "members": members,
        "flip_axes": [list(axes) for axes in FLIPS], "bias_grid": BIASES.tolist(),
        "bias_selection": "validation pooled Flood IoU; ties nearest zero, then lower bias",
        "code_sha256": {
            **evaluation_code_hashes(),
            **{name: digest(Path(__file__).with_name(name))
               for name in (Path(__file__).name, "evaluate_unet_eoa_flip_tta.py")},
        },
    }


def locked_bias(directory, protocol):
    protocol_path = directory / "protocol.json"
    lock = load_json(directory / "bias_selection.json")
    if (load_json(protocol_path) != protocol or lock["split"] != "validation"
            or lock["protocol_sha256"] != digest(protocol_path) or lock["bias"] not in BIASES):
        raise ValueError("Validation bias lock differs from the frozen online ensemble")
    return lock["bias"]


def metric_row(case, bias, curve, index):
    return {"case": case, "bias": float(bias), **{key: float(curve[key][index]) for key in METRICS}}


def verify_original(path, protocol, split, row):
    saved = load_json(path)
    if saved["split"] != split or saved["selection_sha256"] != protocol["selection_sha256"]:
        raise ValueError(f"Unexpected reference metrics: {path}")
    original, = (entry for entry in saved["rows"] if entry["case"] == "erm_online_ensemble")
    if original["checkpoints"] != " | ".join(member["checkpoint"] for member in protocol["members"]):
        raise ValueError(f"Reference metrics use different online checkpoints: {path}")
    for key in METRICS:
        np.testing.assert_allclose(row[key], original[key], rtol=0, atol=1e-12, err_msg=key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--validation-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (args.split == "test") != (args.validation_dir is not None):
        parser.error("test requires --validation-dir with a completed validation sweep")
    if args.out.exists():
        parser.error("--out must be a new directory")
    selection, protocol = freeze(args.selection)
    bias = locked_bias(args.validation_dir, protocol) if args.split == "test" else None
    seed_everything(42)
    device = torch.device("cuda")
    settings = selection["protocol"]["training"]
    archive = RawArchive(settings["raw"])
    loader = DataLoader(KuroDataset(archive, args.split), batch_size=settings["batch_size"],
                        num_workers=settings["workers"], shuffle=False, pin_memory=True)
    models = [load_model(member, device) for member in protocol["members"]]
    biases = None if bias is None else {"eoa": 0.0, "eoa_flip_tta": bias}
    if biases is not None:
        print(f"Locked online Flip TTA bias: {bias:+.2f}", flush=True)
    events = collect(models, loader, device, split=args.split, locked_biases=biases)
    archive.check_unchanged()
    if any(len(group) != (6 if bias is None else 10) for group in events.values()):
        raise ValueError(f"Unexpected {args.split} event count")
    original, flipped = (metrics(events[case]) for case in ("eoa", "eoa_flip_tta"))
    rows = [metric_row("online_ensemble_no_tta", 0, original, ZERO if bias is None else 0)]
    reference = ("validation_metrics.json" if bias is None else "test/test_metrics.json")
    verify_original(args.selection.parent / reference, protocol, args.split, rows[0])
    args.out.mkdir(parents=True, exist_ok=False)
    if bias is None:
        best = best_bias(flipped)
        bias = float(BIASES[best])
        rows.append(metric_row("online_ensemble_flip_tta_b_zero", 0, flipped, ZERO))
        rows.append(metric_row("online_ensemble_flip_tta", bias, flipped, best))
        save_csv(args.out / "validation_bias_sweep.csv", [
            metric_row("online_ensemble_flip_tta", b, flipped, i) for i, b in enumerate(BIASES)
        ])
        write_json(args.out / "protocol.json", protocol)
        write_json(args.out / "bias_selection.json", {
            "split": "validation", "bias": bias, "protocol_sha256": digest(args.out / "protocol.json"),
        })
    else:
        rows.append(metric_row("online_ensemble_flip_tta", bias, flipped, 0))
        write_json(args.out / "test_metrics.json", {
            "split": "test", "rows": rows,
            "bias_lock_sha256": digest(args.validation_dir / "bias_selection.json"),
        })
    save_csv(args.out / f"{args.split}_comparison.csv", rows)
    print(json.dumps(rows, indent=2))
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
