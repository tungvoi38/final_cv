#!/usr/bin/env python3
"""Calibrate flip TTA on validation or evaluate the locked recipes on test."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from evaluate_unet_eoa import (
    FLIP_TTA_BASELINE, METRICS, REPO, KuroDataset, RawArchive, digest, evaluation_code_hashes,
    load_calibrated_selection, load_json, load_model, pre_variability_inputs,
    seed_everything, write_json,
)
from torch.utils.data import DataLoader
from train_unet import FLOOD, IGNORE, N_CLASSES, move

FLIPS = ((), (-1,), (-2,), (-2, -1))
CASES = ("eoa", "eoa_flip_tta")
BIASES = np.round(np.arange(-8, 8.001, 0.05), 2)
ZERO = int(np.flatnonzero(BIASES == 0)[0])


def paired_logits(models, inputs):
    transformed = [torch.flip(inputs, axes) for axes in FLIPS[1:]]
    baseline, augmented = [], []
    for model in models:
        identity = model(inputs).float()
        logits = identity
        for view, axes in zip(transformed, FLIPS[1:], strict=True):
            logits = logits + torch.flip(model(view).float(), axes)
        baseline.append(identity)
        augmented.append(logits / len(FLIPS))
    return [(a + b + c) / 3 for a, b, c in (baseline, augmented)]


class FlipTTAEnsemble(torch.nn.Module):
    def __init__(self, members):
        super().__init__()
        if len(members) != 3:
            raise ValueError("Flip TTA requires three SMA members")
        self.members = torch.nn.ModuleList(members)

    def forward(self, inputs):
        _, flipped = paired_logits(self.members, inputs)
        return flipped


def bins_for(margin, target, nonflood):
    # The original threshold rule is margin > -bias, with ties won by nonflood.
    first = np.searchsorted(BIASES, -margin, side="right")
    width = len(BIASES) + 1
    groups = target.astype(np.int64) * FLOOD + nonflood
    return np.bincount(groups * width + first, minlength=N_CLASSES * FLOOD * width
                       ).reshape(N_CLASSES, FLOOD, width)


def confusion(bins):
    flood = bins.cumsum(axis=-1)[..., :-1]
    other = bins.sum(axis=-1, keepdims=True) - flood
    return np.concatenate((other, flood.sum(axis=1, keepdims=True)), axis=1).transpose(2, 0, 1)


def metrics(events):
    pooled = np.sum(list(events.values()), axis=0)
    tp = np.diagonal(pooled, axis1=-2, axis2=-1)
    actual, predicted = pooled.sum(axis=-1), pooled.sum(axis=-2)
    union = actual + predicted - tp
    cms = np.stack(list(events.values()))
    event_tp = cms[..., FLOOD, FLOOD]
    event_union = cms[..., FLOOD, :].sum(-1) + cms[..., :, FLOOD].sum(-1) - event_tp
    return {
        "pooled_miou": (tp / np.maximum(union, 1)).mean(-1),
        "pooled_flood_iou": tp[:, FLOOD] / np.maximum(union[:, FLOOD], 1),
        "pooled_flood_f1": 2 * tp[:, FLOOD] / np.maximum((actual + predicted)[:, FLOOD], 1),
        "event_macro_flood_iou": np.nanmean(
            event_tp / np.where(event_union > 0, event_union, np.nan), axis=0
        ),
        "pooled_flood_precision": tp[:, FLOOD] / np.maximum(predicted[:, FLOOD], 1),
        "pooled_flood_recall": tp[:, FLOOD] / np.maximum(actual[:, FLOOD], 1),
    }


def fixed_confusion(margin, target, nonflood, bias):
    predicted = np.where(margin.astype(np.float64) > -bias, FLOOD, nonflood)
    groups = target.astype(np.int64) * N_CLASSES + predicted
    return np.bincount(groups, minlength=N_CLASSES ** 2).reshape(1, N_CLASSES, N_CLASSES)


@torch.inference_mode()
def collect(models, loader, device, *, split="validation", locked_biases=None):
    counts = {case: {} for case in CASES}
    for step, batch in enumerate(loader, 1):
        x, y = move(batch, device)
        with torch.autocast("cuda"):
            outputs = paired_logits(models, pre_variability_inputs(x)[0])
        for case, logits in zip(CASES, outputs, strict=True):
            other, winner = logits[:, :FLOOD].max(dim=1)
            margins = logits[:, FLOOD] - other
            for i, event in enumerate(batch["event_id"]):
                event = str(event.item() if torch.is_tensor(event) else event)
                valid = y[i] != IGNORE
                margin = margins[i][valid].cpu().numpy()
                if not np.isfinite(margin).all():
                    raise ValueError("Nonfinite logit margin")
                target = y[i][valid].cpu().numpy()
                nonflood = winner[i][valid].cpu().numpy()
                bins = (bins_for(margin, target, nonflood) if locked_biases is None else
                        fixed_confusion(margin, target, nonflood, locked_biases[case]))
                if event in counts[case]:
                    counts[case][event] += bins
                else:
                    counts[case][event] = bins
        if step % 100 == 0 or step == len(loader):
            print(f"{split}: {step}/{len(loader)} batches", flush=True)
    if locked_biases is not None:
        return counts
    return {case: {event: confusion(bins) for event, bins in sorted(counts[case].items())}
            for case in CASES}


def save_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def best_bias(curve):
    return min(range(len(BIASES)),
               key=lambda i: (-curve["pooled_flood_iou"][i], abs(BIASES[i]), BIASES[i]))


def load_bias_lock(directory, selection, selection_path, recipe_path):
    protocol_path = directory / "protocol.json"
    lock_path = directory / "bias_selection.json"
    protocol, lock = load_json(protocol_path), load_json(lock_path)
    protocol_hash = digest(protocol_path)
    code = evaluation_code_hashes()
    old_code = protocol["evaluation_code_sha256"]
    baseline = load_json(FLIP_TTA_BASELINE)
    if (protocol_hash == baseline["validation_protocol_sha256"]
            and digest(lock_path) == baseline["validation_bias_lock_sha256"]):
        # The pinned archive predates the calibrated CLI using Flip TTA by default.
        # Its trainer and dataset digests must still match.
        code["evaluate_unet_eoa.py"] = old_code["evaluate_unet_eoa.py"]
    if (lock["split"] != "validation" or protocol["split"] != "validation"
            or lock["protocol_sha256"] != protocol_hash
            or protocol["selection_sha256"] != digest(selection_path)
            or protocol["calibration_sha256"] != digest(recipe_path)
            or protocol["checkpoint_members"] != selection["sma"]
            or protocol["flip_axes"] != [list(axes) for axes in FLIPS]
            or old_code != code
            or protocol["selection_metric"] != "validation pooled Flood IoU"
            or protocol["bias_grid"] != BIASES.tolist()
            or lock["baseline_confusions_match_exactly"] is not True):
        raise ValueError("Validation bias lock does not match the frozen selection and protocol")
    biases = lock["biases"]
    if set(biases) != set(CASES):
        raise ValueError("Validation bias lock is missing a comparison case")
    for case in CASES:
        expected, = (row["bias"] for row in lock["rows"]
                     if row["case"] == case and row["operating_point"] == "validation_optimal")
        if biases[case] not in BIASES or biases[case] != expected:
            raise ValueError(f"Invalid locked validation bias for {case}")
    return biases


def evaluate_test(out, validation_dir, models, loader, device, biases):
    print(f"Locked validation biases: {biases}", flush=True)
    events = collect(models, loader, device, split="test", locked_biases=biases)
    if any(len(events[case]) != 10 for case in CASES):
        raise ValueError("Expected 10 test events")
    rows = []
    for case in CASES:
        curve = metrics(events[case])
        rows.append({
            "case": case, "bias": biases[case],
            **{key: float(curve[key][0]) for key in METRICS},
        })
        print(" ".join(f"{key}={value}" for key, value in rows[-1].items()), flush=True)
    save_csv(out / "test_comparison.csv", rows)
    write_json(out / "test_metrics.json", {
        "split": "test", "validation_bias_lock_sha256": digest(validation_dir / "bias_selection.json"),
        "script_sha256": digest(Path(__file__)), "rows": rows,
    })


def plot(out, curves, best):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), layout="constrained")
    for ax, key, title in zip(axes, ("pooled_flood_iou", "pooled_miou"),
                              ("Flood IoU", "Pooled mIoU"), strict=True):
        for case, label, color in zip(CASES, ("EoA", "EoA + Flip TTA"),
                                      ("#254c87", "#b44e32"), strict=True):
            values = 100 * curves[case][key]
            i = best[case]
            ax.plot(BIASES, values, color=color, label=f"{label} (bias {BIASES[i]:+.2f})")
            ax.scatter(BIASES[i], values[i], color=color, zorder=3)
        ax.axvline(0, color="0.55", ls=":", lw=1)
        ax.set(xlim=(-2, 4), xlabel="Flood-logit bias", ylabel=f"Validation {title} (%)")
        ax.grid(alpha=0.18)
        ax.legend()
    fig.savefig(out / "validation_bias.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--baseline-confusions", type=Path)
    parser.add_argument("--validation-dir", type=Path, help="Directory with the validation bias lock")
    args = parser.parse_args()
    if args.split == "test":
        if args.validation_dir is None or args.baseline_confusions is not None:
            parser.error("test requires --validation-dir")
    elif args.baseline_confusions is None or args.validation_dir is not None:
        parser.error("validation requires --baseline-confusions")
    recipe_path = REPO / "configs/experiments/unet_a3_eoa_calibrated.json"
    recipe = load_json(recipe_path)
    selection_path = Path(recipe["selection"])
    selection = load_calibrated_selection(recipe, selection_path)
    biases = (load_bias_lock(args.validation_dir, selection, selection_path, recipe_path)
              if args.split == "test" else None)
    seed_everything(42)
    device = torch.device("cuda")
    settings = selection["protocol"]["training"]
    archive = RawArchive(settings["raw"])
    loader = DataLoader(KuroDataset(archive, args.split), batch_size=settings["batch_size"],
                        num_workers=settings["workers"], shuffle=False, pin_memory=True)
    models = [load_model(member, device) for member in selection["sma"]]
    args.out.mkdir(parents=True, exist_ok=False)
    if args.split == "test":
        evaluate_test(args.out, args.validation_dir, models, loader, device, biases)
        archive.check_unchanged()
        print(f"Saved: {args.out}", flush=True)
        return
    write_json(args.out / "protocol.json", {
        "split": "validation", "selection_sha256": digest(selection_path),
        "checkpoint_members": selection["sma"], "calibration_sha256": digest(recipe_path),
        "script_sha256": digest(Path(__file__)), "evaluation_code_sha256": evaluation_code_hashes(),
        "baseline_confusions": str(args.baseline_confusions.resolve()),
        "baseline_confusions_sha256": digest(args.baseline_confusions),
        "bias_grid": BIASES.tolist(), "selection_metric": "validation pooled Flood IoU",
        "tie_break": "nearest zero, then lower bias", "flip_axes": FLIPS,
        "aggregation": "undo each flip, mean logits per seed in FP32, then mean three seeds",
    })
    events = collect(models, loader, device)
    archive.check_unchanged()
    with np.load(args.baseline_confusions, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["biases"], BIASES)
        if set(saved.files) - {"biases"} != events["eoa"].keys() or len(events["eoa"]) != 6:
            raise ValueError("Unexpected validation events")
        for event, cm in events["eoa"].items():
            np.testing.assert_array_equal(cm, saved[event], err_msg=f"EoA baseline event {event}")
    curves = {case: metrics(events[case]) for case in CASES}
    best = {case: best_bias(curves[case]) for case in CASES}
    if BIASES[best["eoa"]] != recipe["flood_logit_bias"]:
        raise ValueError("Archived baseline bias did not reproduce")
    for key in METRICS:
        np.testing.assert_allclose(curves["eoa"][key][best["eoa"]],
                                   recipe["validation_metrics"][key], rtol=0, atol=1e-12)
    rows = []
    for case in CASES:
        curve = curves[case]
        for operating_point, i in (("zero", ZERO), ("validation_optimal", best[case])):
            rows.append({"case": case, "operating_point": operating_point, "bias": BIASES[i],
                         **{key: float(curve[key][i]) for key in METRICS}})
        save_csv(args.out / f"validation_bias_{case}.csv", [
            {"bias": bias, **{key: curve[key][i] for key in METRICS}}
            for i, bias in enumerate(BIASES)
        ])
        np.savez_compressed(args.out / f"validation_confusion_{case}.npz",
                            biases=BIASES, **events[case])
    save_csv(args.out / "validation_comparison.csv", rows)
    write_json(args.out / "bias_selection.json", {
        "split": "validation", "protocol_sha256": digest(args.out / "protocol.json"),
        "baseline_confusions_match_exactly": True,
        "biases": {case: float(BIASES[i]) for case, i in best.items()}, "rows": rows,
    })
    plot(args.out, curves, best)
    print(f"Saved: {args.out}", flush=True)


if __name__ == "__main__":
    main()
