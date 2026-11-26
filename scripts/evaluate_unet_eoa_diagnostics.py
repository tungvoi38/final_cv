#!/usr/bin/env python3
"""Run flood PR and bias diagnostics for the frozen 11-channel Flip TTA baseline."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from evaluate_unet_eoa import (
    FLIP_TTA_BASELINE, METRICS, NO_TTA_BASELINE, KuroDataset, RawArchive, digest,
    evaluation_code_hashes, load_calibrated_selection, load_json, load_model,
    pre_variability_inputs, seed_everything, write_json,
)
from evaluate_unet_eoa_flip_tta import (
    BIASES, FLIPS, ZERO, FlipTTAEnsemble, best_bias, bins_for, confusion, load_bias_lock,
    metrics, save_csv,
)
from torch.utils.data import DataLoader
from train_unet import FLOOD, IGNORE, move

FLOOD_IOU, FLOOD_F1 = "pooled_flood_iou", "pooled_flood_f1"


def metric_row(curve, index):
    return {"bias": float(BIASES[index]), **{key: float(curve[key][index]) for key in METRICS}}


def optima(curve):
    return {key: best_bias({FLOOD_IOU: curve[key]}) for key in (FLOOD_IOU, FLOOD_F1)}


def validated_bias(out, directory, recipe, selection, selection_path):
    biases = load_bias_lock(directory, selection, selection_path, NO_TTA_BASELINE)
    counts_file = directory / "validation_confusion_eoa_flip_tta.npz"
    with np.load(counts_file, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["biases"], BIASES)
        events = {event: saved[event] for event in saved.files if event != "biases"}
    if len(events) != 6:
        raise ValueError("Expected six validation events in the saved sweep")
    curve = metrics(events)
    best = optima(curve)
    locked = best[FLOOD_IOU]
    if BIASES[locked] != biases["eoa_flip_tta"] or BIASES[locked] != recipe["flood_logit_bias"]:
        raise ValueError("Archived sweep does not reproduce the fixed baseline bias")
    for key in METRICS:
        np.testing.assert_allclose(curve[key][locked], recipe["validation_metrics"][key],
                                   rtol=0, atol=1e-12, err_msg=key)
    save_csv(out / "validation_bias_sweep.csv", [metric_row(curve, i) for i in range(len(BIASES))])
    write_json(out / "validation_bias_selection.json", {
        "split": "validation", "selection": "pooled Flood IoU",
        "archive_validation_lock_sha256": digest(directory / "bias_selection.json"),
        "validation_confusion_sha256": digest(counts_file),
        "at_zero": metric_row(curve, ZERO),
        "selected_global_bias": metric_row(curve, locked),
        "validation_optima": {key: metric_row(curve, i) for key, i in best.items()},
    })
    print(f"Validation-selected global bias verified and locked: {BIASES[locked]:+.2f}", flush=True)
    return locked


@torch.inference_mode()
def collect_test(model, loader, device):
    bins, chunks = {}, defaultdict(list)
    model.eval()
    for step, batch in enumerate(loader, 1):
        x, target = move(batch, device)
        with torch.autocast("cuda"):
            logits = model(pre_variability_inputs(x)[0])
        other, winner = logits[:, :FLOOD].max(dim=1)
        margin = logits[:, FLOOD] - other
        for i, raw_event in enumerate(batch["event_id"]):
            event = str(raw_event.item() if torch.is_tensor(raw_event) else raw_event)
            valid = target[i] != IGNORE
            score = margin[i][valid].cpu().numpy()
            if not score.size:
                continue
            if not np.isfinite(score).all():
                raise ValueError(f"Nonfinite flood margin in event {event}")
            label = target[i][valid].cpu().numpy()
            values = bins_for(score, label, winner[i][valid].cpu().numpy())
            if event not in bins:
                bins[event] = values
            else:
                bins[event] += values
            chunks[event].append((score, label == FLOOD))
        if step % 100 == 0 or step == len(loader):
            print(f"test: {step}/{len(loader)} batches", flush=True)
    if len(bins) != 10:
        raise ValueError("Expected 10 test events with valid pixels")
    return {event: confusion(bins[event]) for event in sorted(bins, key=int)}, chunks


def rank_event(chunks):
    scores = np.concatenate([chunk[0] for chunk in chunks])
    positive = np.concatenate([chunk[1] for chunk in chunks])
    order = np.argsort(scores)
    scores = scores[order]
    starts = np.r_[0, np.flatnonzero(scores[1:] != scores[:-1]) + 1]
    return (scores[starts], np.add.reduceat(positive[order], starts, dtype=np.int64),
            np.diff(np.r_[starts, scores.size]))


def rank_pooled(groups):
    thresholds, index = np.unique(np.concatenate([group[0] for group in groups]), return_inverse=True)
    # Counts are summed after combining exact ties across events.
    positives = np.bincount(index, weights=np.concatenate([group[1] for group in groups])).astype(np.int64)
    totals = np.bincount(index, weights=np.concatenate([group[2] for group in groups])).astype(np.int64)
    return thresholds, positives, totals


def pr_curve(group):
    thresholds, positives, totals = (values[::-1] for values in group)
    n_positive = positives.sum()
    if n_positive == 0:
        raise ValueError("Flood PR needs positive flood labels")
    tp, predictions = positives.cumsum(), totals.cumsum()
    precision, recall = tp / predictions, tp / n_positive
    ap = float(np.dot(positives / n_positive, precision))
    return thresholds, precision, recall, ap


def save_bias_diagnostics(out, events, locked):
    pooled = metrics(events)
    paper = [{"operating_point": point, **metric_row(pooled, index)}
             for point, index in (("b_zero", ZERO), ("global_bias_selected_on_validation", locked))]
    save_csv(out / "test_paper_metrics.csv", paper)
    rows, sweeps = [], []
    for event, cm in events.items():
        curve = metrics({event: cm})
        best = optima(curve)
        iou_oracle, f1_oracle = best[FLOOD_IOU], best[FLOOD_F1]
        values = {
            "event_id": event, "valid_pixels": int(cm[ZERO].sum()),
            "flood_pixels": int(cm[ZERO, FLOOD].sum()), "validation_global_bias": BIASES[locked],
            "flood_iou_zero": curve[FLOOD_IOU][ZERO], "flood_iou_validation_bias": curve[FLOOD_IOU][locked],
            "flood_f1_zero": curve[FLOOD_F1][ZERO], "flood_f1_validation_bias": curve[FLOOD_F1][locked],
            "oracle_bias_iou_DIAGNOSTIC_ONLY": BIASES[iou_oracle],
            "oracle_bias_f1_DIAGNOSTIC_ONLY": BIASES[f1_oracle],
            "oracle_flood_iou_DIAGNOSTIC_ONLY": curve[FLOOD_IOU][iou_oracle],
            "oracle_flood_f1_DIAGNOSTIC_ONLY": curve[FLOOD_F1][f1_oracle],
            "oracle_iou_on_grid_edge": iou_oracle in (0, len(BIASES) - 1),
            "oracle_f1_on_grid_edge": f1_oracle in (0, len(BIASES) - 1),
        }
        rows.append(values)
        sweeps.extend({"event_id": event, **metric_row(curve, i)} for i in range(len(BIASES)))
    save_csv(out / "test_event_oracle_DIAGNOSTIC_ONLY.csv", rows)
    save_csv(out / "test_event_bias_sweep_DIAGNOSTIC_ONLY.csv", sweeps)
    zero_sd, global_sd, oracle_sd = [
        float(np.std([row[key] for row in rows], ddof=0))
        for key in ("flood_iou_zero", "flood_iou_validation_bias", "oracle_flood_iou_DIAGNOSTIC_ONLY")
    ]
    oracle_bias = {}
    for key in ("iou", "f1"):
        values = np.array([row[f"oracle_bias_{key}_DIAGNOSTIC_ONLY"] for row in rows])
        oracle_bias[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=0)),
                            "min": float(values.min()), "max": float(values.max()),
                            "range": float(np.ptp(values))}
    report = {
        "paper": {"test_metrics": paper, "event_iou_std_zero": zero_sd,
                  "event_iou_std_with_validation_global_bias": global_sd,
                  "event_iou_std_reduction_pp": 100 * (zero_sd - global_sd)},
        "oracle_DIAGNOSTIC_ONLY": {
            "definition": "Per-event test-label upper bound over the declared bias grid",
            "n_events": len(rows), "oracle_bias_distribution": oracle_bias,
            "event_iou_std_oracle": oracle_sd,
            "event_iou_std_reduction_vs_zero_pp": 100 * (zero_sd - oracle_sd),
            "event_iou_std_reduction_vs_validation_bias_pp": 100 * (global_sd - oracle_sd),
        },
        "std_ddof": 0, "caveat": "Oracle is a diagnostic upper bound only; never use it for claims or selection.",
    }
    write_json(out / "bias_diagnostics_summary.json", report)
    print(json.dumps(report, indent=2), flush=True)


def save_pr(out, chunks, events, locked):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("Computing exact flood PR curves from cached test scores", flush=True)
    groups = {event: rank_event(chunks.pop(event)) for event in events}
    pooled = rank_pooled(list(groups.values()))
    points = [("pooled", pooled, metrics(events))]
    points.extend((event, group, metrics({event: events[event]})) for event, group in groups.items())
    pr_dir = out / "pr_exact"
    pr_dir.mkdir()
    fig, ax = plt.subplots(figsize=(5.8, 4.5), layout="constrained")
    event_fig, event_axes = plt.subplots(2, 5, figsize=(14, 5.8), layout="constrained",
                                         sharex=True, sharey=True)
    rows = []
    for (scope, group, curve), panel in zip(points, (ax, *event_axes.flat), strict=True):
        threshold, precision, recall, ap = pr_curve(group)
        np.savez_compressed(pr_dir / f"{scope}.npz", threshold=threshold,
                            precision=precision, recall=recall)
        rows.append({"scope": scope, "average_precision": ap, "n_thresholds": int(threshold.size)})
        draw = np.unique(np.linspace(0, len(recall) - 1, min(20000, len(recall)), dtype=int))
        panel.plot(np.r_[0, recall[draw]], np.r_[1, precision[draw]], color="#315791",
                   label=f"AP {ap:.3f}")
        for i, style, label in ((ZERO, "o", "b=0"), (locked, "^", f"val b={BIASES[locked]:+.2f}")):
            panel.scatter(curve["pooled_flood_recall"][i], curve["pooled_flood_precision"][i],
                          marker=style, label=label)
        panel.set(xlim=(0, 1.02), ylim=(0, 1.02), title="Pooled test" if scope == "pooled" else f"Event {scope}")
        panel.grid(alpha=0.2)
        panel.legend(fontsize=8 if scope == "pooled" else 6, loc="lower left")
    ax.set(xlabel="Flood recall", ylabel="Flood precision")
    event_fig.supxlabel("Flood recall")
    event_fig.supylabel("Flood precision")
    fig.savefig(out / "test_flood_pr_pooled.png", dpi=180)
    event_fig.savefig(out / "test_flood_pr_by_event.png", dpi=180)
    plt.close("all")
    save_csv(out / "test_flood_pr_summary.csv", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    recipe = load_json(FLIP_TTA_BASELINE)
    selection_path = Path(recipe["selection"])
    selection = load_calibrated_selection(recipe, selection_path)
    if recipe["flip_axes"] != [list(axes) for axes in FLIPS]:
        raise ValueError("The saved baseline does not use the expected flip views")
    args.out.mkdir(parents=True, exist_ok=False)
    locked = validated_bias(args.out, args.validation_dir, recipe, selection, selection_path)
    write_json(args.out / "protocol.json", {
        "checkpoint_members": selection["sma"], "selection_sha256": digest(selection_path),
        "recipe_sha256": digest(FLIP_TTA_BASELINE), "flip_axes": FLIPS,
        "bias_grid": BIASES.tolist(), "bias_ties": "nearest zero, then lower bias",
        "bias_lock_sha256": digest(args.out / "validation_bias_selection.json"),
        "flood_score": "z_flood - max(z_background, z_permanent_water)",
        "prediction": "margin > -bias; ties resolved to nonflood",
        "pr": "exact all-pixel ranking using margin; ties grouped, ignore=255 excluded",
        "script_sha256": digest(Path(__file__)), "evaluation_code_sha256": evaluation_code_hashes(),
        "flip_evaluator_sha256": digest(Path(__file__).with_name("evaluate_unet_eoa_flip_tta.py")),
    })
    seed_everything(42)
    device = torch.device("cuda")
    settings = selection["protocol"]["training"]
    archive = RawArchive(settings["raw"])
    loader = DataLoader(KuroDataset(archive, "test"), batch_size=settings["batch_size"],
                        num_workers=settings["workers"], shuffle=False, pin_memory=True)
    model = FlipTTAEnsemble([load_model(member, device) for member in selection["sma"]])
    events, chunks = collect_test(model, loader, device)
    archive.check_unchanged()
    save_bias_diagnostics(args.out, events, locked)
    save_pr(args.out, chunks, events, locked)
    print(f"Saved: {args.out}", flush=True)


if __name__ == "__main__":
    main()
