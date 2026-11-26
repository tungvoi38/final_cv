#!/usr/bin/env python3
"""Event-level percentile bootstrap using saved A3 artifacts only."""
import argparse
import csv
from dataclasses import dataclass
from itertools import combinations
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ITERATIONS = 10_000
SEED = 42
METRICS = {
    "pooled_miou": "Pooled mIoU",
    "pooled_flood_f1": "Pooled Flood F1",
    "event_macro_flood_iou": "Event-macro Flood IoU",
    "event_flood_iou_std": "Event Flood IoU std",
}
ERM, SMA, ONLINE, EOA, FLIP, LOCKED = ("Vanilla ERM", "Single SMA", "Online Ensemble", "EoA",
                                       "EoA + Flip TTA", "EoA + Flip TTA + Global Bias 0.95")
EOA_BIAS = "EoA + Global Bias 0.95"
SMA42 = "Single SMA seed 42"
REPO = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def one(rows, **fields):
    row, = (r for r in rows if all(r[key] == value for key, value in fields.items()))
    return row


def verify(actual, expected, label, atol=1e-9):
    if not np.allclose(actual, expected, rtol=0, atol=atol):
        raise ValueError(f"{label}: {actual} differs from {expected}")


@dataclass
class Case:
    name: str
    point: dict
    source: str
    flood: np.ndarray | None = None  # member, event, (TP, FP, FN)
    multiclass: np.ndarray | None = None  # member, event, true class, predicted class


def flood_counts(cm):
    tp = cm[..., 2, 2]
    return np.stack((tp, cm[..., :, 2].sum(-1) - tp, cm[..., 2, :].sum(-1) - tp), axis=-1)


def load_cm(path, ids, bias):
    with np.load(path) as archive:
        idx, = np.flatnonzero(np.isclose(archive["biases"], bias, atol=1e-12, rtol=0))
        if set(archive.files) != {"biases", *ids}:
            raise ValueError(f"Unexpected event set in {path}")
        return np.array([[archive[key][idx] for key in ids]], dtype=np.int64)


def recover_flood_counts(path, bias, true_flood, valid):
    """Exact archived PR points store cumulative recall and precision at each threshold."""
    with np.load(path) as archive:
        threshold = archive["threshold"]
        if len(threshold) == 0 or not np.all(threshold[:-1] > threshold[1:]):
            raise ValueError(f"Unsorted PR thresholds: {path}")
        predicted_groups = np.count_nonzero(threshold.astype(np.float64) > -bias)
        if predicted_groups == 0:
            return 0, 0, true_flood
        i = predicted_groups - 1
        recall, precision = float(archive["recall"][i]), float(archive["precision"][i])
    tp = round(recall * true_flood)
    if tp <= 0 or precision <= 0:
        raise ValueError(f"PR point does not identify FP when TP=0: {path}")
    predicted = round(tp / precision)
    if not 0 < tp <= min(predicted, true_flood) or predicted - tp > valid - true_flood:
        raise ValueError(f"Invalid reconstructed counts: {path}")
    verify([tp / true_flood, tp / predicted], [recall, precision], str(path), atol=1e-12)
    return tp, predicted - tp, true_flood - tp


def get_point(rows, method, split):
    row = one(rows, method=method, split=split)
    return {key: float(row[key]) for key in METRICS if row.get(key)}


def load_cases(root, run, ablation):
    history = read_csv(ablation)
    selection_path = run / "comparison/selection.json"
    selection = read_json(selection_path)
    selection_hash = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    for method in (ERM, SMA, ONLINE, EOA, FLIP, LOCKED):
        group = "online" if method in (ERM, ONLINE) else "sma"
        paths = " | ".join(member["checkpoint"] for member in selection[group])
        for split in ("validation", "test_historical"):
            if one(history, method=method, split=split)["checkpoints"] != paths:
                raise ValueError(f"Ablation uses different checkpoints: {split} {method}")
    flip = root / "a3-eoa-flip-tta-q8H1dB/results"
    diag = root / "a3-eoa-flood-diagnostics-YtaAJg/results"
    for directory in (flip, diag):
        protocol = read_json(directory / "protocol.json")
        if (protocol["checkpoint_members"] != selection["sma"]
                or protocol["selection_sha256"] != selection_hash):
            raise ValueError(f"Checkpoint provenance mismatch: {directory}")
    val_cm_path = flip / "validation_confusion_eoa.npz"
    flip_cm_path = flip / "validation_confusion_eoa_flip_tta.npz"
    with np.load(val_cm_path) as archive:
        val_ids = sorted(set(archive.files) - {"biases"}, key=int)
    val_base = load_cm(val_cm_path, val_ids, 0)
    val_true_flood = val_base[0, :, 2, :].sum(-1)
    val_total = val_base[0].sum((-1, -2))
    val = []
    for group, name in (("online", ERM), ("sma", SMA)):
        counts, sources = [], []
        for member in selection[group]:
            filename = "online_validation_events.csv" if group == "online" else "validation_events.csv"
            path = Path(member["checkpoint"]).parent / filename
            by_event = {r["event_id"]: r for r in read_csv(path) if int(r["epoch"]) == member["epoch"]}
            if set(by_event) != set(val_ids):
                raise ValueError(f"Validation event mismatch in {path}")
            triple = np.array([[int(by_event[event][k]) for k in ("flood_tp", "flood_fp", "flood_fn")]
                               for event in val_ids])
            verify(triple[:, 0] + triple[:, 2], val_true_flood, str(path))
            counts.append(triple)
            sources.append(str(path))
        val.append(Case(name, get_point(history, name, "validation"), " | ".join(sources),
                        flood=np.array(counts)))
    val.append(Case(ONLINE, get_point(history, ONLINE, "validation"), str(ablation)))
    compare = flip / "validation_comparison.csv"
    for name, path, bias in ((EOA, val_cm_path, 0), (EOA_BIAS, val_cm_path, .95),
                             (FLIP, flip_cm_path, 0), (LOCKED, flip_cm_path, .95)):
        cm = load_cm(path, val_ids, bias)
        verify(cm[0, :, 2, :].sum(-1), val_true_flood, str(path))
        verify(cm[0].sum((-1, -2)), val_total, str(path))
        if name == EOA_BIAS:
            row = one(read_csv(compare), case="eoa", operating_point="validation_optimal")
            point = {key: float(row[key]) for key in METRICS if key in row}
        else:
            point = get_point(history, name, "validation")
        val.append(Case(name, point, str(path), flood_counts(cm), cm))

    event_csv = diag / "test_event_oracle_DIAGNOSTIC_ONLY.csv"
    event_rows = {r["event_id"]: r for r in read_csv(event_csv)}
    test_ids = sorted(event_rows, key=int)
    true_flood = [int(event_rows[e]["flood_pixels"]) for e in test_ids]
    total = [int(event_rows[e]["valid_pixels"]) for e in test_ids]
    if len(val_ids) != 6 or len(test_ids) != 10 or set(val_ids) & set(test_ids):
        raise ValueError("Unexpected event split")
    test = [Case(name, get_point(history, name, "test_historical"), str(ablation))
            for name in (ERM, SMA, ONLINE, EOA)]
    calibrated_json = root / "a3-eoa-baseline-test-mm42m8/results/metrics.json"
    calibrated = read_json(calibrated_json)
    if (calibrated["split"], calibrated["flip_tta"], calibrated["flood_logit_bias"],
            calibrated["selection_sha256"]) != ("test", True, .95, selection_hash):
        raise ValueError("Calibrated test artifact has unexpected protocol")
    for name, bias, iou_key in ((FLIP, 0., "flood_iou_zero"), (LOCKED, .95, "flood_iou_validation_bias")):
        triples = []
        for event, n, n_valid in zip(test_ids, true_flood, total, strict=True):
            if float(event_rows[event]["validation_global_bias"]) != .95:
                raise ValueError("Unexpected validation-locked bias in test source")
            path = diag / f"pr_exact/{event}.npz"
            tp, fp, fn = recover_flood_counts(path, bias, n, n_valid)
            verify(tp/(tp+fp+fn), float(event_rows[event][iou_key]), str(path))
            if bias:
                raw = calibrated["events"][event]
                verify([tp, fp, fn], [raw[f"flood_{key}"] for key in ("tp", "fp", "fn")], str(path))
            triples.append((tp, fp, fn))
        test.append(Case(name, get_point(history, name, "test_historical"),
                         f"{diag / 'pr_exact'} | {event_csv} | {calibrated_json}",
                         flood=np.array([triples])))
    scaling = root / "scaling_sma_diagnostics/seed42-mfXZJj/results"
    if read_json(scaling / "protocol.json")["members"]["4000"] != selection["sma"][0]:
        raise ValueError("Archived 4k comparison is not the same SMA seed 42")
    path = scaling / "test_events_4000.csv"
    by_event = {r["event_id"]: r for r in read_csv(path)}
    if set(by_event) != set(test_ids):
        raise ValueError("Archived SMA test coverage differs")
    verify([int(by_event[e]["zero_valid_pixels"]) for e in test_ids], total, str(path))
    verify([int(by_event[e]["zero_flood_pixels"]) for e in test_ids], true_flood, str(path))
    c = np.array([[[int(by_event[e][f"zero_{key}"]) for key in ("tp", "fp", "fn")] for e in test_ids]])
    sma42 = one(read_csv(ablation.with_name("seed_details.csv")), split="test_historical",
                method=SMA, seed="42")
    test.insert(4, Case(SMA42, {k: float(sma42[k]) for k in METRICS if sma42.get(k)}, str(path), flood=c))
    return {"validation": (val_ids, val), "test_historical": (test_ids, test)}


def event_statistics(case, draws):
    """Each row of draws contains multiplicities of the same ordered event list."""
    output = {}
    if case.multiclass is not None:
        pooled = np.einsum("re,meij->mrij", draws, case.multiclass)
        diagonal = np.diagonal(pooled, axis1=-2, axis2=-1)
        union = pooled.sum(-1) + pooled.sum(-2) - diagonal
        output["pooled_miou"] = (diagonal / np.maximum(union, 1)).mean(axis=(0, 2))
    if case.flood is not None:
        tp, fp, fn = np.moveaxis(np.einsum("re,mek->mrk", draws, case.flood), -1, 0)
        output["pooled_flood_f1"] = (2 * tp / np.maximum(2 * tp + fp + fn, 1)).mean(0)
        union = case.flood.sum(-1)
        attended = draws @ (union > 0).T
        iou = case.flood[..., 0] / np.maximum(union, 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = (draws @ iou.T) / attended
            squared_mean = (draws @ (iou**2).T) / attended
        output["event_macro_flood_iou"] = mean.mean(1)
        output["event_flood_iou_std"] = np.sqrt(np.maximum(squared_mean - mean**2, 0)).mean(1)
    return output


def interval(values):
    finite = values[np.isfinite(values)] if values is not None else np.array([])
    lo, hi = np.quantile(finite, (.025, .975)) if len(finite) else (None, None)
    return lo, hi, len(finite)


def bootstrap(split, ids, cases, rng):
    n = len(ids)
    picks = rng.integers(0, n, size=(ITERATIONS, n))
    draws = (picks[..., None] == np.arange(n)).sum(axis=1)
    summary, paired, points, replicates = [], [], {}, {}
    for case in cases:
        exact = {k: float(v[0]) for k, v in event_statistics(case, np.ones((1, n), dtype=int)).items()}
        for key, value in exact.items():
            if key in case.point:
                verify(value, case.point[key], f"{split} {case.name} {key}")
        points[case.name] = case.point | exact
        replicates[case.name] = event_statistics(case, draws)
        for metric in METRICS:
            lo, hi, count = interval(replicates[case.name].get(metric))
            reason = ("missing per-event multiclass confusion" if metric == "pooled_miou"
                      else "missing matched per-event binary confusion for all members")
            summary.append(dict(split=split, method=case.name, metric=metric, events=n,
                                event_ids=" | ".join(ids), point=points[case.name].get(metric),
                                ci_low=lo, ci_high=hi, n_valid_iterations=count,
                                status="available" if count else reason, source=case.source))
    for base, variant in combinations(cases, 2):
        for metric in METRICS:
            a = replicates[base.name].get(metric)
            b = replicates[variant.name].get(metric)
            if a is None or b is None:
                continue
            lo, hi, count = interval(b-a)
            paired.append(dict(split=split, reference=base.name, method=variant.name, metric=metric,
                               events=n, event_ids=" | ".join(ids),
                               delta=points[variant.name][metric]-points[base.name][metric],
                               ci_low=lo, ci_high=hi, n_valid_iterations=count))
    return summary, paired


def export_event_counts(splits):
    classes = ("background", "water", "flood")
    for split, (ids, cases) in splits.items():
        for case in cases:
            if case.flood is None:
                continue
            members = (42, 43, 44) if len(case.flood) == 3 else (42 if case.name == SMA42 else "ensemble",)
            for index, member in enumerate(members):
                for j, event in enumerate(ids):
                    tp, fp, fn = case.flood[index, j]
                    row = dict(split=split, method=case.name, member=member, event_id=event,
                               flood_tp=tp, flood_fp=fp, flood_fn=fn)
                    for actual, name in enumerate(classes):
                        for predicted, prediction in enumerate(classes):
                            row[f"true_{name}_pred_{prediction}"] = (
                                case.multiclass[index, j, actual, predicted] if case.multiclass is not None else None
                            )
                    yield row


def draw_intervals(path, rows, *, paired=False):
    fig, axs = plt.subplots(2, 2, figsize=(14, 8), layout="constrained")
    for ax, (metric, label) in zip(axs.flat, METRICS.items(), strict=True):
        selected = [r for r in rows if r["metric"] == metric and r["ci_low"] is not None]
        for i, row in enumerate(selected):
            key = "delta" if paired else "point"
            ax.hlines(i, 100*row["ci_low"], 100*row["ci_high"], color="#285285")
            ax.plot(100*row[key], i, "o", color="#285285")
        names = [f'{r["method"]} − {r["reference"]}' if paired else r["method"] for r in selected]
        ax.set(yticks=range(len(names)), yticklabels=names,
               xlabel=f"{label} ({'pp' if paired or metric == 'event_flood_iou_std' else '%'})")
        ax.tick_params(axis="y", labelsize=8)
        ax.invert_yaxis()
        if paired:
            ax.axvline(0, color=".65", linestyle="--")
        if not selected:
            ax.text(.5, .5, "Per-event data unavailable", ha="center", transform=ax.transAxes)
            ax.set_xticks([])
        ax.grid(axis="x", alpha=.2)
    fig.suptitle(("Paired change" if paired else "Point estimate") + " and 95% event bootstrap intervals")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--ablation", required=True, type=Path)
    parser.add_argument("--artifacts", type=Path, default=REPO.parent / "artifacts")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must be a new directory")
    splits = load_cases(args.artifacts.resolve(), args.run.resolve(), args.ablation.resolve())
    rng = np.random.default_rng(SEED)
    estimates, differences = [], []
    for split, (ids, cases) in splits.items():
        summary, paired = bootstrap(split, ids, cases, rng)
        estimates.extend(summary)
        differences.extend(paired)
    args.out.mkdir(parents=True, exist_ok=False)
    write_csv(args.out / "confidence_intervals.csv", estimates)
    write_csv(args.out / "paired_differences.csv", differences)
    write_csv(args.out / "event_counts.csv", list(export_event_counts(splits)))
    for split in splits:
        draw_intervals(args.out / f"{split}_intervals.png", [r for r in estimates if r["split"] == split])
        primary = [r for r in differences if r["split"] == split and r["method"] == LOCKED]
        draw_intervals(args.out / f"{split}_paired_vs_calibrated.png", primary, paired=True)

    paper = "https://proceedings.neurips.cc/paper_files/paper/2024/file/43612b0662cb6a4986edf859fd6ebafe-Paper-Conference.pdf"
    blackbench = []
    for metric, reference in (("pooled_miou", .762), ("pooled_flood_f1", .8012)):
        row = one(estimates, split="test_historical", method=LOCKED, metric=metric)
        blackbench.append(dict(metric=metric, a3_point=row["point"], a3_ci_low=row["ci_low"],
                               a3_ci_high=row["ci_high"], blackbench_point=reference,
                               a3_minus_blackbench_pp=100*(row["point"]-reference), blackbench_ci_low=None,
                               blackbench_ci_high=None, status="point_reference_only_no_per_event_blackbench",
                               source=paper))
    write_csv(args.out / "blackbench_reference.csv", blackbench)
    (args.out / "config.json").write_text(json.dumps({
        "iterations": ITERATIONS, "random_seed": SEED, "ci": "percentile [2.5%, 97.5%]",
        "event_iou_std_ddof": 0, "sampling": "events with replacement; same draws within split",
        "multiseed_single_model": "recompute within each fixed seed, then average across seeds per draw",
        "paired_contrast": "method minus reference on matching event multiplicities",
        "coverage": {key: ids for key, (ids, _) in splits.items()},
        "run": str(args.run.resolve()), "ablation": str(args.ablation.resolve()),
        "scope": "historical artifacts only; conditional on observed events and fixed checkpoints/bias",
    }, indent=2)+"\n", encoding="utf-8")
    print(f"Saved bootstrap and paired intervals to {args.out}")


if __name__ == "__main__":
    main()
