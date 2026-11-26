#!/usr/bin/env python3
"""Matched event bootstrap and sensitivity analysis for saved IW-IoU validation runs."""
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from bootstrap_unet_events import (
    Case, ITERATIONS, METRICS as BOOT_METRICS, SEED, event_statistics, flood_counts,
    interval, read_csv, read_json, verify, write_csv,
)

GLOBAL, REF, BBSE = "global_bias", "source_macro_iou_reference", "bbse"
MARGIN, SAR, ORACLE = "iw_iou_margin", "iw_iou_margin_sar", "oracle_DIAGNOSTIC_ONLY"
VARIANTS = (MARGIN, SAR)
REFERENCES = (GLOBAL, REF, BBSE)
METHODS = (*REFERENCES, *VARIANTS)
PAIRS = tuple(product(VARIANTS, REFERENCES))
METRICS = {key: label for key, label in BOOT_METRICS.items() if key != "pooled_miou"}
NAMES = {GLOBAL: "Global +0.95", REF: "Source reference", BBSE: "BBSE",
         MARGIN: "IW margin", SAR: "IW margin + SAR"}
DIAGNOSTICS = {
    "ess_fraction_of_base": "Clipped ESS / base ESS",
    "source_overlap_mass": "Source overlap mass",
    "target_overlap_fraction": "Target overlap fraction",
}
TOL = 1e-12


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def keyed(rows, *fields):
    result = {tuple(row[field] for field in fields): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate rows keyed on {fields}")
    return result


def checked_counts(row, positives, scored):
    """Recover exact off-grid counts from published full-precision validation rates."""
    tp = round(float(row["flood_recall"]) * positives)
    predicted = round(float(row["predicted_prevalence_scored"]) * scored)
    if not 0 <= tp <= min(predicted, positives) or predicted - tp > scored - positives:
        raise ValueError(f"Invalid confusion counts for {row['event_id']} / {row['method']}")
    fp, fn = predicted - tp, positives - tp
    verify([tp / positives, tp / max(predicted, 1), predicted / scored,
            tp / max(tp + fp + fn, 1), 2*tp / max(2*tp + fp + fn, 1), positives / scored],
           [float(row[k]) for k in ("flood_recall", "flood_precision", "predicted_prevalence_scored",
                                    "flood_iou", "flood_f1", "actual_flood_prevalence")],
           f"{row['event_id']} / {row['method']} rates", atol=TOL)
    return np.array([tp, fp, fn], dtype=np.int64)


def load_artifacts(iw, history, confusions):
    iw_config, history_config = (read_json(root/"evaluation_config.json") for root in (iw, history))
    iw_lock, history_lock = (read_json(root/"bias_lock.json") for root in (iw, history))
    if (any(config["split"] != "validation" for config in (iw_config, history_config))
            or iw_lock["split"] != "validation"
            or iw_config["selection_sha256"] != history_lock["checkpoint_selection_sha256"]
            or iw_config["mean_cache_sha256"] != history_lock["cache_manifest_sha256"]
            or sha256(confusions) != iw_config["validation_confusions_sha256"]
            or any(sha256(root/"bias_lock.json") != config["bias_lock_sha256"]
                   for root, config in ((iw, iw_config), (history, history_config)))):
        raise ValueError("Artifacts do not share the locked validation predictions")
    events = sorted(iw_lock["events"], key=int)
    if len(events) != 6 or set(events) != set(history_lock["events"]):
        raise ValueError("Expected matching six official validation events")

    def records(directory, name):
        return keyed(read_csv(directory/name), "event_id", "method")

    iw_rows = records(iw, "event_metrics.csv")
    history_rows = records(history, "event_metrics.csv")
    diagnostics = records(iw, "density_ratio_diagnostics.csv")
    if set(diagnostics) != set(product(events, VARIANTS)):
        raise ValueError("Missing or unexpected density-ratio diagnostics")
    event_rows = []
    with np.load(confusions, allow_pickle=False) as archive:
        grid = archive["biases"]
        if set(archive.files) != {*events, "biases"} or not np.array_equal(grid, iw_lock["candidate_grid"]):
            raise ValueError("Validation confusion archive does not match the locked bias grid")
        for event in events:
            matrices = archive[event]
            n = int(matrices[0].sum())
            positives = int(matrices[0, 2].sum())
            if not 0 < positives <= n or np.any(matrices.sum((1, 2)) != n) or np.any(matrices[:, 2].sum(1) != positives):
                raise ValueError(f"Inconsistent ground truth support: {event}")
            exact = iw_rows[event, ORACLE]
            oracle_bias, oracle = float(exact["bias"]), float(exact["flood_iou"])
            old_oracle = history_rows[event, ORACLE]
            verify([oracle_bias, oracle], [float(old_oracle["bias"]), float(old_oracle["flood_iou"])],
                   f"{event} shared exact oracle", atol=TOL)
            for method in METHODS:
                row = (history_rows if method in (GLOBAL, BBSE) else iw_rows)[event, method]
                bias = float(row["bias"])
                if method in (GLOBAL, BBSE):
                    locked = (history_lock["global_bias"] if method == GLOBAL else
                              history_lock["events"][event]["methods"][BBSE]["bias"])
                    verify(bias, locked, f"{event} {method} bias lock", atol=TOL)
                    verify(int(row["evaluation_pixels"]), n, f"{event} {method} scored pixels", atol=TOL)
                    counts = checked_counts(row, positives, n)
                else:
                    verify(bias, iw_lock["events"][event]["biases"][method]["bias"],
                           f"{event} {method} bias lock", atol=TOL)
                    counts = np.array([int(row[k]) for k in ("tp", "fp", "fn")], dtype=np.int64)
                if method != BBSE:
                    ix, = np.flatnonzero(np.isclose(grid, bias, atol=TOL, rtol=0))
                    np.testing.assert_array_equal(counts, flood_counts(matrices[ix]),
                                                  err_msg=f"{event} {method} confusion archive")
                tp, fp, fn = map(int, counts)
                iou, f1 = tp/(tp + fp + fn), 2*tp/(2*tp + fp + fn)
                verify([tp + fn, iou, f1], [positives, float(row["flood_iou"]), float(row["flood_f1"])],
                       f"{event} {method} saved metrics", atol=TOL)
                if oracle < iou - TOL:
                    raise ValueError(f"IoU exceeds the exact oracle: {event} / {method}")
                event_rows.append(dict(event_id=event, method=method, bias=bias, scored_pixels=n,
                                       tp=tp, fp=fp, fn=fn, flood_iou=iou, flood_f1=f1,
                                       oracle_bias=oracle_bias, oracle_iou=oracle,
                                       oracle_gap_pp=100*(oracle-iou)))
    return events, event_rows, diagnostics


def analyze(events, event_rows, diagnostics):
    n = len(events)
    counts = {(r["event_id"], r["method"]): r for r in event_rows}
    cases = {method: Case(method, {}, "", flood=np.array([[
        [counts[event, method][key] for key in ("tp", "fp", "fn")] for event in events
    ]])) for method in METHODS}
    picks = np.random.default_rng(SEED).integers(n, size=(ITERATIONS, n))
    draws = (picks[..., None] == np.arange(n)).sum(axis=1)
    # First row is the full observed set, the remaining rows omit one event each.
    sensitivity = np.vstack((np.ones((1, n), dtype=int), 1-np.eye(n, dtype=int)))
    observed = {method: event_statistics(case, sensitivity) for method, case in cases.items()}
    replicates = {method: event_statistics(case, draws) for method, case in cases.items()}
    summary = [dict(method=method, events=n, **{f"{key}_pct": 100*float(values[0])
               for key, values in observed[method].items()},
               mean_oracle_gap_pp=float(np.mean([counts[e, method]["oracle_gap_pp"] for e in events])))
               for method in METHODS]

    paired, loo, differences, records, correlations = [], [], [], [], []
    for method, ref in PAIRS:
        wins = {"win": 0, "tie": 0, "loss": 0}
        delta_iou = []
        for event in events:
            a, b = counts[event, method], counts[event, ref]
            delta = a["flood_iou"] - b["flood_iou"]
            outcome = "win" if delta > TOL else "loss" if delta < -TOL else "tie"
            wins[outcome] += 1
            delta_iou.append(delta)
            d = diagnostics[event, method]
            differences.append(dict(event_id=event, method=method, reference=ref, outcome=outcome,
                                    method_bias=a["bias"], reference_bias=b["bias"],
                                    method_iou_pct=100*a["flood_iou"], reference_iou_pct=100*b["flood_iou"],
                                    delta_iou_pp=100*delta, delta_f1_pp=100*(a["flood_f1"]-b["flood_f1"]),
                                    oracle_iou_pct=100*a["oracle_iou"], method_oracle_gap_pp=a["oracle_gap_pp"],
                                    reference_oracle_gap_pp=b["oracle_gap_pp"],
                                    **{key: float(d[key]) for key in DIAGNOSTICS},
                                    ess_unclipped=float(d["ess_unclipped"]), ess_clipped=float(d["ess_clipped"])))
        records.append(dict(method=method, reference=ref, **wins,
                            mean_delta_iou_pp=100*float(np.mean(delta_iou)),
                            median_delta_iou_pp=100*float(np.median(delta_iou))))
        for key in DIAGNOSTICS:
            x = np.round([float(diagnostics[event, method][key]) for event in events], 12)
            y = np.round(delta_iou, 12)
            constant = np.ptp(x) == 0 or np.ptp(y) == 0
            correlations.append(dict(method=method, reference=ref, variable=key, events=n,
                                     spearman_rho=None if constant else float(spearmanr(x, y).statistic),
                                     status="constant input" if constant else "descriptive"))
        for metric in METRICS:
            changes = 100*(observed[method][metric]-observed[ref][metric])
            bootstrap_delta = 100*(replicates[method][metric]-replicates[ref][metric])
            lo, hi, valid = interval(bootstrap_delta)
            if valid != ITERATIONS:
                raise ValueError(f"Undefined bootstrap replicate: {method}/{ref}/{metric}")
            paired.append(dict(method=method, reference=ref, metric=metric, events=n, iterations=valid,
                               delta_pp=float(changes[0]), ci_low_pp=float(lo), ci_high_pp=float(hi)))
            for i, omitted in enumerate(("none", *events)):
                loo.append(dict(method=method, reference=ref, metric=metric, omitted_event=omitted,
                                events=n-int(i > 0), method_value_pct=100*float(observed[method][metric][i]),
                                reference_value_pct=100*float(observed[ref][metric][i]),
                                delta_pp=float(changes[i]), change_vs_full_delta_pp=float(changes[i]-changes[0])))
    return summary, paired, loo, differences, records, correlations


def plot_intervals(out, paired):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True, layout="constrained")
    labels = [f"{NAMES[method]} vs {NAMES[ref]}" for method, ref in PAIRS]
    for ax, (metric, label) in zip(axes, METRICS.items(), strict=True):
        rows = [r for r in paired if r["metric"] == metric]
        for i, row in enumerate(rows):
            color = "#176b65" if row["method"] == MARGIN else "#aa4b36"
            ax.hlines(i, row["ci_low_pp"], row["ci_high_pp"], color=color)
            ax.plot(row["delta_pp"], i, "o", color=color)
        ax.axvline(0, linestyle="--", color=".55")
        ax.set(xlabel="Difference (pp)", title=label, yticks=range(len(labels)), yticklabels=labels)
    axes[0].invert_yaxis()
    fig.suptitle("Paired event bootstrap · 95% percentile intervals")
    fig.savefig(out/"paired_bootstrap.png", dpi=165)
    plt.close(fig)


def plot_sensitivity(out, events, loo, differences):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True, layout="constrained")
    row_labels = [f"{NAMES[method]} vs {NAMES[ref]}" for method, ref in PAIRS]
    panels = [("Per-event Flood IoU difference", events,
               [[r["delta_iou_pp"] for r in differences if (r["method"], r["reference"]) == pair]
                for pair in PAIRS])]
    for metric, label in METRICS.items():
        panels.append((f"{label} · omit one event", ("All", *events),
                       [[r["delta_pp"] for r in loo if (r["method"], r["reference"], r["metric"]) == (*pair, metric)]
                        for pair in PAIRS]))
    for ax, (title, columns, values) in zip(axes.flat, panels, strict=True):
        values = np.array(values)
        limit = max(np.abs(values).max(), 1e-9)
        img = ax.imshow(values, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
        for (i, j), v in np.ndenumerate(values):
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                    color="white" if abs(v) > .65*limit else "black", fontsize=8)
        ax.set(title=title, xticks=range(len(columns)), xticklabels=columns,
               yticks=range(len(row_labels)), yticklabels=row_labels)
        ax.tick_params(axis="x", rotation=25)
        fig.colorbar(img, ax=ax, label="Difference (pp)", shrink=.75)
    fig.savefig(out/"event_sensitivity.png", dpi=165)
    plt.close(fig)


def plot_diagnostics(out, differences):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=True, layout="constrained")
    for axs, method in zip(axes, VARIANTS, strict=True):
        subset = [r for r in differences if r["reference"] == REF and r["method"] == method]
        for ax, (key, label) in zip(axs, DIAGNOSTICS.items(), strict=True):
            for r in subset:
                x, y = 100*r[key], r["delta_iou_pp"]
                ax.scatter(x, y, color="#176b65" if method == MARGIN else "#aa4b36")
                ax.annotate(r["event_id"], (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
            ax.axhline(0, linestyle="--", color=".55")
            ax.set(xlabel=f"{label} (%)")
            ax.margins(x=.30, y=.15)
        axs[0].set_ylabel(f"{NAMES[method]}\nFlood IoU difference (pp)")
    fig.suptitle("ESS and overlap vs event IoU difference from Source Reference")
    fig.savefig(out/"density_ratio_diagnostics.png", dpi=165)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("iw-results", "bias-results", "validation-confusions", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    events, rows, diagnostics = load_artifacts(args.iw_results, args.bias_results, args.validation_confusions)
    summary, paired, loo, differences, records, correlations = analyze(events, rows, diagnostics)
    args.out.mkdir(parents=True, exist_ok=False)
    for name, data in (("summary", summary), ("event_metrics", rows), ("paired_bootstrap", paired),
                       ("leave_one_event_out", loo), ("event_differences", differences),
                       ("win_tie_loss", records), ("ess_overlap_correlations", correlations)):
        write_csv(args.out/f"{name}.csv", data)
    plot_intervals(args.out, paired)
    plot_sensitivity(args.out, events, loo, differences)
    plot_diagnostics(args.out, differences)
    inputs = [root/name for root, files in (
        (args.iw_results, ("evaluation_config.json", "bias_lock.json", "event_metrics.csv",
                           "density_ratio_diagnostics.csv")),
        (args.bias_results, ("evaluation_config.json", "bias_lock.json", "event_metrics.csv")),
    ) for name in files] + [args.validation_confusions]
    config = dict(split="validation", scope="exploratory analysis of frozen validation event metrics",
                  event_ids=events, source_sha256={str(p.resolve()): sha256(p) for p in inputs},
                  code_sha256=sha256(Path(__file__)),
                  bootstrap_seed=SEED, iterations=ITERATIONS, interval_percentile=[2.5, 97.5],
                  bootstrap="paired events with replacement; same event multiplicities across methods",
                  pooled="sum sampled event TP/FP/FN before computing Flood F1",
                  event_std_ddof=0, difference="IW-IoU minus comparator", tie_iou_tolerance=TOL,
                  leave_one_out="omit each event from metric aggregation using already locked biases",
                  oracle="exact event oracle from shared historical artifacts, diagnostics only",
                  limitations=["six events; bootstrap intervals are descriptive",
                               "adaptation folds overlap in source events",
                               "Global Bias and frozen checkpoints were selected on these validation events",
                               "bootstrap does not refit the adaptation rules"])
    (args.out/"analysis_config.json").write_text(json.dumps(config, indent=2, allow_nan=False)+"\n",
                                                encoding="utf-8")
    print(f"Saved six-event IW-IoU validation audit to {args.out}", flush=True)


if __name__ == "__main__":
    main()
