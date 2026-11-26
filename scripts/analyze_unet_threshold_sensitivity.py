#!/usr/bin/env python3
"""Experiment 1: event threshold curves from frozen logits and saved annotations."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from adapt_unet_event_bias import digest, flood_margin, load_json, oracle_from_ranks, scores, write_json
from audit_unet_bias_failures import read_csv, save_csv
from ccla_crossfit_cache import cache_path, lookup, read_manifest, values
from evaluate_unet_best_methods_test import (
    EVENT, F1, GLOBAL, NOBIAS, REF, SCOPE, check_oof, labels_for_event, prediction_lock,
    select_rows, test_rows,
)
from evaluate_unet_eoa_diagnostics import pr_curve
from kuro_dataset import RawArchive
from train_unet import IGNORE

LOCKED = {NOBIAS: 0., GLOBAL: .95, F1: 1.50, REF: 3.70}
LABEL = {NOBIAS: "No Bias", GLOBAL: "Global +0.95", F1: "OOF F1 +1.50",
         REF: "OOF Reference +3.70", EVENT: "Event Oracle"}
COLORS = {NOBIAS: "#555555", GLOBAL: "#0072b2", F1: "#009e73", REF: "#cc79a7", EVENT: "#d55e00"}
SPLITS = {"oof_train": (27, 4000), "validation": (6, 959), "historical_test": (10, 2000)}
SAR_FEATURES = ("post_vv_db", "post_vh_db", "delta_vv_db", "delta_vh_db", "vh_minus_vv_db", "slope_deg")
RANK_COLUMNS = ("flood_iou", "flood_f1", "flood_precision", "flood_recall")
CHUNK = 500_000


def save_figure(fig, out, name):
    for extension in ("pdf", "png"):
        fig.savefig(out / f"{name}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def historical_matrices(root):
    with np.load(root / "event_confusions.npz", allow_pickle=False) as data:
        events, methods = data["event_ids"].tolist(), data["methods"].tolist()
        return {e: dict(zip(methods, row, strict=True))
                for e, row in zip(events, data["confusion"], strict=True)}


def inputs(run, test, audit):
    frozen_path, predictor_path = run / "frozen_config.json", run / "fit/predictor.json"
    plan, predictor = load_json(frozen_path), load_json(predictor_path)
    run_test_cfg = load_json(test / "frozen_config.json")
    audit_cfg = load_json(audit / "audit_config.json")
    test_results = test / "results"
    for name in ("event_confusions.npz", "evaluation_config.json", "event_metrics.csv",
                 "test_coverage.csv", "oracle_DIAGNOSTIC_ONLY/event_biases.csv"):
        path = test_results / name
        if audit_cfg["input_sha256"].get(str(path)) != digest(path):
            raise ValueError(f"Historical test input differs from previous audit: {path}")
    for name in ("test_manifest.json", "test_predictions_lock.json", "test_bias_lock.json"):
        path = test / name
        if audit_cfg["input_sha256"].get(str(path)) != digest(path):
            raise ValueError(f"Historical test lock differs from previous audit: {path}")
    scalar_lock = test / "scalar_inputs/scalar_bias_lock.json"
    lock = load_json(scalar_lock)
    if (lock["run_id"] != plan["run_id"] or lock["frozen_config_sha256"] != digest(frozen_path)
            or lock["predictor_sha256"] != digest(predictor_path)
            or lock["oof_curves_sha256"] != digest(run / "fit/oof_patch_curves.npz")
            or any(lock["chosen"][m]["bias"] != LOCKED[m] for m in (F1, REF))):
        raise ValueError("Archived OOF scalar lock differs from the source artifacts")
    oof = check_oof(run, predictor)
    for path, _ in oof:
        fold = path.parent.name.removeprefix("fold-")
        if digest(path.parent / "split_manifest.json") != plan["folds"][fold]:
            raise ValueError(f"Changed split manifest: {fold}")
    val = run / "validation"
    val_cfg = load_json(run / "results/evaluation_config.json")
    val_scope = run / "validation_manifest.json"
    val_manifest = read_manifest(val, dict(split="validation", role="validation_ensemble",
        features_only=False, member_seeds=[42,43,44], predictor_sha256=digest(predictor_path),
        scope_sha256=digest(val_scope), checkpoint_sha256=[m["sha256"] for m in plan["full_train"]]))
    if ([p["patch_id"] for p in val_manifest["patches"]] != load_json(val_scope)["patch_ids"]
            or digest(val / "manifest.json") != val_cfg["validation_cache_sha256"]):
        raise ValueError("Changed validation cache scope")
    test_cache, test_manifest = prediction_lock(test, run_test_cfg)
    if [m["sha256"] for m in run_test_cfg["members"]] != val_manifest["checkpoint_sha256"]:
        raise ValueError("Validation and historical test used different checkpoints")
    paths = {"oof_train": oof, "validation": [(val, val_manifest)],
             "historical_test": [(test_cache, test_manifest)]}
    indexes = {s: lookup(p) for s, p in paths.items()}
    seen_events, seen_patches = set(), set()
    for split, (index, patches) in indexes.items():
        ids = [p["patch_id"] for rows in patches.values() for p in rows]
        if ((len(index), len(ids)) != SPLITS[split] or seen_events.intersection(index)
                or len(set(ids)) != len(ids) or seen_patches.intersection(ids)):
            raise ValueError(f"Unexpected event or patch coverage: {split}")
        seen_events.update(index)
        seen_patches.update(ids)
    return indexes, paths, run_test_cfg


def cumulative_confusions(margin, nonflood, labels, grid):
    """Move each pixel from its non-flood label to Flood at its first strict crossing."""
    if not (len(margin) == len(nonflood) == len(labels)) or not len(labels):
        raise ValueError("Empty or misaligned sweep inputs")
    bins = np.zeros((len(grid) + 1, 3, 2), dtype=np.int64)
    for k in range(0, len(labels), CHUNK):
        sl = slice(k, k + CHUNK)
        crossing = np.searchsorted(grid, -margin[sl].astype(np.float64), side="right")
        codes = 6 * crossing + 2 * labels[sl].astype(np.int64) + nonflood[sl]
        bins += np.bincount(codes, minlength=bins.size).reshape(bins.shape)
    switched = bins.cumsum(0)[:-1]
    cms = np.empty((len(grid), 3, 3), dtype=np.int64)
    cms[:, :, :2] = bins.sum(0) - switched
    cms[:, :, 2] = switched.sum(2)
    return cms


def score_groups(margin, flood):
    order = np.argsort(margin)
    ordered = margin[order]
    edges = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    return ordered[edges], np.add.reduceat(flood[order], edges, dtype=np.int64), np.diff(np.r_[edges, len(order)])


def sar_medians(sar, extra, keep, flood):
    data = (sar[:, 0], extra[:, 0], sar[:, 1], sar[:, 2], extra[:, 0] - sar[:, 0], extra[:, 1])
    row = {}
    for name, channel in zip(SAR_FEATURES, data, strict=True):
        x = channel[keep]
        for subset, a in (("all", x), ("flood", x[flood])):
            finite = a[np.isfinite(a)]
            row[f"{subset}_{name}_median"] = float(np.median(finite)) if len(finite) else None
            row[f"{subset}_{name}_n"] = len(finite)
    return row


def event_curve(z, y, sar, extra, grid):
    keep = y != IGNORE
    z, labels = z[keep], y[keep]
    if not len(labels) or not np.isfinite(z).all() or not np.isin(labels, [0,1,2]).all():
        raise ValueError("Invalid logits or ground truth")
    margin, nonflood = flood_margin(z), z[:, :2].argmax(1).astype(np.uint8)
    flood = labels == 2
    rank = score_groups(margin, flood)
    bias, iou = oracle_from_ranks(rank) if flood.any() else (0., 0.)
    ap = pr_curve(rank)[3] if flood.any() else None
    extremes = [-float(rank[0][-1]), float(np.nextafter(-np.float64(rank[0][0]), np.inf))]
    full_grid = np.unique(np.r_[grid, extremes, bias])
    cms = cumulative_confusions(margin, nonflood, labels, full_grid)
    at = lambda b: cms[np.searchsorted(full_grid, b)]
    np.testing.assert_allclose(scores(at(bias))["flood_iou"], iou, atol=1e-14, rtol=0)
    if np.any(at(extremes[0])[:, 2]) or np.any(at(extremes[1])[:, :2]):
        raise ValueError("Threshold extremes do not attain the empty and full Flood masks")
    row = dict(pixels=len(labels), flood_pixels=int(flood.sum()), prevalence=float(flood.mean()), ap=ap,
               oracle_bias=bias, oracle_iou=iou, oracle_identifiable=bool(flood.any()),
               empty_flood_bias=extremes[0], full_flood_bias=extremes[1])
    for method, b in LOCKED.items():
        at_bias = scores(at(b))
        row[f"{method}_iou"] = at_bias["flood_iou"]
        row[f"{method}_oracle_gain_pp"] = 100 * (iou - at_bias["flood_iou"])
    row.update(sar_medians(sar, extra, keep, flood))
    return full_grid, cms, row


def verify_event(split, e, grid, cms, row, archived, oof_counts):
    actual = {m: cms[np.searchsorted(grid, b)] for m, b in LOCKED.items()}
    if split == "oof_train":
        for b, expected in oof_counts[e].items():
            cm = cms[np.searchsorted(grid, b)]
            tp, fp, fn = cm[2, 2], cm[:2, 2].sum(), cm[2, :2].sum()
            np.testing.assert_array_equal([tp, fp, fn], expected, err_msg=f"{e} OOF bias {b}")
        return
    saved, ap, coverage, oracles = archived[split]
    for method, cm in actual.items():
        if method in saved[e]:
            np.testing.assert_array_equal(cm, saved[e][method], err_msg=f"{split}/{e}/{method}")
    np.testing.assert_allclose(row["ap"], ap[e], rtol=0, atol=1e-10, err_msg=f"{split}/{e} AP")
    if e in coverage:
        np.testing.assert_array_equal([row["pixels"], row["flood_pixels"]], coverage[e])
    oracle_cm = cms[np.searchsorted(grid, row["oracle_bias"])]
    if split == "historical_test":
        np.testing.assert_array_equal(oracle_cm, saved[e][EVENT], err_msg=f"{e} historical oracle")
        np.testing.assert_allclose([row["oracle_bias"], row["oracle_iou"]], oracles[e], atol=1e-12, rtol=0)
    elif scores(oracle_cm)["flood_iou"] + 1e-12 < scores(saved[e][EVENT])["flood_iou"]:
        raise ValueError(f"Exact oracle is worse than the archived grid oracle: {e}")


def render_curves(out, split, curves, summaries):
    columns = ("IoU", "F1", "Precision", "Recall")
    entries = sorted(curves, key=int)
    for page, start in enumerate(range(0, len(entries), 6), 1):
        chunk = entries[start:start+6]
        fig, axes = plt.subplots(len(chunk), 4, figsize=(13, 1.65*len(chunk)+.8),
                                 sharey=True, squeeze=False, layout="constrained")
        for row, e in zip(axes, chunk, strict=True):
            grid, cm = curves[e]
            metrics = [scores(c) for c in cm]
            selected = LOCKED | {EVENT: summaries[e]["oracle_bias"]}
            for ax, name, key in zip(row, columns, RANK_COLUMNS, strict=True):
                ax.plot(grid, [m[key] for m in metrics], color="#222222", lw=1)
                for method, b in selected.items():
                    ax.axvline(b, color=COLORS[method], alpha=.7, lw=.8, ls="--" if method == EVENT else "-")
                ax.set_ylim(-.025, 1.025)
                ax.set_xlim(min(-8, selected[EVENT]-.75), max(10, selected[EVENT]+.75))
                ax.set_title(f"{e} · {name}", fontsize=9)
                ax.grid(alpha=.15)
        fig.supxlabel("Flood-logit bias (b)", fontsize=10)
        handles = [plt.Line2D([0], [0], color=color, label=LABEL[m], ls="--" if m == EVENT else "-")
                   for m, color in COLORS.items()]
        fig.legend(handles=handles, loc="outside upper center", ncols=5, frameon=False, fontsize=8)
        save_figure(fig, out, f"{split}_event_curves_{page:02d}")


def plot_pair(out, curves, summaries):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.1), sharey=True, layout="constrained")
    for ax, key in zip(axes, RANK_COLUMNS, strict=True):
        for e,color in (("411", "#0072b2"), ("445", "#d55e00")):
            grid, cm = curves[e]
            b = summaries[e]["oracle_bias"]
            ax.plot(grid, [scores(c)[key] for c in cm], color=color, label=f"Event {e}")
            ax.axvline(b, ls="--", color=color, alpha=.75)
        for b in LOCKED.values():
            ax.axvline(b, color="#888888", lw=.65, alpha=.6)
        ax.set(xlabel="Flood-logit bias", title=key.removeprefix("flood_").title(), ylim=(-.025, 1.025))
        ax.set_xlim(min(-8, *(summaries[e]["oracle_bias"]-.75 for e in ("411","445"))),
                    max(10, *(summaries[e]["oracle_bias"]+.75 for e in ("411","445"))))
        ax.grid(alpha=.15)
    axes[0].legend(frameon=False)
    save_figure(fig, out, "historical_test_411_vs_445")


def correlations(events):
    fields = ["prevalence", "ap", *(f"{pop}_{key}_median" for pop in ("all", "flood") for key in SAR_FEATURES)]
    rows = []
    for field in fields:
        pairs = [(r["oracle_bias"], r[field]) for r in events if r["oracle_identifiable"] and r[field] is not None]
        rho = p = None
        if len(pairs) >= 3:
            x, y = np.asarray(pairs).T
            if np.ptp(x) and np.ptp(y):
                rho, p = map(float, spearmanr(x, y))
        rows.append(dict(feature=field, events=len(pairs), spearman_rho=rho, p_value_unadjusted=p))
    return rows


def distributions(events):
    rows = []
    for field in ["oracle_bias", *(f"{m}_oracle_gain_pp" for m in LOCKED)]:
        values = np.array([r[field] for r in events if r["oracle_identifiable"]])
        q1, median, q3 = np.quantile(values, [.25,.5,.75])
        rows.append(dict(variable=field, events=len(values), mean=values.mean(), median=median,
            population_std=values.std(), q1=q1, q3=q3, iqr=q3-q1, minimum=values.min(), maximum=values.max()))
    return rows


def report(out, distributions_, correlations_, pair, step):
    lines = ["# Experiment 1 — Threshold sensitivity", "",
        f"Plotting grid b ∈ [−20, +20] in steps of {step:g}, with locked biases, exact per-event oracle and "
        "empty/all-Flood endpoints added. Curves use strict margin > −b. Precision is zero with zero predicted positives. "
        "AP uses continuous raw flood margins with exact ties grouped. Oracle maximizes event Flood IoU over all attainable "
        "thresholds, resolves ties by smallest absolute bias, and is diagnostic only.", "",
        "OOF covers 4,000 patches from event-disjoint fold-specific ensembles. Validation covers 959 footprint-filtered patches "
        "and historical test covers the official 2,000 patches using the full-training ensemble. The test analysis is a "
        "historical-test exploratory evaluation. The historical test cache contains no labels: annotations were reread from "
        "the existing dataset release and aligned by patch ID and saved input-valid masks. Archived test confusions and "
        "oracle metrics were reproduced. The inputs and predictions remained frozen.", "",
        "| Split | Events | Oracle bias mean | Median | Std | IQR | Min to max | Mean IoU gain vs No/Global/F1/Ref (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for split, rows in distributions_.items():
        b = rows[0]
        gains = " / ".join(f"{r['mean']:.2f}" for r in rows[1:])
        lines.append(f"| {split} | {b['events']} | {b['mean']:.2f} | {b['median']:.2f} | {b['population_std']:.2f} | "
                     f"{b['iqr']:.2f} | {b['minimum']:.2f} to {b['maximum']:.2f} | {gains} |")
    lines += ["", "## Events 411 and 445", "",
        "| Event | Prevalence (%) | Flood AP (%) | Oracle bias | IoU No/Global/F1/Ref/Oracle (%) | Flood median post-VV / ΔVV (dB) |",
        "|---|---:|---:|---:|---:|---:|"]
    for e, r in pair.items():
        vals = " / ".join(f"{100*r[k]:.2f}" for k in [*(f"{m}_iou" for m in LOCKED), "oracle_iou"])
        lines.append(f"| {e} | {100*r['prevalence']:.3f} | {100*r['ap']:.2f} | {r['oracle_bias']:+.4f} | {vals} | "
                     f"{r['flood_post_vv_db_median']:.2f} / {r['flood_delta_vv_db_median']:.2f} |")
    lines += ["", "## Evidence and limitations", "",
        "An event with high AP and a large oracle IoU gain has a stronger ranking than its fixed operating point suggests. "
        "The 411–445 figure separates this effect from the trade-off between recovering positives and admitting false positives."]
    for split, rows in correlations_.items():
        selected = [r for r in rows if r["feature"] in ("prevalence", "ap") and r["spearman_rho"] is not None]
        if selected:
            values = ", ".join(f"{r['feature']} ρ={r['spearman_rho']:+.3f} (n={r['events']})" for r in selected)
            lines.append(f"- {split}: {values}.")
    lines += ["", "SAR correlations use event medians on all scored pixels and separately on actual Flood pixels. "
        "SAR associations, prevalence and AP are observational. Event counts are small and p-values in the CSV are "
        "unadjusted across correlated features. OOF fold models have a smaller training event set than the full-training "
        "ensemble, so split differences combine sample composition and model coverage.", "",
        "The scalar biases are historical operating points. Diagnostic curves and oracles do not select or update them."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oof-run", type=Path, required=True)
    p.add_argument("--test-run", type=Path, required=True)
    p.add_argument("--prior-audit", type=Path, required=True, help="Prior audit results directory")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    run, test, audit, out = (x.resolve() for x in (args.oof_run, args.test_run, args.prior_audit, args.out))
    indexes, paths, test_cfg = inputs(run, test, audit)
    out.mkdir(parents=True, exist_ok=False)
    with np.load(run / "fit/oof_patch_curves.npz", allow_pickle=False) as d:
        counts, grid, events = d["counts"], d["biases"], d["event_ids"]
        oof_counts = {e: dict(zip(grid, counts[events==e].sum(0), strict=True)) for e in np.unique(events)}
    step = .05
    main_grid = np.round(np.arange(-400,401)*step, 2)
    common = np.unique(np.r_[main_grid, grid, list(LOCKED.values())])
    archived = {}
    for split, root in (("validation", run/"results"), ("historical_test", test/"results")):
        tables = {r["event_id"]: float(r["flood_ap" if split == "historical_test" else "ap"])
                  for r in read_csv(root/"event_metrics.csv") if r["method"]==NOBIAS}
        cov, oracles = {}, {}
        if split == "historical_test":
            cov = {r["event_id"]: [int(r["scored_pixels"]), int(r["flood_pixels"])]
                   for r in read_csv(root/"test_coverage.csv")}
            oracles = {r["event_id"]: [float(r["bias"]), float(r["flood_iou"])]
                       for r in read_csv(root/"oracle_DIAGNOSTIC_ONLY/event_biases.csv")}
        archived[split] = (historical_matrices(root), tables, cov, oracles)
    distributions_, correlations_, pair, provenance = {}, {}, {}, {}
    for split, (index, patches) in indexes.items():
        split_out = out / split
        split_out.mkdir()
        is_test = split == "historical_test"
        if is_test:
            archive = RawArchive(test_cfg["raw"])
            scope = test_rows(archive.frame, test, test_cfg)
        summaries, curves = {}, {}
        rows = []
        for e in sorted(index, key=int):
            z = values(index,e,"logits")
            y = (labels_for_event(archive, select_rows(scope,[p["patch_id"] for p in patches[e]]),
                    patches[e], cache_path(index[e][0],e,"masks"),len(z)) if is_test else values(index,e,"labels"))
            bs, cms, summary = event_curve(z,y,values(index,e,"sar"),values(index,e,"extra"),common)
            summary["patches"] = len(patches[e])
            verify_event(split,e,bs,cms,summary,archived,oof_counts)
            summaries[e] = summary
            curves[e] = (bs,cms)
            for b, cm in zip(bs,cms,strict=True):
                method = next((m for m, v in LOCKED.items() if b==v), "")
                rows.append(dict(event_id=e,bias=b,locked_method=method,
                    exact_event_oracle=b==summary["oracle_bias"],**scores(cm),
                    tp=cm[2,2],fp=cm[:2,2].sum(),fn=cm[2,:2].sum()))
            print(f"{split}: event {e} verified", flush=True)
        save_csv(split_out/"event_curves.csv",rows)
        save_csv(split_out/"event_summary.csv",[dict(event_id=e,**r) for e,r in summaries.items()])
        distributions_[split], correlations_[split] = distributions(list(summaries.values())), correlations(list(summaries.values()))
        save_csv(split_out/"oracle_distributions.csv",distributions_[split])
        save_csv(split_out/"spearman.csv",correlations_[split])
        ids = list(curves)
        matrices = np.stack([curves[e][1][np.searchsorted(curves[e][0],common)] for e in ids])
        oracle = np.stack([curves[e][1][np.searchsorted(curves[e][0],summaries[e]["oracle_bias"])] for e in ids])
        product = split_out/"event_threshold_confusions.npz"
        np.savez_compressed(product,event_ids=ids,grid=common,confusions=matrices,oracle_confusions=oracle)
        provenance[split] = dict(archive_sha256=digest(product),events=len(ids),patches=sum(len(v) for v in patches.values()),
            cache_manifests={str(path/"manifest.json"):digest(path/"manifest.json") for path,_ in paths[split]})
        render_curves(split_out, split, curves, summaries)
        if is_test:
            pair = {e:summaries[e] for e in ("411","445")}
            save_csv(split_out/"events_411_445.csv",[dict(event_id=e,**r) for e,r in pair.items()])
            plot_pair(split_out,curves,summaries)
    report(out,distributions_,correlations_,pair,step)
    write_json(out/"analysis_config.json",dict(scope=SCOPE,locked=LOCKED,grid_step=step,
        oracle="exact per-event Flood IoU upper bound, diagnostic only",splits=provenance,
        oof_run=str(run),test_run=str(test),prior_audit=str(audit),
        prior_audit_config_sha256=digest(audit/"audit_config.json"),script_sha256=digest(Path(__file__))))
    print(f"Saved Experiment 1: {out}",flush=True)


if __name__ == "__main__":
    main()
