#!/usr/bin/env python3
"""Experiment 2: paired event analysis of the four frozen scalar thresholds."""
import argparse
from pathlib import Path

import numpy as np

from analyze_unet_threshold_sensitivity import (
    COLORS, LABEL, LOCKED, SPLITS, digest, load_json, plt, read_csv, save_csv, save_figure, scores, write_json,
)
from audit_unet_historical_test import METRICS, ORACLE_METRICS, all_stats
from bootstrap_unet_events import ITERATIONS, SEED, interval
from evaluate_unet_best_methods_test import EVENT, F1, GLOBAL, NOBIAS, REF

REFERENCES = (NOBIAS, GLOBAL, F1)
TOL = 1e-12


def counts(cm):
    return dict(tp=int(cm[2,2]),fp=int(cm[:2,2].sum()),fn=int(cm[2,:2].sum()))


def report_stats(mats, weights):
    result = all_stats(mats, weights)
    # Center each draw directly so constant-IoU draws have zero dispersion.
    for method, cm in mats.items():
        iou = np.array([scores(c)["flood_iou"] for c in cm])
        centered = iou[None, :] - result[method]["event_macro_flood_iou"][:, None]
        result[method]["event_flood_iou_std"] = np.sqrt(
            (weights * centered**2).sum(axis=1) / weights.sum(axis=1))
    return result


def compare(mats, events):
    n = len(events)
    omissions = np.vstack([np.ones(n,dtype=int), 1-np.eye(n,dtype=int)])
    point = report_stats(mats, omissions)
    draws = np.random.default_rng(SEED).integers(n,size=(ITERATIONS,n))
    boot = report_stats(mats, (draws[...,None]==np.arange(n)).sum(axis=1))
    result = {k:[] for k in ("summary","paired_bootstrap","event_differences","win_tie_loss",
                              "leave_one_event_out","event_oracle_gap","pooled_confusion_deltas")}
    oracle = point[EVENT]
    for m, data in point.items():
        result["summary"].append(dict(method=m,bias=LOCKED.get(m),diagnostic_only=m==EVENT,
            **{k:float(v[0]) for k,v in data.items()},**counts(mats[m].sum(0))))
    for ref in REFERENCES:
        a, b = point[REF], point[ref]
        outcomes = {"win":0, "tie":0, "loss":0}
        for e, ac, bc, oc in zip(events,mats[REF],mats[ref],mats[EVENT],strict=True):
            s,t,o = scores(ac), scores(bc), scores(oc)
            d = s["flood_iou"]-t["flood_iou"]
            outcome = "win" if d>TOL else "loss" if d<-TOL else "tie"
            outcomes[outcome] += 1
            result["event_differences"].append(dict(reference=ref,event_id=e,outcome=outcome,
                source_reference_iou=s["flood_iou"], comparator_iou=t["flood_iou"],delta_iou_pp=100*d,
                oracle_iou=o["flood_iou"],delta_precision_pp=100*(s["flood_precision"]-t["flood_precision"]),
                delta_recall_pp=100*(s["flood_recall"]-t["flood_recall"]),
                **{f"delta_{k}":v-counts(bc)[k] for k,v in counts(ac).items()}))
        result["win_tie_loss"].append(dict(reference=ref,events=n,**outcomes))
        result["pooled_confusion_deltas"].append(dict(reference=ref,
            delta_precision_pp=100*(a["pooled_flood_precision"][0]-b["pooled_flood_precision"][0]),
            delta_recall_pp=100*(a["pooled_flood_recall"][0]-b["pooled_flood_recall"][0]),
            **{f"delta_{k}":v-counts(mats[ref].sum(0))[k] for k,v in counts(mats[REF].sum(0)).items()}))
        for metric in (*METRICS,"pooled_flood_precision","pooled_flood_recall"):
            delta = 100*(a[metric]-b[metric])
            lo, hi, k = interval(100*(boot[REF][metric]-boot[ref][metric]))
            result["paired_bootstrap"].append(dict(reference=ref,metric=metric,delta_pp=float(delta[0]),
                ci_low_pp=lo,ci_high_pp=hi,iterations=k))
            for j,e in enumerate(("none", *events)):
                result["leave_one_event_out"].append(dict(reference=ref,metric=metric,omitted_event=e,
                    events=n-int(j>0),source_reference_value=a[metric][j],comparator_value=b[metric][j],
                    delta_pp=delta[j],change_vs_full_pp=delta[j]-delta[0]))
        for metric in ORACLE_METRICS:
            o, x, y = oracle[metric][0], a[metric][0], b[metric][0]
            is_std = metric=="event_flood_iou_std"
            ga, gb = (abs(x-o),abs(y-o)) if is_std else (o-x,o-y)
            result["event_oracle_gap"].append(dict(reference=ref,metric=metric,oracle_value=o,
                source_reference_gap_pp=100*ga,comparator_gap_pp=100*gb,gap_reduction_pp=100*(gb-ga),
                gap_reduction_pct=100*(gb-ga)/gb if gb>TOL else None,
                gap_definition="absolute distance to oracle" if is_std else "oracle minus estimate"))
    return result


def tradeoff(out, split, grid, cms):
    total = cms.sum(0)
    tp = total[:,2,2]
    f1 = 2*tp / np.maximum(total[:,2,:].sum(1)+total[:,:,2].sum(1),1)
    tp = cms[:,:,2,2]
    macro = (tp/np.maximum(cms[:,:,2,:].sum(2)+cms[:,:,:,2].sum(2)-tp,1)).mean(0)
    locked = {b:m for m,b in LOCKED.items()}
    save_csv(out/"tradeoff.csv",[dict(bias=b,pooled_flood_f1=f,event_macro_flood_iou=i,
        status="locked" if b in locked else "diagnostic_only",method=locked.get(b,""))
        for b,f,i in zip(grid,f1,macro,strict=True)])
    fig, (left,right) = plt.subplots(1,2,figsize=(10,4),layout="constrained")
    left.plot(100*f1,100*macro,color="#444444",ls=":",label="Diagnostic bias sweep")
    right.plot(grid,100*f1,color="#1b4965",label="Pooled Flood F1")
    right.plot(grid,100*macro,color="#b15928",label="Macro Flood IoU")
    for method,b in LOCKED.items():
        j = np.searchsorted(grid,b)
        left.scatter(100*f1[j],100*macro[j],color=COLORS[method],label=LABEL[method],zorder=4)
        right.axvline(b,color=COLORS[method],lw=.85,alpha=.8)
    left.set(xlabel="Pooled Flood F1 (%)",ylabel="Event-macro Flood IoU (%)")
    right.set(xlabel="Flood-logit bias",ylabel="Metric (%)",xlim=(-8,10))
    for ax in (left,right):
        ax.legend(frameon=False,fontsize=8)
        ax.grid(alpha=.18)
    save_figure(fig,out,f"{split}_tradeoff")


def plot_ci(out, reports):
    fig,axes = plt.subplots(2,3,figsize=(13,6.3),sharey=True,layout="constrained")
    for ax,(key,label) in zip(axes.flat,METRICS.items(),strict=True):
        positions = np.arange(3)
        for j,(split,rs) in enumerate(reports.items()):
            evidence = {r["reference"]:r for r in rs["paired_bootstrap"] if r["metric"]==key}
            rows = [evidence[m] for m in REFERENCES]
            mid,lo,hi = np.array([[r[k] for r in rows] for k in ("delta_pp","ci_low_pp","ci_high_pp")])
            pos = positions + .22*(j-1)
            color = ("#0072b2","#009e73","#d55e00")[j]
            ax.hlines(pos,lo,hi,color=color,lw=1.2)
            ax.scatter(mid,pos,color=color,s=18,label=split)
        ax.axvline(0,ls="--",lw=.7,color="#444444")
        ax.set(title=label,xlabel="OOF Reference − comparator (pp)",yticks=positions,
               yticklabels=[LABEL[m] for m in REFERENCES],ylim=(2.45,-.45))
        ax.grid(axis="x",alpha=.18)
    axes[0,0].legend(frameon=False,fontsize=8)
    save_figure(fig,out,"paired_bootstrap")


def verify_replayed_metrics(config, split, report):
    if split == "oof_train":
        lock = load_json(Path(config["test_run"])/"scalar_inputs/scalar_bias_lock.json")["chosen"]
        for row in report["summary"]:
            method = row["method"]
            if method in lock:
                np.testing.assert_allclose([row["event_macro_flood_iou"],row["pooled_flood_f1"]],
                    [lock[method]["oof_macro_iou"],lock[method]["oof_pooled_f1"]],atol=TOL,rtol=0)
    if split == "historical_test":
        audit = {r["method"]:r for r in read_csv(Path(config["prior_audit"])/"summary.csv")}
        for row in report["summary"]:
            if row["method"] in audit:
                keys = (*METRICS, "pooled_flood_precision", "pooled_flood_recall")
                np.testing.assert_allclose([row[k] for k in keys],
                    [float(audit[row["method"]][k]) for k in keys],atol=TOL,rtol=0)


def markdown(out, reports):
    lines = ["# Experiment 2 — OOF Source Reference", "",
        "Comparisons use four frozen operating points. +3.70 maximized event-macro Flood IoU on the original OOF grid. "
        "+1.50 maximized pooled Flood F1 on that grid. +0.95 is the historical validation-selected global bias. "
        "OOF values are descriptive and reuse the events that selected the OOF scalars. Validation uses the "
        "959-patch scope, which differs from the original global-bias calibration scope. Historical test on 2,000 patches "
        "is exploratory. Trade-off curve points outside the four locked biases are diagnostic only.", "",
        "Confidence intervals are paired percentile CIs from 10,000 whole-event draws within each split. "
        "Each draw sums confusion matrices before calculating pooled metrics. IoU std uses population std. "
        "Lower-tail IoU averages the lowest ceil(E/3) draws. When an event is omitted, all biases remain fixed and "
        "the lower-tail group size can change. Oracle-gap std uses absolute distance to the Event Oracle std, "
        "a diagnostic reference statistic. CIs are pointwise and use small event samples.", ""]
    for split, result in reports.items():
        lines += [f"## {split}", "",
            "| Method | mIoU | Flood F1 | Macro IoU | IoU std | Worst IoU | Lower-tail IoU | Precision | Recall |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in result["summary"]:
            vals = " | ".join(f"{100*r[k]:.2f}" for k in (*METRICS,"pooled_flood_precision","pooled_flood_recall"))
            lines.append(f"| {LABEL[r['method']]} | {vals} |")
        lines += ["", "Metric values are on the 0–100 scale.", ""]
        for ref in REFERENCES:
            lookup = {r["metric"]:r for r in result["paired_bootstrap"] if r["reference"]==ref}
            r = next(r for r in result["win_tie_loss"] if r["reference"]==ref)
            delta = next(r for r in result["pooled_confusion_deltas"] if r["reference"]==ref)
            effects = []
            for k in ("pooled_flood_f1","event_macro_flood_iou","worst_event_iou"):
                d = lookup[k]
                effects.append(f"{METRICS[k]} {d['delta_pp']:+.2f} pp [{d['ci_low_pp']:+.2f}, {d['ci_high_pp']:+.2f}]")
            lines.append(f"- vs {LABEL[ref]}: {', '.join(effects)}. W/T/L {r['win']}/{r['tie']}/{r['loss']}. "
                         f"Precision {delta['delta_precision_pp']:+.2f} pp and Recall {delta['delta_recall_pp']:+.2f} pp.")
            gaps = [g["gap_reduction_pct"] for g in result["event_oracle_gap"] if g["reference"]==ref]
            closure = " / ".join(f"{g:+.1f}%" if g is not None else "N/A" for g in gaps)
            lines.append(f"  Oracle gap closed for macro IoU, IoU std and worst-event IoU: {closure}.")
        if split == "historical_test":
            lines += ["", "### Sensitivity without event 411", ""]
            for ref in REFERENCES:
                d = {r["metric"]:r["delta_pp"] for r in result["leave_one_event_out"]
                     if r["reference"]==ref and r["omitted_event"]=="411"}
                lines.append(f"- vs {LABEL[ref]}: macro IoU {d['event_macro_flood_iou']:+.2f} pp, "
                             f"worst-event IoU {d['worst_event_iou']:+.2f} pp.")
        lines += ["", f"[All paired CIs]({split}/paired_bootstrap.csv) · "
                  f"[Oracle gap]({split}/event_oracle_gap.csv) · [Omit-one sensitivity]({split}/leave_one_event_out.csv)", ""]
    test = {(r["reference"],r["metric"]):r for r in reports["historical_test"]["paired_bootstrap"]}
    f = test[F1,"pooled_flood_f1"]
    w = test[F1,"worst_event_iou"]
    tail = test[F1,"lower_tail_iou"]
    lines += ["## What the historical comparison supports", "",
        f"Against the OOF F1 scalar, the OOF Source Reference shifts the operating point by {f['delta_pp']:+.2f} pp "
        f"in pooled Flood F1, {w['delta_pp']:+.2f} pp in worst-event IoU and {tail['delta_pp']:+.2f} pp in lower-tail IoU. "
        "Use the paired intervals to distinguish a point-estimate trade-off from changes sustained across the event resamples. "
        "The held-out historical results provide exploratory evidence for assessing threshold transfer across events. "
        "The archived thresholds remain fixed."]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threshold-results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    source, out = args.threshold_results.resolve(), args.out.resolve()
    config = load_json(source/"analysis_config.json")
    prior = Path(config["prior_audit"])/"audit_config.json"
    if config["locked"] != LOCKED or digest(prior) != config["prior_audit_config_sha256"]:
        raise ValueError("Mismatch in frozen scalar thresholds or prior historical audit")
    out.mkdir(parents=True,exist_ok=False)
    reports = {}
    for split, (n_events,n_patches) in SPLITS.items():
        path = source/split/"event_threshold_confusions.npz"
        identity = config["splits"][split]
        if (digest(path)!=identity["archive_sha256"] or identity["events"]!=n_events
                or identity["patches"]!=n_patches):
            raise ValueError(f"Changed threshold artifacts: {split}")
        with np.load(path,allow_pickle=False) as archive:
            events, grid, cms, oracle = (archive[k] for k in ("event_ids","grid","confusions","oracle_confusions"))
        if (len(set(events))!=n_events or cms.shape!=(n_events,len(grid),3,3)
                or oracle.shape!=(n_events,3,3) or np.any(np.diff(grid)<=0)
                or any(b not in grid for b in LOCKED.values())):
            raise ValueError(f"Invalid archive for {split}")
        np.testing.assert_array_equal(cms.sum(3),np.broadcast_to(oracle.sum(2)[:,None],cms.shape[:3]))
        mats = {m:cms[:,np.searchsorted(grid,b)] for m,b in LOCKED.items()} | {EVENT:oracle}
        result = compare(mats,events.tolist())
        verify_replayed_metrics(config, split, result)
        reports[split] = result
        split_out = out/split
        split_out.mkdir()
        for name,rows in result.items():
            save_csv(split_out/f"{name}.csv",rows)
        tradeoff(split_out,split,grid,cms)
    plot_ci(out,reports)
    markdown(out,reports)
    write_json(out/"analysis_config.json",dict(threshold_results=str(source),
        threshold_config_sha256=digest(source/"analysis_config.json"),bootstrap_iterations=ITERATIONS,
        bootstrap_seed=SEED,confidence_interval="95% percentile, paired event resampling",
        scope=config["scope"],locked=LOCKED,script_sha256=digest(Path(__file__))))
    print(f"Saved Experiment 2: {out}",flush=True)


if __name__ == "__main__":
    main()
