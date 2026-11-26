#!/usr/bin/env python3
"""Audit frozen bias adaptation from saved validation predictions and results only."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adapt_unet_event_bias import digest, flood_margin, load_json, open_cache, reference, write_json
from evaluate_unet_eoa_diagnostics import rank_event
from unet_bias_objectives import load_views

ADAPT = ("bbse", "em", "otsu", "entropy", "tta_consistency")
ORACLE = "oracle_DIAGNOSTIC_ONLY"
COMPARISONS = ("no_bias", "global_bias", *ADAPT, ORACLE)
LABELS = dict(zip(COMPARISONS, ("No bias", "Global +0.95", "BBSE", "EM", "Otsu", "Entropy",
                                "TTA consistency", "Oracle (diagnostic)"), strict=True))
COLORS = dict(zip(COMPARISONS, ("#636a70","#6b4ca5","#267847","#b4553a","#9b8e79","#267ca5",
                                "#c28922","#98264c"), strict=True))
OBJ = ("entropy", "tta_consistency")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def checked(actual, expected, description, tolerance=1e-12):
    if not np.allclose(actual, expected, atol=tolerance, rtol=0):
        raise ValueError(f"Mismatch for {description}: {actual} != {expected}")


def archived_inputs(results, cache, views_dir):
    recipe, selection, grid, expected = reference()
    config = load_json(results/"evaluation_config.json")
    bias_lock_path = results/"bias_lock.json"
    lock = load_json(bias_lock_path)
    if (config["split"] != "validation"
            or digest(bias_lock_path) != config["bias_lock_sha256"]
            or digest(cache/"manifest.json") != lock["cache_manifest_sha256"]
            or digest(views_dir/"manifest.json") != lock["view_cache_manifest_sha256"]
            or lock["checkpoint_selection_sha256"] != recipe["selection_sha256"]):
        raise ValueError("Evaluation and cache provenance do not match")
    inputs, labels = open_cache(cache, recipe, selection, expected)
    load_views(views_dir, cache, recipe, inputs)
    events = sorted(inputs,key=int)
    metrics, objectives, thresholds = {}, {}, {}
    for r in read_csv(results/"event_metrics.csv"):
        event, name = r["event_id"], r["method"]
        key = event,name
        if key in metrics:
            raise ValueError(f"Duplicate saved method {key}")
        row = {k:float(r[k]) if r[k] else None for k in (
            "bias", "flood_iou", "flood_f1", "flood_ap", "flood_precision", "flood_recall",
            "actual_flood_prevalence", "estimated_flood_prevalence", "predicted_prevalence_input_valid",
            "predicted_prevalence_scored", "evaluation_pixels", "input_valid_pixels",
        )}
        row["saved_status"] = r["status"]
        if name in ADAPT:
            m = lock["events"][event]["methods"][name]
            checked(row["bias"],m["bias"],str(key))
            checked(row["predicted_prevalence_input_valid"],m["predicted_prevalence_input_valid"],str(key))
            prior = m["prior"][2]
            if prior is not None:
                checked(row["estimated_flood_prevalence"],prior,str(key))
            elif row["estimated_flood_prevalence"] is not None:
                raise ValueError(f"Unexpected prevalence estimate {key}")
        metrics[key] = row
    for r in read_csv(results/"objective_curves.csv"):
        key = r["event_id"],r["method"]
        objectives.setdefault(key,[]).append(dict(
            bias=float(r["bias"]), objective=float(r["objective"]),
            input_prevalence=float(r["flood_prevalence"]), feasible=r["feasible"]=="True",
            selected=r["selected"]=="True"))
    for r in read_csv(results/"otsu_thresholds.csv"):
        thresholds[r["event_id"]]=r
    if (set(metrics)!={(event,name) for event in events for name in COMPARISONS}
            or set(objectives)!={(event,name) for event in events for name in OBJ}
            or set(thresholds)!=set(events)):
        raise ValueError("Expected each saved method and curve for six validation events")
    return events,inputs,labels,metrics,objectives,grid,expected,lock,thresholds


def operating_points(ranked, biases, n):
    score, positive, totals = ranked
    position = np.searchsorted(score.astype(np.float64), -np.asarray(biases), side="right")
    tp = np.r_[np.cumsum(positive[::-1])[::-1],0][position]
    predicted = np.r_[np.cumsum(totals[::-1])[::-1],0][position]
    fp, fn = predicted-tp, positive.sum()-tp
    iou = tp/np.maximum(tp+fp+fn,1)
    return tp,fp,fn,iou,predicted/n


def audit_event(event,inputs,labels,metrics,objectives,archived_biases,archive,lock,threshold):
    eval_mask = labels != 255
    y = labels[eval_mask]
    ranked = rank_event([(flood_margin(inputs)[eval_mask],y==2)])
    biases = sorted({*map(float,archived_biases), *(metrics[event,name]["bias"] for name in COMPARISONS)})
    positions = {b:i for i,b in enumerate(biases)}
    tp,fp,fn,iou,prevalence = operating_points(ranked,biases,len(y))
    archive_tp = archive[:,2,2]
    archive_iou = archive_tp/np.maximum(archive[:,2,:].sum(-1)+archive[:,:,2].sum(-1)-archive_tp,1)
    checked(iou[[positions[b] for b in archived_biases]],archive_iou,f"{event} archived IoU curve")
    actual = float(np.mean(y==2))
    oracle = metrics[event,ORACLE]
    global_row = metrics[event,"global_bias"]
    detailed = []
    for name in COMPARISONS:
        row = metrics[event,name]
        b = row["bias"]
        i = positions[b]
        checked([iou[i],prevalence[i],actual], [row["flood_iou"],row["predicted_prevalence_scored"],
                                               row["actual_flood_prevalence"]],f"{event}/{name}")
        checked([len(y),len(inputs)], [row["evaluation_pixels"],row["input_valid_pixels"]],f"{event} coverage")
        estimated = row["estimated_flood_prevalence"]
        record = dict(event_id=event,method=name,**row,oracle_bias=oracle["bias"],
                      bias_minus_oracle=b-oracle["bias"],oracle_flood_iou=oracle["flood_iou"],
                      oracle_gap_pp=100*(oracle["flood_iou"]-iou[i]),
                      gain_vs_global_pp=100*(iou[i]-global_row["flood_iou"]),
                      oracle_predicted_prevalence_scored=oracle["predicted_prevalence_scored"],
                      estimated_minus_actual_pp=100*(estimated-actual) if estimated is not None else None,
                      input_predicted_minus_estimate_pp=100*(row["predicted_prevalence_input_valid"]-estimated)
                      if estimated is not None else None,
                      scored_predicted_minus_actual_pp=100*(prevalence[i]-actual),
                      scored_predicted_minus_oracle_pp=100*(prevalence[i]-oracle["predicted_prevalence_scored"]),
                      tp=int(tp[i]),fp=int(fp[i]),fn=int(fn[i]))
        if name in ("bbse","em"):
            d=lock["methods"][name]["diagnostics"]
            record.update({f"estimator_{key}":v for key,v in d.items()})
        if name=="otsu":
            record.update(otsu_threshold_db=threshold["threshold_db"],
                          otsu_rejected_candidate_prevalence=threshold["candidate_prevalence"],
                          otsu_excluded_stable_water_pixels=threshold["excluded_stable_water_pixels"])
        detailed.append(record)

    points=[dict(event_id=event,bias=b,flood_iou=float(iou[i]),predicted_prevalence_scored=float(prevalence[i]),
                 tp=int(tp[i]),fp=int(fp[i]),fn=int(fn[i]),
                 methods_at_bias=" | ".join(name for name in COMPARISONS if metrics[event,name]["bias"]==b))
            for i,b in enumerate(biases)]
    diagnostics, objective_points = [], []
    for name in OBJ:
        curve = sorted(objectives[event,name], key=lambda item:item["bias"])
        if len(curve)!=121 or any(r["bias"] not in positions for r in curve):
            raise ValueError(f"Unexpected objective grid: {event}/{name}")
        selected, = [r for r in curve if r["selected"]]
        checked(selected["bias"],metrics[event,name]["bias"],f"{event}/{name} selected")
        checked(selected["objective"],lock["methods"][name]["diagnostics"]["objective_selected"],
                f"{event}/{name} objective")
        checked(selected["input_prevalence"],metrics[event,name]["predicted_prevalence_input_valid"],
                f"{event}/{name} objective prevalence")
        lo,hi = curve[0]["bias"], curve[-1]["bias"]
        objective_iou = {r["bias"]: iou[positions[r["bias"]]] for r in curve}
        feasible = [r for r in curve if r["feasible"]]
        feasible_biases = [r["bias"] for r in feasible]
        full_min = min(curve,key=lambda r:(round(r["objective"],12),abs(r["bias"]),r["bias"]))
        grid_best = max(objective_iou.values())
        allowed_best = max((objective_iou[b] for b in feasible_biases), default=None)
        chosen = metrics[event,name]["flood_iou"]
        oracle_iou = oracle["flood_iou"]
        losses = (100*(oracle_iou-grid_best),100*(grid_best-allowed_best),100*(allowed_best-chosen)) if (
            allowed_best is not None and selected["feasible"]) else (100*(oracle_iou-grid_best),None,None)
        if losses[1] is not None:
            checked(sum(losses),100*(oracle_iou-chosen),f"{event}/{name} decomposition")
        diagnostics.append(dict(event_id=event,method=name,selected_bias=selected["bias"],
            oracle_bias=oracle["bias"],oracle_outside_objective_grid=not lo<=oracle["bias"]<=hi,
            selected_at_grid_boundary=selected["bias"] in (lo,hi),
            selected_at_feasible_lower_bound=bool(feasible_biases and selected["bias"]==feasible_biases[0]),
            selected_at_feasible_upper_bound=bool(feasible_biases and selected["bias"]==feasible_biases[-1]),
            feasible_grid_min=feasible_biases[0] if feasible_biases else None,
            feasible_grid_max=feasible_biases[-1] if feasible_biases else None,
            unconstrained_objective_bias=full_min["bias"],unconstrained_at_boundary=full_min["bias"] in (lo,hi),
            unconstrained_is_feasible=full_min["feasible"],unconstrained_input_prevalence=full_min["input_prevalence"],
            objective_selected=selected["objective"],objective_unconstrained=full_min["objective"],
            selected_flood_iou=chosen,oracle_flood_iou=oracle_iou,best_grid_flood_iou_DIAGNOSTIC_ONLY=grid_best,
            best_feasible_grid_flood_iou_DIAGNOSTIC_ONLY=allowed_best,
            search_grid_gap_pp=losses[0],prevalence_constraint_gap_pp=losses[1],objective_choice_gap_pp=losses[2],
            total_oracle_gap_pp=100*(oracle_iou-chosen)))
        objective_points.extend(dict(event_id=event,method=name,**r,flood_iou=float(objective_iou[r["bias"]]))
                                for r in curve)
    best = max((metrics[event,name]["flood_iou"],-i,name) for i,name in enumerate(ADAPT))
    headroom=dict(event_id=event,actual_flood_prevalence=actual,event_flood_ap=oracle["flood_ap"],
                  oracle_bias=oracle["bias"],oracle_flood_iou=oracle["flood_iou"],
                  global_flood_iou=global_row["flood_iou"],gap_above_global_pp=100*(oracle["flood_iou"]-global_row["flood_iou"]),
                  best_adaptation_DIAGNOSTIC_ONLY=best[2],best_adaptation_iou_DIAGNOSTIC_ONLY=best[0],
                  gap_above_best_adaptation_pp=100*(oracle["flood_iou"]-best[0]))
    return detailed,points,diagnostics,objective_points,headroom


def summarize_failures(details):
    def mean(values):
        return float(np.mean(values)) if values else None

    rows=[]
    for name in COMPARISONS:
        group=[r for r in details if r["method"]==name]
        priors=[r for r in group if r["estimated_flood_prevalence"] is not None]
        tp,fp,fn=(sum(int(r[k]) for r in group) for k in ("tp","fp","fn"))
        rows.append(dict(method=name,n_events=len(group),n_prevalence_estimates=len(priors),
                         n_events_better_than_global=sum(float(r["gain_vs_global_pp"])>1e-9 for r in group),
                         mean_absolute_bias_error=mean([abs(float(r["bias_minus_oracle"])) for r in group]),
                         mean_absolute_estimate_error_pp=mean([abs(float(r["estimated_minus_actual_pp"])) for r in priors]),
                         mean_absolute_input_match_error_pp=mean([abs(float(r["input_predicted_minus_estimate_pp"])) for r in priors]),
                         mean_absolute_scored_prediction_error_pp=mean([abs(float(r["scored_predicted_minus_actual_pp"])) for r in group]),
                         pooled_flood_iou=tp/max(tp+fp+fn,1),pooled_flood_f1=2*tp/max(2*tp+fp+fn,1),
                         event_macro_flood_iou=mean([float(r["flood_iou"]) for r in group]),
                         mean_oracle_gap_pp=mean([float(r["oracle_gap_pp"]) for r in group])))
    return rows


def plot_bias(out, events, details, points):
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout="constrained")
    for ax,event in zip(axes.flat,events,strict=True):
        curve=[r for r in points if r["event_id"]==event]
        ax.plot([r["bias"] for r in curve],[100*r["flood_iou"] for r in curve],color=".25",lw=1.2)
        rows=[r for r in details if r["event_id"]==event]
        for r in rows:
            name=r["method"]
            ax.scatter(r["bias"],100*r["flood_iou"],s=45 if name==ORACLE else 26,
                       marker={"no_bias":"+","global_bias":"s","otsu":"x","entropy":"^",
                               "tta_consistency":"v",ORACLE:"*"}.get(name,"o"),
                       color=COLORS[name],label=LABELS[name],zorder=4,alpha=.85)
        ax.set(title=f"Event {event}",xlabel="Flood logit bias",ylabel="Flood IoU (%)")
        ax.grid(alpha=.18)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=4,fontsize=8)
    fig.suptitle("Validation exploratory · observed IoU as a function of flood bias")
    fig.savefig(out/"flood_iou_vs_bias.png",dpi=180)
    plt.close(fig)


def plot_prevalence(out, events, details):
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout="constrained")
    for ax,event in zip(axes.flat,events,strict=True):
        group=[r for r in details if r["event_id"]==event and r["method"] in ("global_bias",*ADAPT)]
        x=np.arange(len(group))
        ax.scatter(x,[100*r["predicted_prevalence_scored"] for r in group],marker="o",label="Predicted: scored")
        ax.scatter(x,[100*r["predicted_prevalence_input_valid"] for r in group],marker="x",label="Predicted: input-valid")
        estimate=[(i,100*r["estimated_flood_prevalence"]) for i,r in enumerate(group)
                  if r["estimated_flood_prevalence"] is not None]
        ax.scatter([v[0] for v in estimate],[v[1] for v in estimate],marker="^",color="#b4553a",label="Estimated prior")
        ax.axhline(100*group[0]["actual_flood_prevalence"],color=".25",label="Ground truth")
        ax.axhline(100*group[0]["oracle_predicted_prevalence_scored"],color=COLORS[ORACLE],linestyle="--",label="Oracle predicted")
        ax.set(xticks=x,xticklabels=["Global","BBSE","EM","Otsu","Entropy","TTA"],
               title=f"Event {event}",ylabel="Flood prevalence (%)")
        ax.grid(axis="y",alpha=.2)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=5,fontsize=8)
    fig.suptitle("Validation exploratory · prior estimates only where defined")
    fig.savefig(out/"event_prevalence.png",dpi=180)
    plt.close(fig)


def plot_objectives(out,events,points,diagnostics):
    for method in OBJ:
        fig,axes=plt.subplots(2,3,figsize=(14,7),sharex=True,layout="constrained")
        for ax,event in zip(axes.flat,events,strict=True):
            group=[r for r in points if (r["event_id"],r["method"])==(event,method)]
            d=next(r for r in diagnostics if (r["event_id"],r["method"])==(event,method))
            x=[r["bias"] for r in group]
            secondary=ax.twinx()
            secondary.plot(x,[100*r["flood_iou"] for r in group],color="#b4553a",lw=1,label="Flood IoU")
            secondary.set_ylabel("Flood IoU (%)",color="#b4553a")
            ax.plot(x,[r["objective"] for r in group],color="#267ca5",label="Unlabeled objective")
            if d["feasible_grid_min"] is not None:
                ax.axvspan(d["feasible_grid_min"],d["feasible_grid_max"],color=".75",alpha=.3,label="Feasible range")
            ax.axvline(d["selected_bias"],color="#267ca5",ls="--",label="Applied")
            if not d["oracle_outside_objective_grid"]:
                ax.axvline(d["oracle_bias"],color=COLORS[ORACLE],ls=":",label="Oracle")
            title=f"Event {event}"
            if d["oracle_outside_objective_grid"]:
                title+=f" · oracle {d['oracle_bias']:+.2f} off grid"
            ax.set(title=title,ylabel="Objective (nats)")
        axes[0,0].legend(fontsize=7)
        fig.supxlabel("Flood logit bias")
        fig.suptitle(f"{LABELS[method]} · validation exploratory and oracle diagnostics")
        fig.savefig(out/f"{method}_failure_curves.png",dpi=180)
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",required=True,type=Path)
    parser.add_argument("--cache",required=True,type=Path)
    parser.add_argument("--views",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    if args.out.exists():
        parser.error("--out must be a new directory")
    events,inputs,labels,metrics,objectives,grid,expected,lock,thresholds=archived_inputs(
        args.results,args.cache,args.views)
    details,points,diagnostics,objective_points,headroom=[],[],[],[],[]
    for event in events:
        d,p,o,op,h=audit_event(event,inputs[event],labels[event],metrics,objectives,grid,
                               expected[event],lock["events"][event],thresholds[event])
        details.extend(d)
        points.extend(p)
        diagnostics.extend(o)
        objective_points.extend(op)
        headroom.append(h)
        print(f"Audited saved validation event {event}",flush=True)
    args.out.mkdir(parents=True,exist_ok=False)
    save_csv(args.out/"bias_prevalence_by_event.csv",details)
    save_csv(args.out/"method_failure_summary.csv",summarize_failures(details))
    save_csv(args.out/"flood_iou_bias_curve.csv",points)
    save_csv(args.out/"objective_failure.csv",diagnostics)
    save_csv(args.out/"objective_curves_with_iou.csv",objective_points)
    save_csv(args.out/"oracle_headroom.csv",sorted(headroom,key=lambda r:-r["gap_above_global_pp"]))
    plot_bias(args.out,events,details,points)
    plot_prevalence(args.out,events,details)
    plot_objectives(args.out,events,objective_points,diagnostics)
    write_json(args.out/"config.json",dict(
        scope="validation only; exploratory and post-hoc diagnostics",
        frozen_bias_lock_sha256=digest(args.results/"bias_lock.json"),
        mean_cache_manifest_sha256=digest(args.cache/"manifest.json"),
        view_cache_manifest_sha256=digest(args.views/"manifest.json"),
        saved_results=str(args.results.resolve()),events=events,
        curve="archived validation grid plus exact operating points, computed from saved logits and labels",
        objective_decomposition="exact oracle minus best grid IoU; best grid minus best feasible IoU; best feasible minus applied IoU",
        eligibility="saved unlabeled feasibility flags; label-based diagnostics are not selection rules",
        prevalence_domains="estimated and matching predictions use input-valid; actual and scored prediction exclude annotation IGNORE"))
    print(f"Failure audit saved to {args.out}",flush=True)


if __name__=="__main__":
    main()
