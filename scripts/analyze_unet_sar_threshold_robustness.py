#!/usr/bin/env python3
"""Event-level Spearman and prevalence-adjusted rank sensitivity."""
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, t

from analyze_unet_threshold_sensitivity import (
    SAR_FEATURES, SPLITS, digest, load_json, plt, read_csv, save_csv, save_figure, write_json,
)
from ccla_crossfit_cache import lookup, values


def statistics(x, y, prevalence=None):
    n = len(x)
    dof = n - (3 if prevalence is not None else 2)
    if dof <= 0:
        return None, None
    a, b = rankdata(x), rankdata(y)
    if prevalence is not None:
        p = rankdata(prevalence)
        if not np.ptp(p):
            return None, None
        design = np.column_stack([np.ones(n), p])
        a = a - design @ np.linalg.lstsq(design, a, rcond=None)[0]
        b = b - design @ np.linalg.lstsq(design, b, rcond=None)[0]
    a, b = a - a.mean(), b - b.mean()
    scale = np.linalg.norm(a)*np.linalg.norm(b)
    if scale <= 1e-10:
        return None, None
    rho = float(np.clip(a@b/scale,-1,1))
    p_value = 0. if abs(rho)==1 else float(2*t.sf(abs(rho)*np.sqrt(dof/(1-rho*rho)),dof))
    return rho, p_value


def fdr(rows):
    # BH within each split, correlation type and omission. Unestimable hypotheses remain blank.
    order = sorted((i for i,r in enumerate(rows) if r["p"] is not None),key=lambda i: rows[i]["p"])
    prev = 1.
    for j in range(len(order)-1,-1,-1):
        row = rows[order[j]]
        prev = min(prev, row["p"]*len(order)/(j+1))
        row["q_bh"] = prev


def input_medians(cfg, split, events):
    paths = cfg["splits"][split]["cache_manifests"]
    packs = []
    for path,h in paths.items():
        p = Path(path)
        if digest(p)!=h:
            raise ValueError(f"Changed cache manifest {p}")
        packs.append((p.parent,load_json(p)))
    index,_ = lookup(packs)
    if set(index)!={r["event_id"] for r in events}:
        raise ValueError(f"Cache event mismatch: {split}")
    for row in events:
        event = row["event_id"]
        sar, extra = (values(index,event,k) for k in ("sar","extra"))
        channels = (sar[:,0],extra[:,0],sar[:,1],sar[:,2],extra[:,0]-sar[:,0],extra[:,1])
        for name,x in zip(SAR_FEATURES,channels,strict=True):
            finite = x[np.isfinite(x)]
            row[f"input_{name}_median"] = float(np.median(finite)) if len(finite) else None


def correlations(events):
    types = {"prevalence":"requires_ground_truth", "ap":"requires_ground_truth"}
    types.update({f"{pop}_{k}_median": kind for k in SAR_FEATURES
        for pop,kind in (("input","label_free"),("all","annotation_mask_dependent"),
                         ("flood","requires_ground_truth"))})
    out = []
    for omitted in ("none",*(r["event_id"] for r in events)):
        for partial in (False,True):
            rows = []
            for feature,kind in types.items():
                if partial and feature=="prevalence":
                    continue
                data = [(r["oracle_bias"],r[feature],r["prevalence"]) for r in events
                        if r["event_id"]!=omitted and r["oracle_identifiable"] and r[feature] is not None]
                rho,p = statistics(*np.asarray(data).T[:2],
                    prevalence=np.array(data)[:,2] if partial else None) if data else (None,None)
                rows.append(dict(omitted_event=omitted,method="partial_spearman" if partial else "spearman",
                                 feature=feature,availability=kind,events=len(data),rho=rho,p=p,q_bh=None))
            fdr(rows)
            out.extend(rows)
    return out


def plot(out, split, rows):
    features = ["prevalence","ap","input_post_vv_db_median","input_delta_vv_db_median",
                "flood_post_vv_db_median","flood_delta_vv_db_median"]
    names = ["Prevalence","AP","All input-valid post-VV","All input-valid ΔVV",
             "Flood-only post-VV","Flood-only ΔVV"]
    fig,axes = plt.subplots(2,3,figsize=(11,5.6),layout="constrained")
    y = {r["event_id"]:r["oracle_bias"] for r in rows if r["oracle_identifiable"]}
    for ax,feature,name in zip(axes.flat,features,names,strict=True):
        for r in rows:
            e = r["event_id"]
            if e not in y or r[feature] is None:
                continue
            highlight = e in ("411","445")
            ax.scatter(r[feature],y[e],color="#d55e00" if highlight else "#0072b2",s=23)
            if highlight:
                ax.annotate(e,(r[feature],y[e]),xytext=(3,3),textcoords="offset points",fontsize=8)
        ax.set(xlabel=name,ylabel="Exact Event Oracle bias")
        ax.grid(alpha=.18)
    save_figure(fig,out,f"{split}_sar_scatter")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threshold-results",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a = p.parse_args()
    cfg_file = a.threshold_results/"analysis_config.json"
    cfg = load_json(cfg_file)
    a.out.mkdir(parents=True,exist_ok=False)
    lines = ["# Experiment 3 — SAR–threshold relationship", "",
        "Each split is analyzed separately. Partial Spearman correlates residual ranks after controlling for ranked "
        "ground-truth prevalence. Approximate two-sided t p-values and BH FDR within each split and correlation type "
        "are reported. Leave-one-event-out repeats ranking, adjustment and BH with the remaining events.", "",
        "Input-valid SAR features use no annotations and can be computed at deployment. Scored-pixel summaries use the "
        "annotation mask. Flood-only features, prevalence, AP and oracle bias require ground truth. Partial correlations "
        "are retrospective even when the SAR predictor is input-valid. With 27/6/10 events, estimates and approximate "
        "p-values are sensitive to individual events and dependence. Historical test is exploratory. No correlation "
        "establishes a causal physical mechanism.", ""]
    for split,(n,_) in SPLITS.items():
        source = a.threshold_results/split/"event_summary.csv"
        raw = read_csv(source)
        events = [{k: (v=="True" if k=="oracle_identifiable" else v if k=="event_id"
                      else float(v) if v else None) for k,v in r.items()} for r in raw]
        if len(events)!=n or len({r["event_id"] for r in events})!=n:
            raise ValueError(f"Unexpected scope: {split}")
        input_medians(cfg,split,events)
        rows = correlations(events)
        out = a.out/split
        out.mkdir()
        save_csv(out/"event_features.csv",events)
        save_csv(out/"correlations.csv",[r for r in rows if r["omitted_event"]=="none"])
        save_csv(out/"leave_one_event_out.csv",[r for r in rows if r["omitted_event"]!="none"])
        plot(out,split,events)
        lines += [f"## {split}", ""]
        for kind in ("spearman","partial_spearman"):
            complete = [r for r in rows if r["omitted_event"]=="none" and r["method"]==kind and r["q_bh"] is not None]
            hits = [r for r in complete if r["q_bh"]<=.05]
            strongest = max(complete,key=lambda r:abs(r["rho"]))
            lines.append(f"- {kind}: {len(hits)}/{len(complete)} associations at q ≤ .05. Largest |ρ|: "
                         f"{strongest['feature']} ρ={strongest['rho']:+.3f}, q={strongest['q_bh']:.3f}.")
            for key in ("input_delta_vv_db_median","flood_delta_vv_db_median"):
                effects = {r["omitted_event"]:r for r in rows if r["method"]==kind and r["feature"]==key}
                show = ["none","411","445"] if split=="historical_test" else ["none"]
                payload = ", ".join(f"{e}: ρ={effects[e]['rho']:+.3f}, q={effects[e]['q_bh']:.3f}"
                    for e in show if effects[e]["rho"] is not None)
                lines.append(f"  {key} ({payload}).")
        lines += ["",f"[Correlations]({split}/correlations.csv) · [Omit-one estimates]({split}/leave_one_event_out.csv)",""]
    (a.out/"REPORT.md").write_text("\n".join(lines)+"\n")
    write_json(a.out/"analysis_config.json",dict(threshold_results_sha256=digest(cfg_file),
        threshold_results=str(a.threshold_results.resolve()),fdr="BH per split/type/omission",
        scope=cfg["scope"],script_sha256=digest(Path(__file__))))


if __name__=="__main__":
    main()
