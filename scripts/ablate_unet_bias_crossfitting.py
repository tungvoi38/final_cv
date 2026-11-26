#!/usr/bin/env python3
"""Isolate source predictions and objective for locked scalar flood thresholds."""
import argparse
from pathlib import Path

import numpy as np
import torch

from analyze_unet_model_threshold_robustness import (
    cumulative_confusions, digest, ensemble_mean, evaluation_inputs, evaluation_loader,
    flipped_single, load_json, models, plt, save_csv, save_figure, score_array,
    scores, select_rows, split_rows, write_json,
)
from audit_unet_historical_test import METRICS
from bootstrap_unet_events import ITERATIONS, SEED, interval
from evaluate_unet_oof_source_reference import report_stats
from kuro_dataset import RawArchive
from train_unet import IGNORE
from unet_conservative_expected_iou import best_index

SOURCES = ("in_sample","validation","oof")
OBJECTIVES = ("pooled_f1","macro_iou")
REPEATS = 240
SUBSET_SIZE = 5


def threshold_assets(path):
    cfg = load_json(path/"analysis_config.json")
    source,test = (Path(cfg[k]) for k in ("oof_run","test_run"))
    tc = load_json(test/"frozen_config.json")
    members = tc["members"]
    oof = load_json(source/"frozen_config.json")
    if ([m["sha256"] for m in members]!=[m["sha256"] for m in oof["full_train"]]
            or any(m["epoch"]!=20 or m["weights_key"]!="model" for m in members)):
        raise ValueError("In-sample and historical benchmark model provenance differ")
    p = source/"fit/oof_patch_curves.npz"
    lock = load_json(test/"scalar_inputs/scalar_bias_lock.json")
    if digest(p)!=lock["oof_curves_sha256"]:
        raise ValueError("Original OOF bias grid differs from lock")
    with np.load(p,allow_pickle=False) as d:
        grid = d["biases"]
    return cfg,tc,grid


@torch.inference_mode()
def calibrate_insample(threshold, out):
    cfg,tc,grid = threshold_assets(threshold)
    oof_members = cfg["splits"]["oof_train"]["cache_manifests"]
    patch_ids = {}
    for path,sha in oof_members.items():
        p = Path(path)
        if digest(p)!=sha:
            raise ValueError(f"Changed OOF cache manifest: {p}")
        for row in load_json(p)["patches"]:
            key = row["patch_id"]
            if key in patch_ids:
                raise ValueError(f"Duplicate OOF patch {key}")
            patch_ids[key] = row["event_id"]
    archive = RawArchive(tc["raw"])
    rows = split_rows(archive.frame,"train")
    pairs = dict(zip(rows["tortilla:id"].astype(str),rows.actid.astype(str),strict=True))
    if pairs!=patch_ids or len(pairs)!=4000 or len(set(pairs.values()))!=27:
        raise ValueError("In-sample source must match exactly the 27 OOF events and 4,000 patches")
    rows = select_rows(rows,sorted(patch_ids,key=int))
    identification = dict(threshold_sha256=digest(threshold/"analysis_config.json"),
        sha256_by_seed={str(m["seed"]):m["sha256"] for m in tc["members"]},
        tta="identity/horizontal/vertical/both averaged per seed, then FP32 3-seed mean",
        split="train",patch_ids=rows["tortilla:id"].astype(str).tolist(),
        fit_or_tune_test_labels=False,statistic="exact 3×3 confusion over original OOF bias grid")
    if (out/"manifest.json").exists():
        prior = load_json(out/"manifest.json")
        if (any(prior[k]!=v for k,v in identification.items())
                or prior["confusion_sha256"]!=digest(out/"event_confusions.npz")):
            raise ValueError("Existing in-sample cache differs from protocol")
        print(f"Reuse completed in-sample event confusion cache: {out}",flush=True)
        return
    out.mkdir(parents=True,exist_ok=False)
    loader = evaluation_loader(archive,rows)
    nets = models(tc["members"],[42,43,44])
    events = sorted(set(pairs.values()),key=int)
    conf = {e:np.zeros((len(grid),3,3),dtype=np.int64) for e in events}
    patches = dict.fromkeys(events,0)
    for step,(batch,items) in enumerate(loader,1):
        with torch.autocast("cuda"):
            x = evaluation_inputs(batch)
            logits = [flipped_single(m,x).permute(0,2,3,1).cpu().numpy() for m in nets]
        for i,(e,y0) in enumerate(items):
            patches[e] += 1
            keep = y0!=IGNORE
            if not keep.any():
                continue
            z = ensemble_mean(*(v[i][keep] for v in logits))
            margin, other, y = score_array(z,y0[keep].astype(np.uint8,copy=False))
            conf[e] += cumulative_confusions(margin,other,y,grid)
        if step%100==0 or step==len(loader):
            print(f"In-sample frozen Flip TTA: {step}/{len(loader)} batches",flush=True)
    if sum(patches.values())!=4000:
        raise ValueError("Incomplete training patch traversal")
    file = out/"event_confusions.npz"
    np.savez_compressed(file,grid=grid,event_ids=events,confusions=np.stack([conf[e] for e in events]))
    write_json(out/"manifest.json",identification | dict(event_patch_counts=patches,confusion_sha256=digest(file)))
    print(f"Saved in-sample confusion cache: {out}",flush=True)


def load_threshold(path, cfg, split, grid):
    source = path/split/"event_threshold_confusions.npz"
    if digest(source)!=cfg["splits"][split]["archive_sha256"]:
        raise ValueError(f"Changed threshold cache: {source}")
    with np.load(source,allow_pickle=False) as d:
        index = np.searchsorted(d["grid"],grid)
        if not np.array_equal(d["grid"][index],grid):
            raise ValueError("Candidate bias grid is not in the saved threshold grid")
        return d["event_ids"],d["confusions"][:,index]


def objective_values(cm):
    tp = cm[:,:,2,2]
    predicted,actual = cm[:,:,:,2].sum(2),cm[:,:,2,:].sum(2)
    return {"pooled_f1":2*tp.sum(0)/np.maximum(1,(predicted+actual).sum(0)),
            "macro_iou":(tp/np.maximum(1,predicted+actual-tp)).mean(0)}


def choices(cm, grid):
    return {o:dict(bias=float(grid[i]),score=float(curve[i]),index=int(i))
            for o,curve in objective_values(cm).items() for i in [best_index(curve,grid)]}


def select_sources(events, source_cms, grid):
    full = [dict(source=s,objective=o,events=len(events[s]),**value)
            for s in SOURCES for o,value in choices(source_cms[s],grid).items()]
    rng = np.random.default_rng(SEED)
    train = events["oof"]
    val = events["validation"]
    if len(val)!=6 or REPEATS%len(val):
        raise ValueError("Validation omission cycle must cover each event equally")
    omitted = np.tile(val,REPEATS//len(val))
    rng.shuffle(omitted)
    train_index = {s:{e:i for i,e in enumerate(events[s])} for s in ("in_sample","oof")}
    subsampled = []
    for draw,e0 in enumerate(omitted):
        sample = sorted(rng.choice(train,SUBSET_SIZE,replace=False),key=int)
        for s in SOURCES:
            if s=="validation":
                indices = np.flatnonzero(val!=e0)
                member_events = val[indices].tolist()
            else:
                indices = [train_index[s][e] for e in sample]
                member_events = sample
            for o,value in choices(source_cms[s][indices],grid).items():
                subsampled.append(dict(repeat=draw,source=s,objective=o,events=SUBSET_SIZE,
                    event_ids="|".join(member_events),**value))
    return full,subsampled


def paired(mats, events):
    n = len(events)
    omission = np.vstack([np.ones(n,dtype=int),1-np.eye(n,dtype=int)])
    draws = np.random.default_rng(SEED).integers(n,size=(ITERATIONS,n))
    weights = np.eye(n,dtype=int)[draws].sum(1)
    point, boot = report_stats(mats,omission),report_stats(mats,weights)
    comparisons = [
        *((f"oof:{o}",f"{r}:{o}") for o in OBJECTIVES for r in ("in_sample","validation")),
        *((f"validation:{o}",f"in_sample:{o}") for o in OBJECTIVES),
        *((f"{s}:macro_iou",f"{s}:pooled_f1") for s in SOURCES),
    ]
    cis, loo, event = [],[],[]
    for method,ref in comparisons:
        for metric in (*METRICS,"pooled_flood_precision","pooled_flood_recall"):
            vals = 100*(point[method][metric]-point[ref][metric])
            lo,hi,k = interval(100*(boot[method][metric]-boot[ref][metric]))
            cis.append(dict(method=method,reference=ref,metric=metric,delta_pp=vals[0],
                            ci_low_pp=lo,ci_high_pp=hi,iterations=k))
            for i,e in enumerate(("none",*events)):
                loo.append(dict(method=method,reference=ref,metric=metric,omitted_event=e,
                    delta_pp=vals[i],change_vs_all_pp=vals[i]-vals[0]))
        for e,a,b in zip(events,mats[method],mats[ref],strict=True):
            x,y = scores(a),scores(b)
            event.append(dict(method=method,reference=ref,event_id=e,
                delta_flood_iou_pp=100*(x["flood_iou"]-y["flood_iou"]),
                delta_precision_pp=100*(x["flood_precision"]-y["flood_precision"]),
                delta_recall_pp=100*(x["flood_recall"]-y["flood_recall"])))
    return point,cis,loo,event


def matched_bootstrap(subs, test_cms, events):
    # Keep the observed calibration draws fixed while jointly resampling test events.
    n = len(events)
    draws = np.random.default_rng(SEED).integers(n,size=(ITERATIONS,n))
    weights = np.vstack([np.ones(n,dtype=int),np.eye(n,dtype=int)[draws].sum(1)])
    averages = {}
    for source in SOURCES:
        for objective in OBJECTIVES:
            idx, counts = np.unique([r["index"] for r in subs if r["source"]==source and
                                      r["objective"]==objective],return_counts=True)
            d = report_stats({str(i):test_cms[:,i] for i in idx},weights)
            averages[source,objective] = {
                k: (counts @ np.stack([d[str(i)][k] for i in idx]))/counts.sum() for k in METRICS}
    rows = []
    for objective in OBJECTIVES:
        for source in ("in_sample","validation"):
            for k in METRICS:
                delta = 100*(averages["oof",objective][k]-averages[source,objective][k])
                lo,hi,iterations = interval(delta[1:])
                rows.append(dict(objective=objective,method="oof",reference=source,metric=k,
                    delta_pp=delta[0],ci_low_pp=lo,ci_high_pp=hi,iterations=iterations))
    return rows


def tradeoff(out, chosen, subs):
    fig,axes = plt.subplots(1,3,figsize=(12.4,4),layout="constrained",sharey=True)
    palette = {"pooled_f1":"#0072b2","macro_iou":"#d55e00"}
    for ax,s in zip(axes,SOURCES,strict=True):
        for o in OBJECTIVES:
            rows = [r for r in subs if r["source"]==s and r["objective"]==o]
            color = palette[o]
            ax.scatter([100*r["pooled_flood_f1"] for r in rows],
                       [100*r["event_macro_flood_iou"] for r in rows],alpha=.20,s=15,color=color)
            row, = [r for r in chosen if r["source"]==s and r["objective"]==o]
            ax.scatter(100*row["pooled_flood_f1"],100*row["event_macro_flood_iou"],
                       marker="*",s=95,color=color,label=o.replace("_"," "))
        ax.set(xlabel="Historical test pooled Flood F1 (%)",title=s.replace("_"," "))
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Historical test macro Flood IoU (%)")
    axes[0].legend(frameon=False,fontsize=8)
    save_figure(fig,out,"crossfit_tradeoff")


def evaluate(threshold, insample, out):
    cfg,tc,grid = threshold_assets(threshold)
    file = insample/"event_confusions.npz"
    manifest = load_json(insample/"manifest.json")
    if (manifest["threshold_sha256"]!=digest(threshold/"analysis_config.json")
            or manifest["confusion_sha256"]!=digest(file) or manifest["split"]!="train"
            or manifest["fit_or_tune_test_labels"] is not False
            or manifest["sha256_by_seed"]!={str(m["seed"]):m["sha256"] for m in tc["members"]}):
        raise ValueError("Unexpected in-sample cache provenance")
    with np.load(file,allow_pickle=False) as d:
        np.testing.assert_array_equal(d["grid"],grid)
        train_events, train_cms = d["event_ids"],d["confusions"]
    oof_events,oof_cms = load_threshold(threshold,cfg,"oof_train",grid)
    val_events,val_cms = load_threshold(threshold,cfg,"validation",grid)
    np.testing.assert_array_equal(train_events,oof_events)
    np.testing.assert_array_equal(train_cms.sum(3),oof_cms.sum(3))
    events = {"in_sample":train_events,"oof":oof_events,"validation":val_events}
    full, subs = select_sources(events,{"in_sample":train_cms,"oof":oof_cms,"validation":val_cms},grid)
    np.testing.assert_array_equal([r["bias"] for r in full if r["source"]=="oof"],[1.50,3.70])
    out.mkdir(parents=True,exist_ok=False)
    # Both full-pool and subsampled selections are serialized before reading test scores.
    config = dict(scope=cfg["scope"],selection="source-only",full=full,subsets=subs,seed=SEED,
                  repeats=REPEATS,events_per_repeat=SUBSET_SIZE,candidate_grid=grid.tolist(),
                  threshold_sha256=digest(threshold/"analysis_config.json"),
                  in_sample_manifest_sha256=digest(insample/"manifest.json"))
    write_json(out/"bias_lock.json",config)
    lock = load_json(out/"bias_lock.json")
    test_ids,test_cms = load_threshold(threshold,cfg,"historical_test",grid)
    mats = {f"{r['source']}:{r['objective']}":test_cms[:,r["index"]] for r in lock["full"]}
    whole,cis,loo,ev = paired(mats,test_ids.tolist())
    by_grid = report_stats({str(i):test_cms[:,i] for i in {r["index"] for r in lock["subsets"]}},
                           np.ones((1,len(test_ids)),dtype=int))
    chosen = [r|{k:float(v[0]) for k,v in whole[f"{r['source']}:{r['objective']}"].items()} for r in full]
    subs = [r|{k:float(v[0]) for k,v in by_grid[str(r["index"])].items()} for r in subs]
    matched = []
    for s in SOURCES:
        for o in OBJECTIVES:
            rows = [r for r in subs if r["source"]==s and r["objective"]==o]
            for key in ("bias",*METRICS):
                x = [r[key] for r in rows]
                q05,q50,q95 = np.quantile(x,[.05,.5,.95])
                matched.append(dict(source=s,objective=o,variable=key,mean=np.mean(x),std=np.std(x),
                                    q05=q05,median=q50,q95=q95,unique_biases=len({r["bias"] for r in rows})))
    matched_pairs = []
    for o in OBJECTIVES:
        for ref in ("in_sample","validation"):
            for metric in ("bias",*METRICS):
                x = [100*(a[metric]-b[metric]) if metric!="bias" else a[metric]-b[metric]
                     for a,b in zip((r for r in subs if r["objective"]==o and r["source"]=="oof"),
                                    (r for r in subs if r["objective"]==o and r["source"]==ref),strict=True)]
                lo,med,hi = np.quantile(x,[.05,.5,.95])
                matched_pairs.append(dict(objective=o,method="oof",reference=ref,metric=metric,
                    paired_train_events=ref=="in_sample",mean_delta=np.mean(x),median_delta=med,q05=lo,q95=hi))
    for label,rows in (("locked_biases_test",chosen),("paired_bootstrap",cis),("leave_one_event_out",loo),
                       ("event_differences",ev),("matched_subsamples",subs),("matched_summary",matched),
                       ("matched_differences",matched_pairs),
                       ("matched_paired_bootstrap",matched_bootstrap(subs,test_cms,test_ids))):
        save_csv(out/f"{label}.csv",rows)
    tradeoff(out,chosen,subs)
    report(out,chosen,cis,matched)


def report(out, chosen, cis, matched):
    lines = ["# Experiment 2 — Cross-fitting and objective ablation", "",
        "Bias was locked from source-only confusion curves on the original OOF candidate grid. In-sample and OOF use "
        "the same 4,000 patches across 27 events. OOF checkpoints train on 18 source events per fold with footprint "
        "purging. The in-sample full model trains across 27 events. Thus OOF vs in-sample changes both data exposure "
        "and training coverage. Validation uses the 959 footprint-filtered patches of six events. All locked scalars "
        "are applied to the same full-train SMA epoch-20 EoA + Flip TTA cache on the 10 historical test events.", "",
        "The repeated analysis uses 240 five-event subsamples. OOF and in-sample share the exact sampled training "
        "event IDs. Validation cycles through each five-of-six subset equally with shuffled order. Equal event counts "
        "leave patch counts, prevalence and event identities free to differ across source pools. Subsample percentiles "
        "describe calibration variability and the repeated results are dependent. Test CIs use 10,000 paired event "
        "resamples. They cover the six full-source operating points and the mean matched-subsample effect. The "
        "matched intervals condition on the fixed calibration subset draws.", "",
        "| Source | Objective | Bias | mIoU (%) | Flood F1 (%) | Macro IoU (%) | IoU std (pp) | Worst IoU (%) | Lower-tail IoU (%) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in chosen:
        vals = " | ".join(f"{100*r[k]:.2f}" for k in METRICS)
        lines.append(f"| {r['source']} | {r['objective']} | {r['bias']:+.2f} | {vals} |")
    lines += ["", "## Matched five-event source subsampling", "",
              "| Source | Objective | Bias median [5%, 95%] | Unique biases |",
              "|---|---|---:|---:|"]
    for r in matched:
        if r["variable"]=="bias":
            lines.append(f"| {r['source']} | {r['objective']} | {r['median']:+.2f} "
                         f"[{r['q05']:+.2f}, {r['q95']:+.2f}] | {r['unique_biases']} |")
    lines += ["", "## Selected paired contrasts", ""]
    for r in cis:
        if r["method"].startswith("oof:") and r["metric"] in ("pooled_flood_f1","event_macro_flood_iou","worst_event_iou"):
            lines.append(f"- {r['method']} vs {r['reference']}, {r['metric']}: {r['delta_pp']:+.2f} pp "
                         f"[{r['ci_low_pp']:+.2f}, {r['ci_high_pp']:+.2f}].")
    lines += ["", "Overlaps with zero describe the uncertainty of these pointwise intervals. This historical-test "
        "exploratory evaluation keeps the candidate grid, objective definitions and choices fixed by source data. "
        "It does not identify a causal benefit of cross-fitting independently of training coverage and domain shift.", "",
        "[All CIs](paired_bootstrap.csv) · [Leave-one-event-out](leave_one_event_out.csv) · "
        "[Matched paired differences](matched_differences.csv) · "
              "[Matched conditional CIs](matched_paired_bootstrap.csv)"]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("cache-in-sample","evaluate"))
    p.add_argument("--threshold-results",type=Path,required=True)
    p.add_argument("--in-sample",type=Path,required=True)
    p.add_argument("--out",type=Path)
    a = p.parse_args()
    if a.stage=="cache-in-sample":
        calibrate_insample(a.threshold_results,a.in_sample)
    else:
        if a.out is None:
            p.error("--out is required for evaluate")
        evaluate(a.threshold_results,a.in_sample,a.out)


if __name__=="__main__":
    main()
