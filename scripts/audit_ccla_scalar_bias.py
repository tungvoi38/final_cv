#!/usr/bin/env python3
"""Compare locked CCLA to scalar biases selected on the same frozen OOF predictions."""
import argparse
from pathlib import Path

import numpy as np

from adapt_unet_event_bias import IGNORE, digest, flood_margin, load_json, scores, write_json
from adapt_unet_iw_iou import CHUNK
from audit_unet_bias_failures import checked, read_csv, save_csv
from bootstrap_unet_events import ITERATIONS, SEED, interval
from ccla_crossfit_analysis import EVENT, GLOBAL, IW_MARGIN, IW_SAR, MARGIN, NOBIAS, PATCH, REF
from ccla_crossfit_cache import lookup, read_manifest, values
from diagnose_unet_spatial_oracle import predicted_confusion
from unet_conservative_expected_iou import best_index

F1 = "oof_pooled_f1_scalar"
METHODS = (NOBIAS,GLOBAL,REF,F1,IW_MARGIN,IW_SAR,MARGIN,EVENT,PATCH)
PAIRS = [(MARGIN,m) for m in METHODS if m != MARGIN] + [(F1,REF),(F1,GLOBAL)]
TOL = 1e-12


def lock_oof(run, out):
    plan_path = run/"frozen_config.json"
    plan = load_json(plan_path)
    predictor_path = run/"fit/predictor.json"
    predictor = load_json(predictor_path)
    curves_path = run/"fit/oof_patch_curves.npz"
    if (predictor["run_id"]!=plan["run_id"] or predictor["frozen_config_sha256"]!=digest(plan_path)
            or predictor["curve_sha256"]!=digest(curves_path) or plan["seeds"]!=[42,43,44]):
        raise ValueError("Predictor and OOF curves differ from the frozen three-seed run")
    with np.load(curves_path,allow_pickle=False) as archive:
        grid,counts,events,patches = (archive[k] for k in ("biases","counts","event_ids","patch_ids"))
        if list(archive["count_names"])!=["tp","fp","fn"]:
            raise ValueError("Unexpected TP/FP/FN order")
    if (counts.shape!=(4000,len(grid),3) or len(set(patches))!=4000 or len(set(events))!=27
            or not np.array_equal(grid,predictor["candidate_biases"]) or np.any(np.diff(grid)<=0)
            or np.any(counts<0)):
        raise ValueError("Expected frozen 4,000 training patches and candidate grid")
    membership = {}
    for fold,sha in predictor["oof_manifest_sha256"].items():
        folder = run/f"fold-{fold}"
        path = folder/"oof/manifest.json"
        part = load_json(path)
        split = folder/"split_manifest.json"
        if (digest(path)!=sha or part["split"]!="train" or part["member_seeds"]!=[42,43,44]
                or digest(split)!=plan["folds"][fold]):
            raise ValueError(f"Changed OOF fold: {fold}")
        role = load_json(split)
        ids = [p["patch_id"] for p in part["patches"]]
        targets = {int(p["event_id"]) for p in part["patches"]}
        if (ids!=role["holdout_patch_ids"] or targets!=set(role["holdout_events"])
                or targets.intersection(role["source_events"])):
            raise ValueError(f"Invalid OOF membership for fold {fold}")
        for p in part["patches"]:
            if p["patch_id"] in membership:
                raise ValueError("Duplicate OOF patch")
            membership[p["patch_id"]] = p["event_id"]
    if membership!=dict(zip(patches,events,strict=True)):
        raise ValueError("Curve patches do not match OOF artifacts")
    by_event = np.stack([counts[events==e].sum(axis=0) for e in sorted(set(events),key=int)])
    tp,fp,fn = np.moveaxis(by_event,2,0)
    macro = (tp/np.maximum(tp+fp+fn,1)).mean(axis=0)
    tp,fp,fn = by_event.sum(axis=0).T
    f1 = 2*tp/np.maximum(2*tp+fp+fn,1)
    archived = read_csv(run/"fit/source_reference_curve.csv")
    checked([[float(r["bias"]),float(r["macro_flood_iou"])] for r in archived],
            np.column_stack((grid,macro)), "all OOF Source Reference candidates", TOL)
    old_i, new_i = best_index(macro,grid),best_index(f1,grid)
    checked(grid[old_i],predictor["reference_bias"],"locked Source Reference bias",TOL)
    lock = dict(run_id=plan["run_id"],predictor_sha256=digest(predictor_path),
              frozen_config_sha256=digest(plan_path),oof_curves_sha256=digest(curves_path),
              training_events=27,training_patches=4000,protocol="select on frozen OOF curves only",
              tie_break="nearest zero then smaller bias at tolerance 1e-12",
              chosen={REF:dict(bias=float(grid[old_i]),objective="event_macro_flood_iou",
                               oof_macro_iou=float(macro[old_i]),oof_pooled_f1=float(f1[old_i])),
                      F1:dict(bias=float(grid[new_i]),objective="pooled_flood_f1",
                              oof_macro_iou=float(macro[new_i]),oof_pooled_f1=float(f1[new_i]))})
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/"scalar_bias_lock.json",lock)
    save_csv(out/"oof_scalar_curves.csv",[dict(bias=b,pooled_flood_f1=f,event_macro_flood_iou=i,
        tp=t,fp=p,fn=n) for b,f,i,t,p,n in zip(grid,f1,macro,tp,fp,fn,strict=True)])
    return plan


def event_matrices(run, out, plan):
    lock = load_json(out/"scalar_bias_lock.json")
    eval_cfg = load_json(run/"results/evaluation_config.json")
    prev_lock = run/"results/bias_lock.json"
    if (eval_cfg["frozen_config_sha256"]!=lock["frozen_config_sha256"]
            or eval_cfg["predictor_sha256"]!=lock["predictor_sha256"]
            or digest(prev_lock)!=eval_cfg["bias_lock_sha256"]
            or eval_cfg["split"]!="validation" or eval_cfg["patches"]!=959):
        raise ValueError("Archived results differ from the frozen run")
    val = run/"validation"
    scope = run/"validation_manifest.json"
    if (digest(val/"manifest.json")!=eval_cfg["validation_cache_sha256"]
            or digest(scope)!=plan["validation_manifest_sha256"]):
        raise ValueError("Validation cache or patch scope changed")
    manifest = read_manifest(val,dict(split="validation",role="validation_ensemble",
        features_only=False,member_seeds=[42,43,44],predictor_sha256=lock["predictor_sha256"],
        scope_sha256=plan["validation_manifest_sha256"],checkpoint_sha256=[m["sha256"] for m in plan["full_train"]]))
    ids = load_json(scope)["patch_ids"]
    if len(ids)!=959 or [p["patch_id"] for p in manifest["patches"]]!=ids:
        raise ValueError("Validation patch order differs from frozen 959-patch scope")
    cache,_ = lookup([(val,manifest)])
    with np.load(run/"results/event_confusions.npz",allow_pickle=False) as p:
        events,methods,counts = (p[k].tolist() if k!="confusion" else p[k]
                                 for k in ("event_ids","methods","confusion"))
    if (len(events)!=6 or set(events)!=set(cache) or len(set(methods))!=len(methods)
            or not set(METHODS).difference({F1}).issubset(methods) or counts.shape!=(6,len(methods),3,3)):
        raise ValueError("Unexpected archived confusion matrices")
    tables = {(r["event_id"],r["method"]):r for r in read_csv(run/"results/event_metrics.csv")}
    original = load_json(prev_lock)["events"]
    result,event_rows = {},[]
    bias = lock["chosen"][F1]["bias"]
    for i,e in enumerate(events):
        z,y = values(cache,e,"logits"),values(cache,e,"labels")
        cm = np.zeros((3,3),dtype=np.int64)
        truth = np.zeros(3,dtype=np.int64)
        for start in range(0,len(y),CHUNK):
            s = slice(start,start+CHUNK)
            labels = y[s]
            keep = labels!=IGNORE
            labels = labels[keep]
            logits = z[s][keep]
            if not np.isin(labels,[0,1,2]).all() or not np.isfinite(logits).all():
                raise ValueError(f"Invalid validation cache values: {e}")
            truth += np.bincount(labels,minlength=3)
            cm += predicted_confusion(flood_margin(logits),logits[:,:2].argmax(1),labels,bias)
        result[e] = {m:cm if m==F1 else counts[i,methods.index(m)] for m in METHODS}
        checked(original[e]["scalar_biases"][REF],lock["chosen"][REF]["bias"],f"{e} macro scalar lock",TOL)
        for m,c in result[e].items():
            np.testing.assert_array_equal(c.sum(axis=1),truth,err_msg=f"{e}/{m} truth support")
            metrics = scores(c)
            row = tables[e,NOBIAS if m==F1 else m]
            b = bias if m==F1 else float(row["bias"]) if row["bias"] else None
            if m!=F1:
                checked(list(metrics.values()),[float(row[k]) for k in metrics],f"{e}/{m} saved metrics",TOL)
            event_rows.append(dict(event_id=e,method=m,bias=b,**metrics,ap=float(row["ap"]),
                 tp=int(c[2,2]),fp=int(c[:2,2].sum()),fn=int(c[2,:2].sum()),
                 event_oracle_gap_pp=100*(scores(result[e][EVENT])["flood_iou"]-metrics["flood_iou"]),
                 patch_oracle_gap_pp=100*(scores(result[e][PATCH])["flood_iou"]-metrics["flood_iou"])))
    return result,event_rows


def stats(mats, weights):
    c = np.einsum("be,eij->bij",weights,mats)
    diag = np.diagonal(c,axis1=1,axis2=2)
    true,pred = c.sum(2),c.sum(1)
    t = diag[:,2]
    row_t = mats[:,2,2]
    e_iou = row_t/np.maximum(mats[:,2,:].sum(1)+mats[:,:,2].sum(1)-row_t,1)
    mean = (weights@e_iou)/weights.sum(1)
    return dict(pooled_miou=(diag/np.maximum(true+pred-diag,1)).mean(1),
                pooled_flood_iou=t/np.maximum(true[:,2]+pred[:,2]-t,1),
                pooled_flood_f1=2*t/np.maximum(true[:,2]+pred[:,2],1),
                pooled_flood_precision=t/np.maximum(pred[:,2],1),
                pooled_flood_recall=t/np.maximum(true[:,2],1),
                event_macro_flood_iou=mean,
                event_flood_iou_std=np.sqrt(np.maximum((weights@(e_iou**2))/weights.sum(1)-mean**2,0)),
                worst_event_iou=np.where(weights>0,e_iou,np.inf).min(1))


def summarize(cms, weights):
    result = {m:stats(np.stack([row[m] for row in cms.values()]),weights) for m in METHODS}
    for m,r in result.items():
        r.update(mean_event_oracle_gap=result[EVENT]["event_macro_flood_iou"]-r["event_macro_flood_iou"],
                 mean_patch_oracle_gap=result[PATCH]["event_macro_flood_iou"]-r["event_macro_flood_iou"])
    return result


def reports(run,out,cms,rows):
    events = list(cms)
    n = len(events)
    kept = np.vstack((np.ones((1,n),dtype=int),1-np.eye(n,dtype=int)))
    point = summarize(cms,kept)
    lookup_ = {(r["event_id"],r["method"]):r for r in rows}
    archived = {(r["omitted_event"],r["method"]):r for r in read_csv(run/"results/leave_one_event_out.csv")}
    per_omission, loo_delta = [],[]
    for i,e in enumerate(("none",*events)):
        by_method = {}
        for m in METHODS:
            base = archived[e,NOBIAS if m==F1 else m]
            rates = {k:float(v[i]) for k,v in point[m].items()}
            if m!=F1:
                keys = [k for k in rates if not k.endswith("_gap")]
                checked([rates[k] for k in keys],[float(base[k]) for k in keys],
                        f"{e}/{m} frozen leave-one-out report",TOL)
            by_method[m] = rates | {"pooled_ap":float(base["pooled_ap"])}
            per_omission.append(dict(omitted_event=e,events=n-int(i>0),method=m,
                **{k if not k.endswith("_gap") else f"{k}_pp":v if not k.endswith("_gap") else 100*v
                   for k,v in by_method[m].items()}))
        for m,ref in PAIRS:
            for key,a in by_method[m].items():
                loo_delta.append(dict(omitted_event=e,events=n-int(i>0),method=m,reference=ref,
                    metric=key,delta_pp=100*(a-by_method[ref][key])))
    wins, differences = [],[]
    for m,ref in PAIRS:
        wtl = dict(win=0,tie=0,loss=0)
        for e in events:
            a,b=lookup_[e,m],lookup_[e,ref]
            delta = a["flood_iou"]-b["flood_iou"]
            outcome = "win" if delta>TOL else "loss" if delta<-TOL else "tie"
            wtl[outcome]+=1
            differences.append(dict(method=m,reference=ref,event_id=e,outcome=outcome,
                iou_delta_pp=100*delta,f1_delta_pp=100*(a["flood_f1"]-b["flood_f1"])))
        wins.append(dict(method=m,reference=ref,events=n,**wtl))
    picks = np.random.default_rng(SEED).integers(n,size=(ITERATIONS,n))
    draws = (picks[...,None]==np.arange(n)).sum(1)
    stats_ = summarize(cms,draws)
    ci = []
    for m,ref in PAIRS:
        for key,vector in stats_[m].items():
            lo,hi,valid = interval(100*(vector-stats_[ref][key]))
            ci.append(dict(method=m,reference=ref,metric=key,iterations=valid,
                delta_pp=100*(point[m][key][0]-point[ref][key][0]),ci_low_pp=lo,ci_high_pp=hi))
    for name,data in (
        ("summary",[{k:v for k,v in r.items() if k!="omitted_event"} for r in per_omission if r["omitted_event"]=="none"]),
        ("event_metrics",rows),("paired_bootstrap",ci),("win_tie_loss",wins),("event_differences",differences),
        ("leave_one_event_out",per_omission),("leave_one_event_out_paired",loo_delta)):
        save_csv(out/f"{name}.csv",data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    plan = lock_oof(a.run,a.out)
    cm, rows = event_matrices(a.run,a.out,plan)
    reports(a.run,a.out,cm,rows)
    write_json(a.out/"analysis_config.json",dict(split="validation",patches=959,events=6,
        bias_lock_sha256=digest(a.out/"scalar_bias_lock.json"),run=str(a.run.resolve()),
        validation_manifest_sha256=plan["validation_manifest_sha256"],
        source_reference="verified as exactly the matching OOF macro-IoU optimum and reused",
        ap="point estimates; one common scalar leaves all flood-margin ranks unchanged",
        bootstrap="10000 paired whole-event percentile replicates, seed 42, metrics from confusion counts",
        leave_one_event_out="omit event from aggregation only; all biases remain locked",
        caveat="comparison of policies on six events; OOF scalar objectives differ from CCLA's macro objective",
        script_sha256=digest(Path(__file__))))
    print(f"Saved OOF scalar vs CCLA audit: {a.out}",flush=True)


if __name__=="__main__":
    main()
