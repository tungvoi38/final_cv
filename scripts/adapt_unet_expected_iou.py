#!/usr/bin/env python3
"""Exploratory source-calibrated expected Flood IoU on frozen validation predictions."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from adapt_unet_event_bias import (
    FLOOD, IGNORE, SEED, digest, flood_margin, predict_confusion, reference, scores, write_json,
)
from audit_unet_bias_failures import archived_inputs, checked, operating_points, read_csv, save_csv
from evaluate_unet_eoa_diagnostics import pr_curve, rank_event, rank_pooled
from kuro_dataset import KuroDataset, RawArchive

MARGIN = "expected_iou_margin"
SAR = "expected_iou_margin_sar"
FEATURES = {MARGIN: ("flood_margin",),
            SAR: ("flood_margin", "post_vv_db", "delta_vv_db", "delta_vh_db")}
SOURCE_CAP = 100_000
CHUNK = 250_000
ECE_BINS = 15


def sar_features(sar_db, input_valid):
    sar = sar_db[:,:,input_valid]
    post = sar[2]
    delta = post-(sar[0]+sar[1])*.5
    return np.column_stack((post[0],delta[0],delta[1]))


def align_sar(raw, inputs):
    ds = KuroDataset(RawArchive(raw),"validation")
    if set(ds.rows.actid.astype(str))!=set(inputs):
        raise ValueError("Official validation event IDs differ from the logit cache")
    result = {e:np.empty((len(z),3),dtype=np.float32) for e,z in inputs.items()}
    offsets = dict.fromkeys(inputs,0)
    for index in range(len(ds)):
        sample=ds.physical(index)
        event=sample["event_id"]
        features=sar_features(sample["sar_db"],sample["input_valid"])
        start,stop=offsets[event],offsets[event]+len(features)
        if stop>len(result[event]) or not np.isfinite(features).all():
            raise ValueError(f"SAR/logit size or support mismatch: {event}")
        result[event][start:stop]=features
        offsets[event]=stop
        if (index+1)%250==0 or index+1==len(ds):
            print(f"Aligned validation SAR: {index+1}/{len(ds)} patches",flush=True)
    if any(offsets[e]!=len(inputs[e]) for e in inputs):
        raise ValueError("Incomplete SAR alignment to the input-valid logit order")
    return result


def design(logits, sar=None):
    margin=flood_margin(logits)
    return margin[:,None] if sar is None else np.column_stack((margin,sar))


def source_samples(inputs,sar,labels):
    result={}
    for event in inputs:
        eligible=np.flatnonzero(labels[event]!=IGNORE)
        if not len(eligible):
            raise ValueError(f"No scored source pixels in {event}")
        rng=np.random.default_rng(np.random.SeedSequence([SEED,int(event)]))
        chosen=np.sort(rng.choice(eligible,SOURCE_CAP,replace=False)) if len(eligible)>SOURCE_CAP else eligible
        y=labels[event][chosen]==FLOOD
        result[event]=(design(inputs[event][chosen],sar[event][chosen]),y,
                       np.full(len(chosen),len(eligible)/len(chosen)))
    return result


def fit_calibrator(source, width):
    x,y,w=(np.concatenate(parts) for parts in zip(*source,strict=True))
    if np.unique(y).size!=2:
        raise ValueError("Logistic calibration requires both source classes")
    w /= w.mean()
    x=x[:,:width].astype(np.float64)
    scaler=StandardScaler().fit(x,sample_weight=w)
    model=LogisticRegression(C=1.,solver="lbfgs",max_iter=1000,tol=1e-8)
    model.fit(scaler.transform(x),y,sample_weight=w)
    if model.n_iter_[0]>=model.max_iter:
        raise RuntimeError("Source logistic regression did not converge")
    return scaler,model,dict(source_sampled_pixels=len(y),source_weighted_prevalence=float(np.average(y,weights=w)),
        fit_iterations=int(model.n_iter_[0]),scaler_mean=scaler.mean_.tolist(),
        scaler_scale=scaler.scale_.tolist(),coefficients=model.coef_[0].tolist(),
        intercept=float(model.intercept_[0]))


def predict_q(logits,sar,scaler,calibrator):
    """No target annotations or annotation-derived masks enter this function."""
    output=np.empty(len(logits),dtype=np.float64)
    for start in range(0,len(output),CHUNK):
        sl=slice(start,start+CHUNK)
        x=design(logits[sl],None if sar is None else sar[sl]).astype(np.float64)
        output[sl]=calibrator.predict_proba(scaler.transform(x))[:,1]
    if not np.isfinite(output).all() or not len(output):
        raise ValueError("Invalid target probabilities")
    return output


def expected_curve(margin,q,grid):
    if (len(margin)!=len(q) or not len(q) or np.any((q<0)|(q>1))
            or not np.isfinite(q).all() or not np.isfinite(margin).all()
            or not np.all(grid[:-1]<grid[1:])):
        raise ValueError("Invalid expected-IoU inputs")
    # For each pixel, count it at the first candidate bias satisfying margin > -b.
    # The last bin represents pixels predicted nonflood for every grid candidate.
    nb=len(grid)+1
    hit=np.zeros(nb,dtype=np.int64)
    hit_q=np.zeros(nb,dtype=np.float64)
    miss_q=np.zeros(nb,dtype=np.float64)
    for start in range(0,len(q),CHUNK):
        sl=slice(start,start+CHUNK)
        ids=np.searchsorted(grid,-margin[sl].astype(np.float64),side="right")
        hit+=np.bincount(ids,minlength=nb)
        hit_q+=np.bincount(ids,weights=q[sl],minlength=nb)
        miss_q+=np.bincount(ids,weights=1-q[sl],minlength=nb)
    pred=np.cumsum(hit)[:-1]
    tp=np.cumsum(hit_q)[:-1]
    fp=np.cumsum(miss_q)[:-1]
    fn=np.cumsum(hit_q[::-1])[::-1][1:]
    denom=tp+fp+fn
    iou=np.divide(tp,denom,out=np.zeros_like(tp),where=denom>0)
    winners=np.flatnonzero(np.isclose(iou,iou.max(),atol=1e-12,rtol=0))
    winner=min(winners,key=lambda index:(abs(grid[index]),grid[index]))
    return dict(tp=tp,fp=fp,fn=fn,predicted=pred,iou=iou),int(winner)


def calibration_sums(q,labels):
    bins=np.minimum((q*ECE_BINS).astype(int),ECE_BINS-1)
    clipped=np.clip(q,1e-15,1-1e-15)
    return (len(q),float(np.sum((q-labels)**2)),
            float(-np.where(labels,np.log(clipped),np.log1p(-clipped)).sum()),
            np.bincount(bins,minlength=ECE_BINS),
            np.bincount(bins,weights=q,minlength=ECE_BINS),
            np.bincount(bins,weights=labels,minlength=ECE_BINS))


def calibration_metrics(stat):
    n,brier,nll,_,sum_q,sum_y=stat
    return dict(n_scored_pixels=int(n),brier=float(brier/n),nll=float(nll/n),
                ece_15_equal_width=float(np.abs(sum_q-sum_y).sum()/n),
                mean_calibrated_probability=float(sum_q.sum()/n),actual_flood_prevalence=float(sum_y.sum()/n))


def add_oracle_comparison(summary, rows):
    oracle = {r["event_id"]:r for r in rows if r["method"]=="oracle_DIAGNOSTIC_ONLY"}
    for item in summary:
        group = [r for r in rows if r["method"]==item["method"]]
        if len(group)!=len(oracle) or not group:
            raise ValueError(f"Missing oracle comparison coverage for {item['method']}")
        item["mean_absolute_bias_error"] = float(np.mean([
            abs(float(r["bias"])-float(oracle[r["event_id"]]["bias"])) for r in group]))
        item["mean_oracle_gap_pp"] = 100*float(np.mean([
            float(oracle[r["event_id"]]["flood_iou"])-float(r["flood_iou"]) for r in group]))


def plot(out,events,curves):
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout="constrained")
    for ax,event in zip(axes.flat,events,strict=True):
        for method,color,label in ((MARGIN,"#2c638b","Margin only"),(SAR,"#2e885f","Margin + SAR")):
            rows=[r for r in curves if (r["event_id"],r["method"])==(event,method)]
            ax.plot([r["bias"] for r in rows],[100*r["estimated_iou"] for r in rows],color=color,label=label)
            selected=next(r for r in rows if r["selected"])
            ax.scatter(selected["bias"],100*selected["actual_iou"],color=color,marker="x",zorder=5)
        ax.plot([r["bias"] for r in rows],[100*r["actual_iou"] for r in rows],color=".35",ls="--",label="Actual IoU")
        ax.set(title=f"Event {event}",xlabel="Flood bias",ylabel="IoU (%)")
        ax.grid(alpha=.18)
    axes[0,0].legend(fontsize=8)
    fig.suptitle("Validation exploratory · crosses show actual IoU at label-free selected bias")
    fig.savefig(out/"estimated_vs_actual_iou.png",dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",required=True,type=Path,help="Saved adaptation study with results/ and views/")
    parser.add_argument("--cache",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--conservative",action="store_true",help="Run event-balanced and robust ablations")
    parser.add_argument("--original-expected-iou",type=Path,help="Saved original Expected IoU results for comparison")
    parser.add_argument("--sar-cache",type=Path,help="Optional reusable input-aligned SAR feature cache")
    args=parser.parse_args()
    if args.conservative and args.original_expected_iou is None:
        parser.error("--conservative requires --original-expected-iou")
    if not args.conservative and args.original_expected_iou is not None:
        parser.error("--original-expected-iou is only used with --conservative")
    if args.out.exists():
        parser.error("--out must be a new directory")
    study,cache,out=(p.resolve() for p in (args.study,args.cache,args.out))
    sar_cache=args.sar_cache.resolve() if args.sar_cache is not None else None
    if args.conservative:
        from unet_conservative_expected_iou import run

        run(study,cache,out,args.original_expected_iou.resolve(),sar_cache)
        return
    old=study/"results"
    events,x,y,original,_,grid,_,_,_=archived_inputs(old,cache,study/"views")
    # Follow the selection pinned to the same saved cache, rather than accepting a new dataset.
    recipe,selection,_,_=reference()
    raw=selection["protocol"]["training"]["raw"]
    if sar_cache is None:
        sar=align_sar(raw,x)
    else:
        from unet_conservative_expected_iou import get_sar

        sar=get_sar(raw,x,cache,sar_cache)
    source=source_samples(x,sar,y)
    locked,probabilities,expected={},{},{}
    for target in events:
        source_ids=[event for event in events if event!=target]
        locked[target]={"source_events":source_ids,"methods":{}}
        margin=flood_margin(x[target])
        for method,names in FEATURES.items():
            scaler,model,info=fit_calibrator([source[e] for e in source_ids],len(names))
            q=predict_q(x[target],sar[target] if method==SAR else None,scaler,model)
            curve,best=expected_curve(margin,q,grid)
            probabilities[target,method]=q
            expected[target,method]=curve
            locked[target]["methods"][method]=dict(bias=float(grid[best]),index=best,
                estimated_iou=float(curve["iou"][best]),mean_q_input_valid=float(q.mean()),
                predicted_prevalence_input_valid=float(curve["predicted"][best]/len(q)),
                features=list(names),**info)
            print(f"Locked {target} {method}: b={grid[best]:+.2f}, estimated IoU={curve['iou'][best]:.4f}",flush=True)
    out.mkdir(parents=True,exist_ok=False)
    lock_path=out/"bias_lock.json"
    write_json(lock_path,dict(scope="exploratory validation only",event_biases=locked,
        source_policy="other five validation events; logits from frozen pre-existing training, all six were involved in historical checkpoint selection",
        sampling="uniform without replacement among annotation-valid source pixels per event; inverse sample weights",
        source_cap=SOURCE_CAP,seed=SEED,grid=grid.tolist(),candidate_tie="nearest zero then lower bias",
        method="maximize ratio of expected counts with fixed q, not exact Bayes expected IoU",
        logistic=dict(C=1.,solver="lbfgs",standardizer="weighted, fit on source only"),
        sar="post VV dB; delta VV/VH is post minus mean of both pre-event acquisitions in dB, joined in cache order on input_valid",
        input_scope="target input-valid without target annotation IGNORE; target labels used for evaluation only",
        mean_cache_sha256=digest(cache/"manifest.json"),
        original_bias_lock_sha256=digest(old/"bias_lock.json"),checkpoint_selection_sha256=recipe["selection_sha256"]))

    event_metrics,calibration,curves=[],[],[]
    matrix={method:[] for method in FEATURES}
    ap={method:[] for method in FEATURES}
    stats={method:[] for method in FEATURES}
    for target in events:
        keep=y[target]!=IGNORE
        labels=y[target][keep]
        scored=x[target][keep]
        flood=labels==FLOOD
        ranked=rank_event([(flood_margin(scored),flood)])
        _,_,_,actual,_=operating_points(ranked,grid,len(labels))
        oracle=original[target,"oracle_DIAGNOSTIC_ONLY"]
        for method in FEATURES:
            q=probabilities.pop((target,method))
            details=locked[target]["methods"][method]
            c=expected[target,method]
            selected=details["index"]
            bias=details["bias"]
            cm=predict_confusion(scored,labels,bias)
            metrics=scores(cm)
            checked(metrics["flood_iou"],actual[selected],f"{target} {method}")
            cm_cal=calibration_sums(q[keep],flood)
            calibration.append(dict(scope="event",event_id=target,method=method,**calibration_metrics(cm_cal)))
            stats[method].append(cm_cal)
            matrix[method].append(cm)
            ap[method].append((ranked[0].astype(np.float64)+bias,*ranked[1:]))
            corr= None if np.ptp(c["iou"])==0 or np.ptp(actual)==0 else float(spearmanr(c["iou"],actual).statistic)
            event_metrics.append(dict(event_id=target,method=method,bias=bias,
                estimated_iou=details["estimated_iou"],oracle_bias=oracle["bias"],
                bias_minus_oracle=bias-oracle["bias"],oracle_flood_iou=oracle["flood_iou"],
                oracle_gap_pp=100*(oracle["flood_iou"]-actual[selected]),
                iou_prediction_error_pp=100*(details["estimated_iou"]-actual[selected]),
                expected_actual_curve_spearman=corr,predicted_prevalence_input_valid=details["predicted_prevalence_input_valid"],
                mean_q_input_valid=details["mean_q_input_valid"],flood_ap=oracle["flood_ap"],
                predicted_prevalence_scored=float(cm[:,FLOOD].sum()/len(labels)),
                actual_flood_prevalence=float(flood.mean()),**metrics))
            curves.extend(dict(event_id=target,method=method,bias=float(b),selected=i==selected,
                estimated_tp=float(c["tp"][i]),estimated_fp=float(c["fp"][i]),estimated_fn=float(c["fn"][i]),
                estimated_iou=float(c["iou"][i]),actual_iou=float(actual[i])) for i,b in enumerate(grid))
    summary=read_csv(old/"summary.csv")
    for method in FEATURES:
        metrics=scores(sum(matrix[method]))
        rows=[r for r in event_metrics if r["method"]==method]
        iou=[r["flood_iou"] for r in rows]
        summary.append(dict(method=method,n_events=len(rows),**metrics,
            pooled_flood_ap=pr_curve(rank_pooled(ap[method]))[-1],
            event_macro_flood_ap=float(np.mean([r["flood_ap"] for r in rows])),
            event_macro_flood_iou=float(np.mean(iou)),event_flood_iou_std=float(np.std(iou,ddof=0))))
        pooled=[sum(value) for value in zip(*stats[method],strict=True)]
        calibration.append(dict(scope="pooled_held_out",event_id=None,method=method,**calibration_metrics(pooled)))
    old_events=read_csv(old/"event_metrics.csv")
    all_events=old_events+event_metrics
    add_oracle_comparison(summary,all_events)
    save_csv(out/"event_metrics.csv",all_events)
    save_csv(out/"calibration_metrics.csv",calibration)
    save_csv(out/"estimated_actual_iou_curves.csv",curves)
    save_csv(out/"summary.csv",summary)
    plot(out,events,curves)
    write_json(out/"evaluation_config.json",dict(
        scope="exploratory six held-out calibration folds; validation only",
        bias_lock_sha256=digest(lock_path),code_sha256=digest(Path(__file__)),
        calibration_metrics="binary NLL, Brier, 15 equal-width ECE on held-out annotation-valid events",
        oracle_and_actual_curves="diagnostic only after bias lock; never used for selection",
        metrics_sources=str(old),cache=str(cache)))
    print(f"Saved expected-IoU validation comparison: {out}",flush=True)


if __name__=="__main__":
    main()
