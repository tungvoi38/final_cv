"""Conservative multi-domain Expected IoU using frozen validation cache only."""
from pathlib import Path

import numpy as np
from scipy.special import expit

import adapt_unet_expected_iou as original
from adapt_unet_event_bias import FLOOD, IGNORE, digest, flood_margin, load_json, oracle_from_ranks, predict_confusion, reference, scores, write_json
from audit_unet_bias_failures import archived_inputs, checked, operating_points, read_csv, save_csv
from evaluate_unet_eoa_diagnostics import rank_event

BALANCED = "event_balanced_expected_iou"
ROBUST = "conservative_multi_domain"
REF = "source_macro_iou_reference"
ORACLE = "oracle_DIAGNOSTIC_ONLY"
METHODS = (original.SAR, BALANCED, ROBUST, REF, ORACLE)
GAIN_TOL = 1e-12


def get_sar(raw, inputs, logits_cache, directory):
    """Reuse physically aligned SAR features tied to the immutable cache order."""
    if directory is None:
        return original.align_sar(raw,inputs)
    cache_hash=digest(logits_cache/"manifest.json")
    counts={e:len(values) for e,values in inputs.items()}
    from kuro_dataset import RAW_SHA256
    identity = dict(split="validation",raw_sha256=RAW_SHA256,mean_cache_sha256=cache_hash,
                    features=list(original.FEATURES[original.SAR][1:]),event_counts=counts)
    if not directory.exists():
        arrays=original.align_sar(raw,inputs)
        directory.mkdir(parents=True,exist_ok=False)
        file_hashes={}
        for event,array in arrays.items():
            path=directory/f"{event}.npy"
            np.save(path,array,allow_pickle=False)
            file_hashes[event]=digest(path)
        write_json(directory/"manifest.json",identity|{"file_sha256":file_hashes})
        del arrays
    manifest=load_json(directory/"manifest.json")
    if any(manifest.get(k)!=v for k,v in identity.items()):
        raise ValueError("SAR feature cache provenance does not match the validation logit cache")
    result={}
    for event,size in counts.items():
        path=directory/f"{event}.npy"
        if digest(path)!=manifest["file_sha256"][event]:
            raise ValueError(f"Changed SAR features: {path}")
        values=np.load(path,mmap_mode="r",allow_pickle=False)
        if values.shape!=(size,3) or values.dtype!=np.float32:
            raise ValueError(f"Invalid SAR feature dimensions: {path}")
        result[event]=values
    return result


def best_index(values, grid, *, anchor=0.):
    best=np.max(values)
    tied=np.flatnonzero(np.isclose(values,best,atol=GAIN_TOL,rtol=0))
    return int(min(tied,key=lambda i:(abs(grid[i]-anchor),abs(grid[i]),grid[i])))


def reference_from_sources(source_ids, archived_confusions, grid):
    """Original 3-class confusion order is true label by prediction."""
    event_ious=[]
    for event in source_ids:
        cm=archived_confusions[event]
        tp=cm[:,2,2]
        union=cm[:,2,:].sum(-1)+cm[:,:,2].sum(-1)-tp
        if np.any(union<=0):
            raise ValueError(f"Undefined Flood IoU on source grid: {event}")
        event_ious.append(tp/union)
    curve=np.mean(event_ious,axis=0)
    return best_index(curve,grid),curve


def fit_balanced(samples, source_ids):
    # Equal weight per event. Uniform within event sampling preserves its prevalence.
    parts=[(samples[e][0],samples[e][1],np.full(len(samples[e][1]),1/len(samples[e][1])))
           for e in source_ids]
    scaler,model,metadata=original.fit_calibrator(parts,4)
    coef=model.coef_[0]/scaler.scale_
    intercept=float(model.intercept_[0]-coef@scaler.mean_)
    return coef,intercept,{"events":source_ids,**metadata}


def multi_curves(logits,sar,grid,coefficients,intercepts):
    """Compute six expected-IoU curves without keeping per-pixel posteriors."""
    if len(logits)!=len(sar) or not len(logits):
        raise ValueError("Mismatched or empty input-valid feature arrays")
    columns=coefficients.shape[0]
    bins=len(grid)+1
    pred_count=np.zeros(bins,dtype=np.int64)
    mass=np.zeros((columns,bins),dtype=np.float64)
    nonflood_mass=np.zeros_like(mass)
    for start in range(0,len(logits),original.CHUNK):
        sl=slice(start,start+original.CHUNK)
        x=original.design(logits[sl],sar[sl]).astype(np.float64)
        index=np.searchsorted(grid,-x[:,0],side="right")
        q=expit(x@coefficients.T+intercepts)
        if not np.isfinite(q).all():
            raise ValueError("Nonfinite multi-domain calibrated probabilities")
        pred_count+=np.bincount(index,minlength=bins)
        for k in range(columns):
            mass[k]+=np.bincount(index,weights=q[:,k],minlength=bins)
            nonflood_mass[k]+=np.bincount(index,weights=1-q[:,k],minlength=bins)
    tp=mass.cumsum(axis=1)[:,:-1]
    fp=nonflood_mass.cumsum(axis=1)[:,:-1]
    fn=mass[:,::-1].cumsum(axis=1)[:,::-1][:,1:]
    union=tp+fp+fn
    iou=np.divide(tp,union,out=np.zeros_like(tp),where=union>0)
    return iou,pred_count.cumsum()[:-1]


def choose_conservative(curves, ref, grid):
    gains=curves-curves[:,ref,None]
    minimum=gains.min(axis=0)
    candidate=best_index(minimum,grid,anchor=grid[ref])
    fallback=bool(minimum[candidate]<=GAIN_TOL)
    return (ref if fallback else candidate),fallback,gains,minimum


def lock_target(target, inputs, sar, source, confusions, grid):
    source_ids = [e for e in sorted(source,key=int) if e!=target]
    if len(source_ids)!=5:
        raise ValueError("Expected five source events after excluding the held-out event")
    ref,source_curve = reference_from_sources(source_ids,confusions,grid)
    domains = [source_ids,*[[e for e in source_ids if e!=omitted] for omitted in source_ids]]
    fitted=[fit_balanced(source,domain) for domain in domains]
    coefficients=np.stack([fit[0] for fit in fitted])
    intercepts=np.array([fit[1] for fit in fitted])
    curves,pred=multi_curves(inputs,sar,grid,coefficients,intercepts)
    balanced=best_index(curves[0],grid)
    robust,fallback,gains,minimum=choose_conservative(curves,ref,grid)
    metadata=dict(source_events=source_ids,reference_bias=float(grid[ref]),
        source_macro_iou_at_reference=float(source_curve[ref]),balanced_bias=float(grid[balanced]),
        robust_bias=float(grid[robust]),robust_fallback=fallback,min_gain_at_robust=float(minimum[robust]),
        min_gain_at_balanced=float(minimum[balanced]),calibrators={
            key:fit[2] for key,fit in zip(["all_sources",*[f"omit_{e}" for e in source_ids]],fitted,strict=True)})
    return metadata,(ref,balanced,robust,curves,pred,gains,minimum,source_curve)


def verify_original(results, study_results, cache, recipe):
    lock_path=results/"bias_lock.json"
    config=load_json(results/"evaluation_config.json")
    lock=load_json(lock_path)
    if (config["bias_lock_sha256"]!=digest(lock_path)
            or lock["checkpoint_selection_sha256"]!=recipe["selection_sha256"]
            or lock["mean_cache_sha256"]!=digest(cache/"manifest.json")
            or lock["original_bias_lock_sha256"]!=digest(study_results/"bias_lock.json")):
        raise ValueError("Original Expected IoU artifacts do not match frozen validation provenance")
    rows=[r for r in read_csv(results/"event_metrics.csv") if r["method"]==original.SAR]
    by_event={r["event_id"]:r for r in rows}
    if len(rows)!=6 or set(by_event)!=set(lock["event_biases"]):
        raise ValueError("Incomplete historical margin + SAR comparison")
    for event,row in by_event.items():
        checked(float(row["bias"]),lock["event_biases"][event]["methods"][original.SAR]["bias"],event)
    return by_event,np.array(lock["grid"])


def summarize(rows):
    result=[]
    for method in METHODS:
        group=[r for r in rows if r["method"]==method]
        tp,fp,fn=[sum(r[key] for r in group) for key in ("tp","fp","fn")]
        ious=[r["flood_iou"] for r in group]
        result.append(dict(method=method,events=len(group),
            pooled_flood_f1=2*tp/max(2*tp+fp+fn,1),event_macro_flood_iou=float(np.mean(ious)),
            event_flood_iou_std=float(np.std(ious,ddof=0)),worst_event_iou=float(min(ious)),
            worst_event_id=min(group,key=lambda r:r["flood_iou"])["event_id"],
            mean_exact_oracle_gap_pp=float(np.mean([r["exact_oracle_gap_pp"] for r in group])),
             mean_grid_oracle_gap_pp=(float(np.mean([r["grid_oracle_gap_pp"] for r in group]))
                                      if method!=ORACLE else None),
            fallback_events=sum(r["fallback"] for r in group) if method==ROBUST else None,
            fallback_rate=float(np.mean([r["fallback"] for r in group])) if method==ROBUST else None))
    return result


def run(study,cache,out,original_results,sar_cache):
    if out.exists():
        raise FileExistsError(out)
    saved=study/"results"
    events,x,y,saved_metrics,_,grid,confusions,_,_=archived_inputs(saved,cache,study/"views")
    recipe,selection,_,_=reference()
    previous,old_grid=verify_original(original_results,saved,cache,recipe)
    if not np.array_equal(grid,old_grid):
        raise ValueError("Original and conservative Expected IoU have different candidate grids")
    sar=get_sar(selection["protocol"]["training"]["raw"],x,cache,sar_cache)
    source=original.source_samples(x,sar,y)
    locked,inputs={},{}

    for target in events:
        locked[target],inputs[target]=lock_target(target,x[target],sar[target],source,confusions,grid)
        print(f"Locked {target}: ref={locked[target]['reference_bias']:+.2f} "
              f"balanced={locked[target]['balanced_bias']:+.2f} robust={locked[target]['robust_bias']:+.2f} "
              f"fallback={locked[target]['robust_fallback']}",flush=True)

    out.mkdir(parents=True,exist_ok=False)
    lock_path=out/"bias_lock.json"
    write_json(lock_path,dict(scope="exploratory official validation only",
        dataset_alignment="frozen mean logit cache, source SAR in original input-valid order",
        cache_sha256=digest(cache/"manifest.json"),sar_cache=str(sar_cache) if sar_cache else None,
        sar_cache_sha256=digest(sar_cache/"manifest.json") if sar_cache else None,
        candidate_grid=grid.tolist(),C=1.,solver="lbfgs",sample_cap=original.SOURCE_CAP,
        weighting="equal total weight per source event, source-only standardization, mean sample weight one",
        reference="macro Flood IoU across the five source events, ties nearest zero then lower",
        objective="max over candidate bias of min across calibrators of the expected IoU increase vs source reference",
        fallback=f"no worst-calibrator gain above {GAIN_TOL}; keep source reference",
        robust_tie="nearest reference, then nearest zero, then lower bias",
        evaluation="target annotations and oracles only after lock",
        original_expected_iou=str(original_results),events=locked))

    rows,curves,oracle_checks=[],[],[]
    for event in events:
        keep=y[event]!=IGNORE
        labels=y[event][keep]
        z=x[event][keep]
        ranked=rank_event([(flood_margin(z),labels==FLOOD)])
        _,_,_,actual,_=operating_points(ranked,grid,len(labels))
        grid_best,exact=best_index(actual,grid),oracle_from_ranks(ranked)
        exact_bias,exact_iou=exact
        historical_oracle=saved_metrics[event,ORACLE]
        checked([exact_bias,exact_iou],[historical_oracle["bias"],historical_oracle["flood_iou"]],
                f"{event} exact oracle")
        checked([exact_bias,exact_iou],[float(previous[event]["oracle_bias"]),float(previous[event]["oracle_flood_iou"])],
                f"{event} original Expected IoU oracle")
        oracle_checks.append(dict(event_id=event,exact_oracle_bias=exact_bias,exact_oracle_iou=exact_iou,
            grid_oracle_bias=float(grid[grid_best]),grid_oracle_iou=float(actual[grid_best]),
            exact_bias_outside_grid=not grid[0]<=exact_bias<=grid[-1],
            exact_minus_grid_iou_pp=100*(exact_iou-actual[grid_best]),matches_historical=True))
        ref,balanced,robust,estimated,pred,gains,minimum,source_curve=inputs[event]
        biases={original.SAR:float(previous[event]["bias"]),BALANCED:float(grid[balanced]),
                ROBUST:float(grid[robust]),REF:float(grid[ref]),ORACLE:exact_bias}
        for method,bias in biases.items():
            cm=predict_confusion(z,labels,bias)
            values=scores(cm)
            iou=values["flood_iou"]
            if method==original.SAR:
                checked(iou,float(previous[event]["flood_iou"]),f"{event} historical candidate")
            if method==ORACLE:
                checked(iou,exact_iou,f"{event} exact oracle operating point")
            i=int(np.searchsorted(grid,bias)) if method!=ORACLE else None
            if i is not None:
                checked(grid[i],bias,f"{event} bias on agreed grid")
                checked(iou,actual[i],f"{event} candidate IoU")
            tp=int(cm[2,2])
            rows.append(dict(event_id=event,method=method,bias=bias,reference_bias=float(grid[ref]),
                fallback=locked[event]["robust_fallback"] if method==ROBUST else None,
                estimated_iou_all_sources=float(estimated[0,i]) if i is not None else None,
                worst_estimated_gain_vs_reference=float(minimum[i]) if i is not None else None,
                exact_oracle_bias=exact_bias,exact_oracle_gap_pp=100*(exact_iou-iou),
                grid_oracle_bias=float(grid[grid_best]),grid_oracle_gap_pp=None if method==ORACLE else 100*(actual[grid_best]-iou),
                actual_flood_prevalence=float(np.mean(labels==FLOOD)),
                predicted_prevalence_scored=float(cm[:,2].sum()/len(labels)),
                tp=tp,fp=int(cm[:,2].sum())-tp,fn=int(cm[2].sum())-tp,**values))
        names=list(locked[event]["calibrators"])
        for k,bias in enumerate(grid):
            curves.append(dict(event_id=event,bias=float(bias),source_macro_iou=float(source_curve[k]),
                actual_target_iou=float(actual[k]),min_gain=float(minimum[k]),predicted_input_valid=int(pred[k]),
                reference=k==ref,balanced=k==balanced,robust=k==robust,
                **{f"estimated_iou_{name}":float(estimated[j,k]) for j,name in enumerate(names)},
                **{f"estimated_gain_{name}":float(gains[j,k]) for j,name in enumerate(names)}))
    save_csv(out/"summary.csv",summarize(rows))
    save_csv(out/"event_metrics.csv",rows)
    save_csv(out/"event_279.csv",[r for r in rows if r["event_id"]=="279"])
    save_csv(out/"curves.csv",curves)
    save_csv(out/"oracle_audit.csv",oracle_checks)
    write_json(out/"evaluation_config.json",dict(
        scope="six exploratory official validation event folds; no model inference",
        bias_lock_sha256=digest(lock_path),original_expected_iou=str(original_results),
        exact_oracle="all attainable thresholds on scored target margins; matches historical diagnostic",
        grid_oracle="maximum actual IoU on same grid as original Expected IoU, diagnostic only",
        historical_selection_caveat="frozen checkpoints were selected on all six events; this is not full LOEO",
        code_sha256=digest(Path(__file__))))
    print(f"Saved Conservative Expected IoU comparison: {out}",flush=True)
