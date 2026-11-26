#!/usr/bin/env python3
"""Explore patch-conditioned bias on frozen EoA with event-disjoint method folds."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from adapt_unet_event_bias import FLOOD, IGNORE, VAL_ARTIFACT, digest, flood_margin, load_json, predict_confusion, reference, scores, write_json
from adapt_unet_expected_iou import sar_features
from adapt_unet_iw_iou import cache_file, open_predictions, supplement
from audit_unet_bias_failures import checked, save_csv
from audit_unet_iw_iou import MARGIN as IW_MARGIN, SAR as IW_SAR, load_artifacts
from diagnose_sar_scattering import texture
from diagnose_unet_patch_gating import load_spatial
from diagnose_unet_spatial_oracle import EVENT, GLOBAL, PATCH, patch_ranges
from kuro_dataset import KuroDataset, preprocess, read_sample
from unet_conservative_expected_iou import get_sar

MARGIN, SAR = "local_margin", "local_margin_sar"
VARIANTS = (MARGIN, SAR)
METHODS = (GLOBAL, IW_MARGIN, IW_SAR, *VARIANTS, EVENT, PATCH)
FEATURE_VERSION = "local-patch-context-v1"
SIGMA = .35
RIDGE = .015
Z_LIMIT = 5.
BIAS_ANCHOR = 0.
MARGIN_LEVELS = (-2., 0., 1., 2., 4.)
QUANTILES = (.10, .25, .50, .75, .90, .95, .99)


def margin_features(margin):
    if not len(margin):
        return {k:None for k in margin_features(np.array([0.]))}
    fields = {"margin_mean":float(margin.mean()), "margin_std":float(margin.std())}
    fields.update({f"margin_q{round(q*100):02}":float(v) for q,v in
                   zip(QUANTILES,np.quantile(margin,QUANTILES),strict=True)})
    fields.update({f"prevalence_b{bias:g}":float(np.mean(margin.astype(np.float64)>-bias))
                   for bias in MARGIN_LEVELS})
    return fields


def radiometry_features(sample, slope):
    valid = sample["input_valid"]
    pre1,pre2,post = sample["sar_db"]
    pre = (pre1+pre2)*.5
    delta = post-pre
    var = np.abs(pre2-pre1)
    channels = {}
    for name,layers in (("pre_mean",pre),("post",post),("delta",delta),("pre_variability",var)):
        for pol,z in zip(("vv","vh"),layers,strict=True):
            channels[f"{name}_{pol}_db"] = z[valid]
    channels["post_vh_minus_vv_db"] = (post[1]-post[0])[valid]
    channels["slope_deg"] = slope[np.isfinite(slope)]
    fields = {}
    for name, values in channels.items():
        if len(values):
            q25,q50,q75 = np.quantile(values,(.25,.5,.75))
            values = (float(values.mean()),float(values.std()),float(q50),float(q75-q25))
        else:
            values = (None,)*4
        fields.update({f"{name}_{stat}":v for stat,v in zip(("mean","std","median","iqr"),values,strict=True)})
    for size in (3,5):
        for pol,z in zip(("vv","vh"),post,strict=True):
            t = texture(z,valid,size)
            good = t[np.isfinite(t)]
            group = f"post_{pol}_variance_{size}x{size}_db2"
            fields[f"{group}_mean"] = float(good.mean()) if len(good) else None
            fields[f"{group}_q90"] = float(np.quantile(good,.90)) if len(good) else None
    return fields


def patch_features(args, raw, manifest, logits, patches):
    """Read physical inputs without annotation masks and attach patch geometry."""
    sar_hash = digest(args.sar_cache/"manifest.json")
    extra_hash = digest(args.extra_cache/"manifest.json")
    identity = dict(version=FEATURE_VERSION,split="validation",mean_cache_sha256=digest(args.cache/"manifest.json"),
        patches_sha256=digest(args.spatial_results/"patch_manifest.json"),
        sar_manifest_sha256=sar_hash,extra_manifest_sha256=extra_hash)
    path = args.feature_cache or args.out/"feature_cache.json"
    lookup = {(e,p["patch_id"]):p for e,rows in patches.items() for p in rows}
    if not path.exists():
        sar = get_sar(raw,logits,args.cache,args.sar_cache)
        extra = supplement(raw,logits,sar,args.cache,args.extra_cache)
        data = KuroDataset(raw,"validation")
        result = []
        for i,(_,row) in enumerate(data.rows.iterrows(),1):
            data.archive.check_unchanged()
            event,identifier = str(row.actid),str(row["tortilla:id"])
            patch = lookup[event,identifier]
            sl = slice(patch["start"],patch["stop"])
            sample = preprocess(*read_sample(raw,row,include_labels=False),include_labels=False)
            if not np.array_equal(sar_features(sample["sar_db"],sample["input_valid"]),sar[event][sl]):
                raise ValueError(f"SAR alignment mismatch at {event}/{identifier}")
            m = flood_margin(logits[event][sl])
            if not np.isfinite(m).all():
                raise ValueError(f"Nonfinite margin at {event}/{identifier}")
            result.append(dict(event_id=event,patch_id=identifier,input_pixels=len(m),
                               **margin_features(m),**radiometry_features(sample,extra[event][sl,1])))
            if i%250==0 or i==len(data):
                print(f"Extracted patch context: {i}/{len(data)}",flush=True)
        path.parent.mkdir(parents=True,exist_ok=True)
        write_json(path,identity | {"rows":result})
    value = load_json(path)
    if any(value[k]!=v for k,v in identity.items()):
        raise ValueError("Feature cache provenance does not match this validation cache")
    keyed = {(r["event_id"],r["patch_id"]):r for r in value["rows"]}
    if len(keyed)!=len(value["rows"]) or set(keyed)!=set(lookup):
        raise ValueError("Patch context does not cover the validation patch manifest")
    for key, patch in lookup.items():
        if keyed[key]["input_pixels"]!=patch["stop"]-patch["start"]:
            raise ValueError(f"Feature cache has incorrect patch size: {key}")
    rows = {e:[keyed[e,p["patch_id"]] for p in patches[e]] for e in manifest["event_counts"]}
    return rows, path


def count_curve(margin, labels, grid):
    if not np.isin(labels,[0,1,2,IGNORE]).all():
        raise ValueError("Unexpected cached target label")
    keep = labels!=IGNORE
    bins = np.searchsorted(grid,-margin[keep].astype(np.float64),side="right")
    positive = labels[keep]==FLOOD
    width = len(grid)+1
    tp = np.bincount(bins[positive],minlength=width).cumsum()[:-1]
    fp = np.bincount(bins[~positive],minlength=width).cumsum()[:-1]
    return np.stack((tp,fp,int(positive.sum())-tp),axis=1).astype(np.int32)


def source_curves(cache, manifest, logits, grid, event, patches):
    y = cache_file(cache,manifest,event,labels=True)
    values = []
    for patch in patches:
        sl = slice(patch["start"],patch["stop"])
        values.append(count_curve(flood_margin(logits[sl]),y[sl],grid))
    return np.stack(values)


def features_matrix(rows, names):
    return np.array([[np.nan if r[k] is None else r[k] for k in names] for r in rows],
                    dtype=np.float64).reshape(len(rows),len(names))


def design(source_rows, target_rows, names):
    source = features_matrix(source_rows,names)
    target = features_matrix(target_rows,names)
    stats = []
    for j,name in enumerate(names):
        seen = source[np.isfinite(source[:,j]),j]
        median = float(np.median(seen)) if len(seen) else 0.
        # A feature absent on all sources carries no information in this fold.
        if not len(seen):
            target[:,j] = median
        for data in (source,target):
            data[~np.isfinite(data[:,j]),j] = median
        mean = float(source[:,j].mean())
        scale = float(source[:,j].std())
        if scale<1e-10:
            scale = 1.
            target[:,j] = mean
        for data in (source,target):
            data[:,j] = np.clip((data[:,j]-mean)/scale,-Z_LIMIT,Z_LIMIT)
        stats.append(dict(feature=name,median=median,mean=mean,std_used=scale))
    return np.column_stack((np.ones(len(source)),source)),np.column_stack((np.ones(len(target)),target)),stats


def event_iou(tp,fp,positives):
    union = fp+positives
    return np.divide(tp,union,out=np.zeros_like(tp,dtype=np.float64),where=union>0)


def soft_objective(theta, x, grid, tp, fp, event_idx, totals, anchor):
    bias = anchor+x@theta
    log_weights = (bias[:,None]*grid[None,:]-.5*grid[None,:]**2)/SIGMA**2
    weights = np.exp(log_weights-log_weights.max(axis=1,keepdims=True))
    weights /= weights.sum(axis=1,keepdims=True)
    tp_soft = np.einsum("pg,pg->p",weights,tp)
    fp_soft = np.einsum("pg,pg->p",weights,fp)
    slope = weights*(grid[None,:]-(weights@grid)[:,None])/SIGMA**2
    dt = np.einsum("pg,pg->p",slope,tp)
    df = np.einsum("pg,pg->p",slope,fp)
    et = np.bincount(event_idx,weights=tp_soft,minlength=len(totals))
    ef = np.bincount(event_idx,weights=fp_soft,minlength=len(totals))
    union = totals+ef
    patch_union = union[event_idx]
    per_patch_gradient = np.divide(dt*patch_union-df*et[event_idx],patch_union**2,
                                   out=np.zeros_like(dt),where=patch_union>0)
    value = -event_iou(et,ef,totals).mean()+.5*RIDGE*(theta@theta)
    gradient = -x.T@per_patch_gradient/len(totals)+RIDGE*theta
    return float(value),gradient


def selected_indices(bias, grid):
    return np.abs(bias[:,None]-grid[None,:]).argmin(axis=1)


def fit_variant(source_rows, target_rows, names, curves, event_idx, grid, anchor):
    src,tar,scaler = design(source_rows,target_rows,names)
    tp,fp,fn = np.moveaxis(curves,2,0)
    totals = np.bincount(event_idx,weights=tp[:,0]+fn[:,0])
    if not totals.sum():
        raise ValueError("Source objective needs flood pixels")
    result = minimize(soft_objective,np.zeros(src.shape[1]),jac=True,method="L-BFGS-B",
                      args=(src,grid,tp,fp,event_idx,totals,anchor),
                      options={"maxiter":500,"ftol":1e-11,"gtol":1e-7})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Source-only optimization failed: {result.message}")
    chosen = selected_indices(anchor+tar@result.x,grid)
    # Empty input patches have no decisions to adjust.
    for i,row in enumerate(target_rows):
        if not row["input_pixels"]:
            chosen[i] = selected_indices(np.array([anchor]),grid)[0]
    src_choice = selected_indices(anchor+src@result.x,grid)
    src_counts = curves[np.arange(len(src)),src_choice]
    src_tp = np.bincount(event_idx,weights=src_counts[:,0])
    src_fp = np.bincount(event_idx,weights=src_counts[:,1])
    return chosen,dict(coefficients=result.x.tolist(),feature_names=["intercept",*names],
                       source_feature_scaler=scaler,iterations=int(result.nit),
                       objective=float(result.fun),
                       source_hard_macro_iou=float(event_iou(src_tp,src_fp,totals).mean()))


def fit_folds(args, manifest, logits, patches, features, grid, anchor):
    events = sorted(patches,key=int)
    names = [k for k in features[events[0]][0] if k not in ("event_id","patch_id","input_pixels")]
    margin_names = [k for k in names if k.startswith(("margin_","prevalence_"))]
    curves = {}
    locks = {}
    for held_out in events:
        source = [e for e in events if e != held_out]
        for e in source:
            if e not in curves:
                curves[e] = source_curves(args.cache,manifest,logits[e],grid,e,patches[e])
        rows = [r for e in source for r in features[e]]
        target_rows = features[held_out]
        event_idx = np.repeat(np.arange(len(source)),[len(features[e]) for e in source])
        counts = np.concatenate([curves[e] for e in source])
        fold = dict(source_events=source,held_out=held_out,methods={})
        for method, columns in ((MARGIN,margin_names),(SAR,names)):
            selected,fit = fit_variant(rows,target_rows,columns,counts,event_idx,grid,anchor)
            bias = grid[selected]
            fold["methods"][method] = dict(fit=fit,patch_biases={
                r["patch_id"]:float(value) for r,value in zip(target_rows,bias,strict=True)})
            print(f"Locked {held_out} {method}: median b={np.median(bias):+.2f}, "
                  f"source hard macro IoU={fit['source_hard_macro_iou']:.4f}",flush=True)
        write_json(args.out/f"lock_{held_out}.json",fold)
        locks[held_out] = fold
    write_json(args.out/"bias_lock.json",dict(split="validation",features=FEATURE_VERSION,
        anchor=anchor,candidate_grid=grid.tolist(),folds=locks))
    np.savez_compressed(args.out/"patch_bias_curves.npz",biases=grid,
        event_ids=np.array([e for e in events for _ in patches[e]]),
        patch_ids=np.array([p["patch_id"] for e in events for p in patches[e]]),
        count_names=np.array(["tp","fp","fn"]),counts=np.concatenate([curves[e] for e in events]))


def flood_counts(cm):
    return int(cm[2,2]),int(cm[:2,2].sum()),int(cm[2,:2].sum())


def counts_metrics(tp,fp,fn):
    return dict(flood_iou=tp/max(1,tp+fp+fn),flood_f1=2*tp/max(1,2*tp+fp+fn),
                flood_precision=tp/max(1,tp+fp),flood_recall=tp/max(1,tp+fn))


def evaluate(args, recipe, manifest, logits, patches):
    previous, spatial, _ = load_spatial(args.spatial_results,args.cache,recipe,manifest)
    if set(previous)!=set(patches):
        raise ValueError("Spatial Oracle validation events differ from the patch manifest")
    events, iw_rows, _ = load_artifacts(args.iw_results,args.bias_results,
                              VAL_ARTIFACT/"validation_confusion_eoa_flip_tta.npz")
    iw = {(r["event_id"],r["method"]):r for r in iw_rows}
    event_rows, patch_rows = [], []
    for e in events:
        fold = load_json(args.out/f"lock_{e}.json")
        if fold["held_out"]!=e or e in fold["source_events"] or len(fold["source_events"])!=5:
            raise ValueError(f"Invalid source events in locked fold {e}")
        y = cache_file(args.cache,manifest,e,labels=True)
        oracle_iou = float(spatial[e,EVENT]["flood_iou"])
        patch_iou = float(spatial[e,PATCH]["flood_iou"])
        current = {m:np.zeros((3,3),dtype=np.int64) for m in VARIANTS}
        for patch in patches[e]:
            sl = slice(patch["start"],patch["stop"])
            y0 = y[sl]
            ok = y0!=IGNORE
            label = y0[ok]
            z = logits[e][sl][ok]
            for m in VARIANTS:
                bias = fold["methods"][m]["patch_biases"][patch["patch_id"]]
                cm = predict_confusion(z,label,bias)
                current[m] += cm
                tp,fp,fn = flood_counts(cm)
                patch_rows.append(dict(event_id=e,patch_id=patch["patch_id"],method=m,bias=bias,
                    scored_pixels=len(label),tp=tp,fp=fp,fn=fn))
        for m in METHODS:
            if m in VARIANTS:
                tp,fp,fn = flood_counts(current[m])
                miou = scores(current[m])["pooled_miou"]
            elif m in (IW_MARGIN,IW_SAR):
                row = iw[e,m]
                tp,fp,fn = (row[k] for k in ("tp","fp","fn"))
                miou = None
            else:
                row = spatial[e,m]
                tp,fp,fn = (int(row[k]) for k in ("tp","fp","fn"))
                miou = float(row["pooled_miou"])
            checked(tp+fn,int(spatial[e,EVENT]["true_flood_pixels"]),f"{e}/{m} scored flood support")
            metrics = counts_metrics(tp,fp,fn)
            event_rows.append(dict(event_id=e,method=m,tp=tp,fp=fp,fn=fn,event_miou=miou,
                **metrics,event_oracle_gap_pp=100*(oracle_iou-metrics["flood_iou"]),
                patch_oracle_gap_pp=100*(patch_iou-metrics["flood_iou"])))
    return event_rows,patch_rows


def summarize(rows):
    result = []
    for method in METHODS:
        subset = [r for r in rows if r["method"]==method]
        counts = [sum(r[k] for r in subset) for k in ("tp","fp","fn")]
        ious = [r["flood_iou"] for r in subset]
        worst = min(subset,key=lambda r:r["flood_iou"])
        result.append(dict(method=method,events=len(ious),tp=counts[0],fp=counts[1],fn=counts[2],
             **{f"pooled_{k}":v for k,v in counts_metrics(*counts).items()},
             event_macro_flood_iou=float(np.mean(ious)),event_iou_std=float(np.std(ious)),
             worst_event_iou=worst["flood_iou"],worst_event_id=worst["event_id"],
             mean_event_oracle_gap_pp=float(np.mean([r["event_oracle_gap_pp"] for r in subset])),
             mean_patch_oracle_gap_pp=float(np.mean([r["patch_oracle_gap_pp"] for r in subset]))))
    return result


def plot(out, rows):
    events = sorted({r["event_id"] for r in rows},key=int)
    lookup = {(r["method"],r["event_id"]):r["flood_iou"] for r in rows}
    names = ("Global bias","IW margin","IW margin + SAR","Local margin","Local margin + SAR",
             "Event Oracle","Patch Oracle")
    fig,(ax,bx) = plt.subplots(1,2,figsize=(12,4.4),layout="constrained")
    for method,name in zip(METHODS,names,strict=True):
        values = [100*lookup[method,e] for e in events]
        if method in (EVENT,PATCH):
            ax.plot(events,values,"s--",lw=1,label=name)
        else:
            ax.plot(events,values,"o-",lw=1,label=name)
    ax.set(xlabel="Held-out validation event",ylabel="Flood IoU (%)")
    ax.legend(fontsize=8,ncols=2)
    indices = np.arange(len(events))
    for i,(method,name) in enumerate(zip(VARIANTS,names[3:5],strict=True)):
        bx.bar(indices+(.4*i-.2),[100*(lookup[method,e]-lookup[GLOBAL,e]) for e in events],
               width=.38,label=name)
    bx.axhline(0,color=".55",linestyle="--")
    bx.set(xticks=indices,xticklabels=events,xlabel="Held-out validation event",
           ylabel="Flood IoU difference vs Global Bias (pp)")
    bx.legend(fontsize=8)
    fig.savefig(out/"event_comparison.png",dpi=165)
    plt.close(fig)


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    recipe,selection,grid,expected = reference(load_confusions=False)
    manifest, logits = open_predictions(args.cache,recipe,selection,sorted(expected,key=int))
    spatial_config = load_json(args.spatial_results/"evaluation_config.json")
    patch_path = args.spatial_results/"patch_manifest.json"
    if (spatial_config["split"]!="validation" or spatial_config["mean_cache_sha256"]!=digest(args.cache/"manifest.json")
            or spatial_config["checkpoint_selection_sha256"]!=recipe["selection_sha256"]
            or spatial_config["patch_manifest_sha256"]!=digest(patch_path)):
        raise ValueError("Patch manifest does not match the frozen validation predictions")
    raw = selection["protocol"]["training"]["raw"]
    patches = patch_ranges(raw,logits,None,patch_path,digest(args.cache/"manifest.json"),digest(args.sar_cache/"manifest.json"))
    args.out.mkdir(parents=True,exist_ok=False)
    features, feature_path = patch_features(args,raw,manifest,logits,patches)
    save_csv(args.out/"patch_features.csv",[r for rows in features.values() for r in rows])
    fit_folds(args,manifest,logits,patches,features,grid,BIAS_ANCHOR)
    rows, patch_rows = evaluate(args,recipe,manifest,logits,patches)
    save_csv(args.out/"summary.csv",summarize(rows))
    save_csv(args.out/"event_metrics.csv",rows)
    save_csv(args.out/"event_279.csv",[r for r in rows if r["event_id"]=="279"])
    save_csv(args.out/"patch_predictions.csv",patch_rows)
    plot(args.out,rows)
    write_json(args.out/"evaluation_config.json",dict(
        split="validation",scope="exploratory method-level leave-one-event-out on frozen EoA",
        checkpoint_selection_sha256=recipe["selection_sha256"],
        factual_cache_sha256=digest(args.cache/"manifest.json"),patch_manifest_sha256=digest(patch_path),
        feature_cache=str(feature_path.resolve()),feature_cache_sha256=digest(feature_path),
        bias_lock_sha256=digest(args.out/"bias_lock.json"),bias_curves_sha256=digest(args.out/"patch_bias_curves.npz"),
        model="bounded-at-deployment affine bias anchored at zero; fractional counts from soft candidates",
        bias_anchor=BIAS_ANCHOR,
        objective="negative equal-event mean of source TP/(true Flood + FP), plus L2 penalty",
        ridge=RIDGE,gaussian_sigma=SIGMA,standardized_feature_clip=Z_LIMIT,
        bias_deployment="nearest point on historical bias grid, smaller bias for exact tie",
        source_preprocessing="fit imputation and standardization on source patch features only",
        comparator_paths={name:str(path.resolve()) for name,path in (
            ("spatial",args.spatial_results),("iw_iou",args.iw_results),("historical_bias",args.bias_results))},
        comparator_event_metrics_sha256={name:digest(path/"event_metrics.csv") for name,path in (
            ("spatial",args.spatial_results),("iw_iou",args.iw_results),("historical_bias",args.bias_results))},
        caveat="frozen EoA checkpoints and fixed global bias were historically selected using these six events",
        event_std_ddof=0,code_sha256=digest(Path(__file__))))
    print(f"Saved Context-Conditioned Local Logit Adjustment: {args.out}",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache","sar-cache","extra-cache","spatial-results","iw-results","bias-results","out"):
        parser.add_argument(f"--{name}",required=True,type=Path)
    parser.add_argument("--feature-cache",type=Path,help="Read or create a provenance-checked unlabeled feature cache")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
