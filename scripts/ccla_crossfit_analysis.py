"""Fit CCLA from cross-fitted training predictions and evaluate locked single-SMA comparators."""
import numpy as np
from scipy.special import expit

from adapt_unet_event_bias import IGNORE, digest, flood_margin, load_json, oracle_from_ranks, scores, write_json
from adapt_unet_iw_iou import OVERLAP, draw, feature_rows, fit_domain, log_ratio, weighted_source_curve
from adapt_unet_local_context import BIAS_ANCHOR, FEATURE_VERSION, RIDGE, SIGMA, count_curve, design, fit_variant, selected_indices
from audit_unet_bias_failures import checked, save_csv
from bootstrap_unet_events import Case, ITERATIONS, SEED, event_statistics, interval
from ccla_crossfit_cache import lookup, values
from ccla_crossfit_protocol import GLOBAL_BIAS
from diagnose_unet_spatial_oracle import apply_groups, average_precision, group, predicted_confusion, shifted_rank
from evaluate_unet_eoa_diagnostics import rank_event, rank_pooled
from evaluate_unet_eoa_flip_tta import BIASES
from unet_conservative_expected_iou import best_index

NOBIAS, GLOBAL, REF = "no_bias", "historical_global_bias", "oof_source_reference"
IW_MARGIN, IW_SAR = "iw_margin", "iw_margin_sar"
MARGIN, SAR = "ccla_margin", "ccla_margin_sar"
EVENT, PATCH = "event_oracle", "patch_oracle"
METHODS = (NOBIAS,GLOBAL,REF,IW_MARGIN,IW_SAR,MARGIN,SAR,EVENT,PATCH)
IW_METHODS = ((IW_MARGIN,1),(IW_SAR,7))
VARIANTS = (MARGIN,SAR)
FEATURE_META = {"event_id","patch_id","input_pixels","aoi_id","crs","affine","start","stop","mask_index"}
GRID = BIASES


def flat_patches(patches):
    return [p for e in sorted(patches,key=int) for p in patches[e]]


def arrays(index, fields):
    return [{e:values(index,e,k) for e in index} for k in fields]


def fit(oof, out, identity):
    if out.exists():
        raise FileExistsError(out)
    index, patches = lookup(oof)
    rows = flat_patches(patches)
    x,y = arrays(index,("logits","labels"))
    events = sorted(patches,key=int)
    event_idx = np.repeat(np.arange(len(events)),[len(patches[e]) for e in events])
    if len(rows)!=4000 or len(events)!=27:
        raise ValueError("Expected 4000 train patches in 27 events")
    counts = []
    for p in rows:
        e,sl = p["event_id"],slice(p["start"],p["stop"])
        counts.append(count_curve(flood_margin(x[e][sl]),y[e][sl],GRID))
    counts = np.stack(counts)
    event_curves = np.stack([counts[event_idx==i].sum(axis=0) for i in range(len(events))])
    tp,fp,fn = np.moveaxis(event_curves,2,0)
    macro = (tp/np.maximum(tp+fp+fn,1)).mean(axis=0)
    ref = float(GRID[best_index(macro,GRID)])
    names = [key for key in rows[0] if key not in FEATURE_META]
    margin_names = [key for key in names if key.startswith(("margin_","prevalence_"))]
    if len(names)!=62 or len(margin_names)!=14:
        raise ValueError(f"Unexpected frozen CCLA feature layout: {len(names)} and {len(margin_names)}")
    models = {}
    for method,columns in ((MARGIN,margin_names),(SAR,names)):
        _,model = fit_variant(rows,[],columns,counts,event_idx,GRID,BIAS_ANCHOR)
        models[method] = model
        print(f"Fit {method}: OOF hard macro IoU={model['source_hard_macro_iou']:.5f}",flush=True)
    out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/"oof_patch_curves.npz",biases=GRID,counts=counts,count_names=["tp","fp","fn"],
                        patch_ids=[p["patch_id"] for p in rows],event_ids=[p["event_id"] for p in rows])
    save_csv(out/"source_reference_curve.csv",[dict(bias=b,macro_flood_iou=s) for b,s in zip(GRID,macro,strict=True)])
    write_json(out/"predictor.json",identity | dict(
        training="27 official train events; one OOF prediction per patch; no validation labels or features",
        feature_version=FEATURE_VERSION,ridge=RIDGE,gaussian_sigma=SIGMA,
        objective="equal-event macro Flood IoU on soft candidate TP/FP plus L2 penalty",
        anchor=BIAS_ANCHOR,reference_bias=ref,models=models,candidate_biases=GRID.tolist(),
        curve_sha256=digest(out/"oof_patch_curves.npz")))


def apply_predictor(source, target, model):
    keys = model["feature_names"][1:]
    _, values_, preprocessing = design(source,target,keys)
    if preprocessing!=model["source_feature_scaler"]:
        raise ValueError("OOF feature scaler differs from the locked predictor")
    bias = GRID[selected_indices(BIAS_ANCHOR+values_@np.array(model["coefficients"]),GRID)]
    return np.where([r["input_pixels"]>0 for r in target],bias,BIAS_ANCHOR)


def lock_biases(oof, validation, predictor):
    src_index, src_patches = lookup(oof)
    val_index, val_patches = lookup([validation])
    source = sorted(src_index,key=int)
    rows = flat_patches(src_patches)
    sx,ys,ss,se = arrays(src_index,("logits","labels","sar","extra"))
    # Validation accessor deliberately excludes the target label file.
    vx,vs,ve = arrays(val_index,("logits","sar","extra"))

    def source_features(event, indices):
        return feature_rows(sx,ss,se,event,indices)

    locks = {}
    for target in sorted(val_index,key=int):
        target_x = feature_rows(vx,vs,ve,target,draw(len(vx[target]),target,target=target))
        source_sizes,source_x = {},[]
        for event in source:
            good = np.flatnonzero(ys[event]!=IGNORE)
            source_sizes[event] = len(good)
            if not len(good):
                raise ValueError(f"No scored OOF source pixels in {event}")
            source_x.append(source_features(event,draw(good,event,target=target)))
        locked = dict(sources=source,scalar_biases={NOBIAS:0.,GLOBAL:GLOBAL_BIAS,REF:predictor["reference_bias"]},
                      patch_biases={},iw_diagnostics={})
        for method,width in IW_METHODS:
            model = fit_domain(source_x,target_x,width)
            p = expit(log_ratio(target_x,model))
            counts, diag = weighted_source_curve(source,ys,source_sizes,source_features,model,GRID)
            locked["scalar_biases"][method] = float(GRID[best_index(counts[-1],GRID)])
            locked["iw_diagnostics"][method] = diag | model | dict(
                target_overlap_fraction=float(np.mean((p>=OVERLAP[0])&(p<=OVERLAP[1]))))
            print(f"IW {method} event {target}: b={locked['scalar_biases'][method]:+.2f}",flush=True)
        for method in VARIANTS:
            bias = apply_predictor(rows,val_patches[target],predictor["models"][method])
            locked["patch_biases"][method] = {p["patch_id"]:float(b) for p,b in zip(val_patches[target],bias,strict=True)}
        locks[target] = locked
        print(f"Locked validation event {target} (without its labels)",flush=True)
    return locks,val_index,val_patches,vx


def score_event(event, x, labels, patches, lock):
    z = x[labels!=IGNORE]
    y = labels[labels!=IGNORE]
    if not len(y) or not np.isin(y,[0,1,2]).all():
        raise ValueError(f"Missing or invalid scored pixels in {event}")
    margin = flood_margin(z)
    other = z[:,:2].argmax(1)
    del z
    gt = y==2
    raw = rank_event([(margin,gt)])
    event_bias,event_iou = oracle_from_ranks(raw) if gt.any() else (0.,0.)
    scalar = lock["scalar_biases"] | {EVENT:event_bias}
    cms,ranks = {},{}
    for method,bias in scalar.items():
        cms[method] = predicted_confusion(margin,other,y,bias)
        ranks[method] = rank_pooled([shifted_rank(raw,bias)])
    checked(scores(cms[EVENT])["flood_iou"],event_iou,f"{event} event oracle")
    groups = []
    offset = 0
    for p in patches:
        n = int(np.count_nonzero(labels[p["start"]:p["stop"]]!=IGNORE))
        groups.append(group(p["patch_id"],slice(offset,offset+n),margin,gt,p["input_pixels"]))
        offset += n
    if offset!=len(y):
        raise ValueError("Incomplete validation patch coverage")
    cms[PATCH],ranks[PATCH],patch_oracles = apply_groups(event,PATCH,groups,margin,other,y)
    for method in VARIANTS:
        cms[method] = np.zeros((3,3),np.int64)
        adjusted = []
        for g in groups:
            if g.rank is not None:
                idx,bias = g.selector,lock["patch_biases"][method][g.identifier]
                cms[method] += predicted_confusion(margin[idx],other[idx],y[idx],bias)
                adjusted.append(shifted_rank(g.rank,bias))
        ranks[method] = rank_pooled(adjusted)
    truth = np.bincount(y,minlength=3)
    patch_iou = scores(cms[PATCH])["flood_iou"]
    rows = []
    for method in METHODS:
        cm = cms[method]
        np.testing.assert_array_equal(cm.sum(axis=1),truth)
        s = scores(cm)
        rows.append(dict(event_id=event,method=method,**s,ap=average_precision(ranks[method]),
                         raw_ap=average_precision(raw),bias=scalar.get(method),
                         tp=int(cm[2,2]),fp=int(cm[:2,2].sum()),fn=int(cm[2,:2].sum()),
                         event_oracle_gap_pp=100*(event_iou-s["flood_iou"]),
                         patch_oracle_gap_pp=100*(patch_iou-s["flood_iou"])))
    return cms,ranks,rows,patch_oracles


def metrics_and_bootstrap(cms):
    events = list(cms)
    n = len(events)
    picks = np.random.default_rng(SEED).integers(n,size=(ITERATIONS,n))
    draws = (picks[...,None]==np.arange(n)).sum(axis=1)
    estim, point = {},{}
    for method in METHODS:
        matrices = np.stack([cms[e][method] for e in events])
        tp = matrices[:,2,2]
        flood = np.column_stack((tp,matrices[:,:2,2].sum(1),matrices[:,2,:2].sum(1)))
        case = Case(method,{},"",flood=flood[None],multiclass=matrices[None])
        vals = tp/np.maximum(flood.sum(1),1)

        def stats(weights):
            s = event_statistics(case,weights)
            s["worst_event_iou"] = np.where(weights>0,vals[None],np.inf).min(1)
            return s
        estim[method] = stats(draws)
        point[method] = {k:float(v[0]) for k,v in stats(np.ones((1,n),dtype=int)).items()}
    confidence, paired = [],[]
    for m in METHODS:
        for metric in point[m]:
            lo,hi,valid = interval(estim[m][metric])
            confidence.append(dict(method=m,metric=metric,estimate=point[m][metric],
                                   ci_low=lo,ci_high=hi,iterations=valid))
    for method in VARIANTS:
        for ref in (NOBIAS,GLOBAL,REF,IW_MARGIN,IW_SAR,EVENT,PATCH):
            for metric in point[method]:
                lo,hi,valid = interval(100*(estim[method][metric]-estim[ref][metric]))
                paired.append(dict(method=method,reference=ref,metric=metric,
                    change_pp=100*(point[method][metric]-point[ref][metric]),ci_low_pp=lo,ci_high_pp=hi,iterations=valid))
    return point,confidence,paired


def evaluate(oof, validation, predictor, out, identity):
    if out.exists():
        raise FileExistsError(out)
    locks,val_index,patches,x = lock_biases(oof,validation,predictor)
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/"bias_lock.json",identity | dict(events=locks,
        source_reference="event-macro Flood IoU on 27 OOF training events",
        iw="27 OOF training events with labels; one validation event with no labels",
        historical_global_bias=GLOBAL_BIAS))
    locks = load_json(out/"bias_lock.json")["events"]
    cms, by_rank, rows, oracle_rows = {},{m:[] for m in METHODS},[],[]
    for event,lock in locks.items():
        # First read of validation labels by the analysis is after the bias lock.
        y = values(val_index,event,"labels")
        cms[event],ranks,metrics,oracle = score_event(event,x[event],y,patches[event],lock)
        for m in METHODS:
            by_rank[m].append(ranks[m])
        rows.extend(metrics)
        oracle_rows.extend(oracle)
    point,ci,paired = metrics_and_bootstrap(cms)
    summary = []
    for m in METHODS:
        sub = [r for r in rows if r["method"]==m]
        summary.append(dict(method=m,**point[m],pooled_ap=average_precision(rank_pooled(by_rank.pop(m))),
                            worst_event=min(sub,key=lambda r:r["flood_iou"])["event_id"],
                            mean_event_oracle_gap_pp=np.mean([r["event_oracle_gap_pp"] for r in sub]),
                            mean_patch_oracle_gap_pp=np.mean([r["patch_oracle_gap_pp"] for r in sub])))
    for name, data in (("summary",summary),("event_metrics",rows),("patch_oracle_biases",oracle_rows),
                       ("bootstrap",ci),("paired_bootstrap",paired)):
        save_csv(out/f"{name}.csv",data)
    write_json(out/"evaluation_config.json",identity | dict(
        validation="exact audited 959 patches, six events, single seed 42 epoch-20 SMA + Flip TTA",
        oracle="ground-truth upper bounds, event scalar and joint event-IoU patch optimizer",
        ap="exact pooled AP on adjusted flood margins; all no-bias event AP values are also saved; AP point estimates",
        bootstrapping="paired draws of six whole events, 10000 percentile replicates, seed 42; no AP bootstrap",
        selection="CCLA and Source Reference fit on OOF training only; IW uses unlabeled validation covariates",
        caveat="Global Bias +0.95 is the historical EoA calibration, carried over as a comparator",
        bias_lock_sha256=digest(out/"bias_lock.json")))
    print(f"Saved held-out validation metrics: {out}",flush=True)
