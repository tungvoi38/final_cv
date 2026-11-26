#!/usr/bin/env python3
"""Event-disjoint OOF benchmark of patchwise Potts graph cuts on frozen EoA logits."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
from itertools import product
import multiprocessing as mp
import os
from pathlib import Path
import unittest

import numpy as np

from adapt_unet_event_bias import IGNORE, digest, flood_margin, load_json, scores, write_json
from adapt_unet_local_context import BIAS_ANCHOR, fit_variant
from audit_ccla_scalar_bias import F1, event_matrices, lock_oof, stats
from audit_unet_bias_failures import checked, read_csv, save_csv
from bootstrap_unet_events import ITERATIONS, SEED, interval
from ccla_crossfit_analysis import EVENT, GLOBAL, IW_MARGIN, IW_SAR, MARGIN, PATCH, REF
from ccla_crossfit_cache import PACKED_BYTES, cache_path, lookup, read_manifest, values
from unet_conservative_expected_iou import best_index

LAMBDAS = (0., .1, .25, .5, 1., 2.)
SIGMAS = (.5,1.,2.)
INITS = (F1,MARGIN)
FAMILIES = ("uniform","sar_guided")
ROBUST = "robust_ccla_margin"
METHODS_OLD = (GLOBAL,F1,REF,IW_MARGIN,IW_SAR,MARGIN,ROBUST,EVENT,PATCH)
FIELDS = ("post_vv_db","post_vh_db","delta_vv_db","delta_vh_db","slope_deg")
TOL = 1e-12
MAX_DROP = .005
SHAPE = (224,224)
RIGHT = np.array([[0,0,0],[0,0,1],[0,0,0]])
DOWN = np.array([[0,0,0],[0,0,0],[0,1,0]])
BLOCK = 16
WORKER = {}


def maxflow_module():
    try:
        import maxflow
    except ImportError as exc:
        raise RuntimeError("Install PyMaxflow==1.3.2 in the project .venv") from exc
    return maxflow


def edge_weights(valid, features=None, sigma=None):
    right = np.zeros(valid.shape,dtype=float)
    down = np.zeros(valid.shape,dtype=float)
    vr = valid[:,:-1]&valid[:,1:]
    vd = valid[:-1,:]&valid[1:,:]
    if features is None:
        right[:,:-1] = vr
        down[:-1,:] = vd
    else:
        def similarity(a,b):
            shared = np.isfinite(a)&np.isfinite(b)
            component = np.where(shared,a-b,0)
            distance = np.sum(component**2,axis=-1)*len(FIELDS)/np.maximum(shared.sum(axis=-1),1)
            return np.exp(-distance/(2*sigma**2))
        right[:,:-1] = np.where(vr,similarity(features[:,:-1],features[:,1:]),0)
        down[:-1,:] = np.where(vd,similarity(features[:-1,:],features[1:,:]),0)
    return right,down


def graph_cut(margin, valid, strength, links=None):
    """Return a flood mask from input geometry and scores only."""
    if strength==0:
        return valid & (margin>0)
    if links is None:
        raise ValueError("Positive lambda requires edge weights")
    graph = maxflow_module().Graph[float](valid.size,2*valid.size)
    nodes = graph.add_grid_nodes(valid.shape)
    # Flood is on the source side. Subtracting min(D0,D1) preserves the softplus energy.
    s = np.where(valid,margin,0)
    graph.add_grid_tedges(nodes,np.maximum(s,0),np.maximum(-s,0))
    for w,structure in zip(links,(RIGHT,DOWN),strict=True):
        graph.add_grid_edges(nodes,strength*w,structure=structure,symmetric=True)
    graph.maxflow()
    return valid & ~graph.get_grid_segments(nodes)


def choices():
    out = []
    for init in INITS:
        out.append(dict(key=f"{init}:zero",init=init,family="none",strength=0.,sigma=None))
        for family in FAMILIES:
            for strength,sigma in product(LAMBDAS[1:],SIGMAS if family=="sar_guided" else (None,)):
                out.append(dict(key=f"{init}:{family}:{strength:g}:{sigma}",init=init,
                                family=family,strength=strength,sigma=sigma))
    return out


def fpr_counts(c):
    return int(c[2,2]),int(c[:2,2].sum()),int(c[2,:2].sum())


def per_event_metric(cm):
    ious = np.array([scores(c)["flood_iou"] for c in cm])
    return scores(cm.sum(axis=0)) | dict(
        event_macro_flood_iou=float(ious.mean()),event_flood_iou_std=float(ious.std()),
        worst_event_iou=float(ious.min()),lower_tail_iou=float(np.sort(ious)[:(len(ious)+2)//3].mean()))


def initialize_worker(files):
    for e,pack in files.items():
        path,n = Path(pack[0]),pack[1]
        WORKER[e] = {f:np.memmap(cache_path(path,e,f),mode="r",dtype=dt,shape=(n,w) if w>1 else (n,))
                     for f,(dt,w) in (("logits",("<f4",3)),("labels",("u1",1)),
                                     ("sar",("<f4",3)),("extra",("<f4",2)))}
        WORKER[e]["masks"] = np.memmap(cache_path(path,e,"masks"),mode="r",dtype="u1").reshape(-1,2,PACKED_BYTES)


def patch_inputs(p, scale, params):
    data = WORKER[p["event_id"]]
    sl = slice(p["start"],p["stop"])
    valid = np.unpackbits(data["masks"][p["mask_index"],0],bitorder="little").reshape(SHAPE).astype(bool)
    z = data["logits"][sl]
    if int(valid.sum())!=len(z) or not np.isfinite(z).all():
        raise ValueError(f"Invalid mask or logits for {p['patch_id']}")
    base = np.zeros(SHAPE,dtype=float)
    base[valid] = flood_margin(z)
    links = {}
    if any(q["family"]=="uniform" and q["strength"] for q in params):
        links["uniform",None] = edge_weights(valid)
    sigmas = sorted({q["sigma"] for q in params if q["family"]=="sar_guided" and q["strength"]})
    if sigmas:
        vv,dvv,dvh = data["sar"][sl].T
        vh,slope = data["extra"][sl].T
        feature = np.full((*SHAPE,len(FIELDS)),np.nan)
        feature[valid] = np.column_stack((vv,vh,dvv,dvh,slope))/scale
        for s in sigmas:
            links["sar_guided",s] = edge_weights(valid,feature,s)
    return data,sl,valid,z,base,links


def task_worker(task):
    path,patches,scale,params = task
    if path.exists():
        return str(path)
    matrices = np.zeros((len(patches),len(params),3,3),dtype=np.int32)
    for i,p in enumerate(patches):
        data,sl,valid,z,base,links = patch_inputs(p,scale,params)
        predictions = []
        for q in params:
            s = base+p["biases"][q["init"]]
            predictions.append(graph_cut(s,valid,q["strength"],links.get((q["family"],q["sigma"])))[valid])
        # Labels enter only the metric scorer, after all candidate masks are computed.
        y = data["labels"][sl]
        scored = y!=IGNORE
        if not np.isin(y,[0,1,2,IGNORE]).all():
            raise ValueError("Unexpected graph-cut label")
        other = z[:,:2].argmax(1)
        for j,flood in enumerate(predictions):
            pred = np.where(flood,2,other)
            matrices[i,j] = np.bincount(3*y[scored].astype(np.int64)+pred[scored],minlength=9).reshape(3,3)
    tmp = path.with_suffix(".partial")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream,patch_ids=[p["patch_id"] for p in patches],keys=[q["key"] for q in params],cm=matrices)
    os.replace(tmp,path)
    return str(path)


def apply_tasks(out, index, patches, scales, params, workers):
    out.mkdir(parents=True,exist_ok=True)
    files = {e:(str(d[0]),d[2]) for e,d in index.items()}
    tasks = []
    for fold,scale in scales.items():
        ordered = [p for p in patches if p["eval_fold"]==fold]
        for i in range(0,len(ordered),BLOCK):
            tasks.append((out/f"{fold}-{i//BLOCK:04}.npz",ordered[i:i+BLOCK],scale,params))
    pending = [t for t in tasks if not t[0].exists()]
    print(f"{out.name}: {len(tasks)-len(pending)}/{len(tasks)} cached tasks, {len(pending)} pending",flush=True)
    if pending:
        if workers==1:
            initialize_worker(files)
            for t in pending:
                task_worker(t)
        else:
            with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context("fork"),
                                     initializer=initialize_worker,initargs=(files,)) as pool:
                for i,_ in enumerate(pool.map(task_worker,pending),1):
                    if i%10==0 or i==len(pending):
                        print(f"{out.name}: {i}/{len(pending)} new tasks",flush=True)
    aggregated = {}
    for path,ps,_,cfg in tasks:
        with np.load(path,allow_pickle=False) as packet:
            if (packet["patch_ids"].tolist()!=[p["patch_id"] for p in ps] or packet["keys"].tolist()!=[q["key"] for q in cfg]
                    or packet["cm"].shape!=(len(ps),len(cfg),3,3)):
                raise ValueError(f"Corrupted or incompatible completed task: {path}")
            for p,cm in zip(ps,packet["cm"],strict=True):
                aggregated[p["patch_id"]] = cm
    if len(aggregated)!=len(patches):
        raise ValueError("Incomplete evaluation patch coverage")
    return aggregated


def read_oof(run):
    predictor = load_json(run/"fit/predictor.json")
    packs = []
    fold_by_patch = {}
    for f,sha in predictor["oof_manifest_sha256"].items():
        cache = run/f"fold-{f}/oof"
        if digest(cache/"manifest.json")!=sha:
            raise ValueError(f"Changed OOF cache manifest for {f}")
        m = read_manifest(cache,dict(split="train",fold=int(f),member_seeds=[42,43,44],features_only=False))
        packs.append((cache,m))
        fold_by_patch.update({p["patch_id"]:int(f) for p in m["patches"]})
    with np.load(run/"fit/oof_patch_curves.npz",allow_pickle=False) as p:
        grid,curves,ids,events = (p[k] for k in ("biases","counts","patch_ids","event_ids"))
    index,patches = lookup(packs)
    by_id = {p["patch_id"]:p for ps in patches.values() for p in ps}
    rows = [by_id[key] for key in ids]
    folds = np.array([fold_by_patch[key] for key in ids])
    if (len(rows)!=4000 or len(set(events))!=27 or any(len(set(events[folds==f]))!=9 for f in (1,2,3))
            or any(r["event_id"]!=e for r,e in zip(rows,events,strict=True))
            or any(len(set(folds[events==e]))!=1 for e in set(events))):
        raise ValueError("OOF folds do not cover once per patch with disjoint events")
    return predictor,grid,curves,ids,events,rows,folds,index


def verified_arrays(index):
    return {e:{k:values(index,e,k) for k in ("logits","labels","sar","extra")} for e in index}


def sar_scale(arrays, events):
    pieces = []
    for e in sorted(events,key=int):
        sample = arrays[e]
        n = len(sample["sar"])
        rng = np.random.default_rng(np.random.SeedSequence([SEED,int(e)]))
        indices = rng.choice(n,min(50000,n),replace=False)
        vv,dvv,dvh = sample["sar"][indices].T
        vh,slope = sample["extra"][indices].T
        pieces.append(np.column_stack((vv,vh,dvv,dvh,slope)))
    x = np.concatenate(pieces).astype(np.float64)
    if not np.isfinite(x[:,:4]).all():
        raise ValueError("Invalid source SAR features")
    scale = [float(v) for v in np.nanstd(x,axis=0)]
    return [v if np.isfinite(v) and v>1e-8 else 1. for v in scale]


def setup_meta(out, arrays, predictor,grid,curves,ids,events,rows,folds):
    p = out/"meta_parameters.json"
    if p.is_file():
        return load_json(p)
    names = predictor["models"][MARGIN]["feature_names"][1:]
    all_folds = {}
    for f in (1,2,3):
        src,held = np.flatnonzero(folds!=f),np.flatnonzero(folds==f)
        source_events = sorted(set(events[src]),key=int)
        lookup_ = {e:i for i,e in enumerate(source_events)}
        counts = curves[src].sum(axis=0)
        t,q,n = counts.T
        scalar = float(grid[best_index(2*t/np.maximum(2*t+q+n,1),grid)])
        local,model = fit_variant([rows[i] for i in src],[rows[i] for i in held],names,curves[src],
                                   np.array([lookup_[e] for e in events[src]]),grid,BIAS_ANCHOR)
        all_folds[str(f)] = dict(source_events=source_events,target_events=sorted(set(events[held]),key=int),
            scalar_f1_bias=scalar,sar_scale=sar_scale(arrays,source_events),model=model,
            patch_biases={ids[i]:{F1:scalar,MARGIN:float(grid[b])} for i,b in zip(held,local,strict=True)})
    result = dict(folds=all_folds,all_oof_sar_scale=sar_scale(arrays,set(events)))
    write_json(p,result)
    return result


def group_counts(patches, matrices, params):
    result = {}
    for p in patches:
        e = p["event_id"]
        if e not in result:
            result[e] = np.zeros((len(params),3,3),dtype=np.int64)
        result[e] += matrices[p["patch_id"]]
    return {e:result[e] for e in sorted(result,key=int)}


def choose_candidates(summary):
    bases = {r["init"]:r for r in summary if r["strength"]==0}
    for row in summary:
        base = bases[row["init"]]
        for field in ("flood_f1","event_macro_flood_iou","lower_tail_iou"):
            row[field+"_delta_pp"] = 100*(row[field]-base[field])
        row["eligible"] = (row["strength"]>0 and row["event_macro_flood_iou"]>base["event_macro_flood_iou"]+TOL
            and row["flood_f1"]>=base["flood_f1"]-MAX_DROP-TOL and row["lower_tail_iou"]>=base["lower_tail_iou"]-TOL)
    selected = {}
    for init,family in product(INITS,FAMILIES):
        matching = [r for r in summary if r["init"]==init and r["family"]==family and r["eligible"]]
        key = f"graph_{family}_{'scalar' if init==F1 else 'ccla'}"
        if matching:
            best = max(r["event_macro_flood_iou"] for r in matching)
            row = min((r for r in matching if best-r["event_macro_flood_iou"]<=TOL),
                      key=lambda r:(r["strength"],r["sigma"] or 0))
        else:
            row = bases[init]
        fields = {k:row[k] for k in ("key","init","family","strength","sigma")}
        selected[key] = fields | dict(key=key,meta_candidate=row["key"],fallback=not bool(matching))
    return selected


def run_meta(out,index,rows,ids,events,folds,grid,curves,setup,workers):
    path = out/"graph_cut_lock.json"
    if path.is_file():
        print("Reuse OOF-selected graph configurations",flush=True)
        return load_json(path)
    params = choices()
    ordered = [p | dict(eval_fold=str(f),biases=setup["folds"][str(f)]["patch_biases"][key])
               for p,key,f in zip(rows,ids,folds,strict=True)]
    scales = {f:r["sar_scale"] for f,r in setup["folds"].items()}
    batches = apply_tasks(out/"meta_blocks",index,ordered,scales,params,workers)
    zero = {g["init"]:j for j,g in enumerate(params) if g["strength"]==0}
    for i,p in enumerate(ordered):
        for init,j in zero.items():
            bi, = np.flatnonzero(np.isclose(grid,p["biases"][init],atol=TOL,rtol=0))
            np.testing.assert_array_equal(fpr_counts(batches[p["patch_id"]][j]),curves[i,bi],
                                          err_msg=f"lambda=0 OOF baseline mismatch on {p['patch_id']}")
    mats = group_counts(ordered,batches,params)
    summary,details = [],[]
    for j,q in enumerate(params):
        summary.append(q | per_event_metric(np.stack([c[j] for c in mats.values()])))
        for e,c in mats.items():
            tp,fp,fn = fpr_counts(c[j])
            details.append(dict(key=q["key"],event_id=e,tp=tp,fp=fp,fn=fn,flood_iou=scores(c[j])["flood_iou"]))
    selected = choose_candidates(summary)
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(dir=out) as d:
        tmp = Path(d)
        for name,records in (("meta_cv_summary",summary),("meta_cv_events",details)):
            save_csv(tmp/f"{name}.csv",records)
            os.replace(tmp/f"{name}.csv",out/f"{name}.csv")
    lock = dict(selected=selected,source="event-disjoint meta-CV over all 27 OOF training events",
                metrics_sha256=digest(out/"meta_cv_summary.csv"),
                rule="per family and initialization: improve macro IoU, drop F1 <=0.5 pp, do not drop lower-tail IoU",
                tie="smaller lambda then smaller sigma",meta_parameters_sha256=digest(out/"meta_parameters.json"))
    write_json(path,lock)
    for name,q in selected.items():
        print(f"Locked {name}: lambda={q['strength']:g} sigma={q['sigma']} fallback={q['fallback']}",flush=True)
    return lock


def robust_artifacts(run, root, plan, patches):
    cfg = load_json(root/"analysis_config.json")
    if (Path(cfg["run"]).resolve()!=run.resolve() or cfg["split"]!="validation" or cfg["patches"]!=959
            or cfg["validation_manifest_sha256"]!=plan["validation_manifest_sha256"]
            or cfg["predictor_lock_sha256"]!=digest(root/"predictor_lock.json")):
        raise ValueError("Historical Robust CCLA differs from this 959-patch study")
    bias = load_json(root/"validation_robust_bias_lock.json")
    fit = load_json(root/"predictor_lock.json")
    if bias["gamma"]!=fit["gamma"] or set(bias["events"])!=set(patches):
        raise ValueError("Robust CCLA bias/fit manifest mismatch")
    for e,ps in patches.items():
        if set(bias["events"][e])!={p["patch_id"] for p in ps}:
            raise ValueError(f"Historical Robust CCLA patch scope differs: {e}")
    old = {r["event_id"]:r for r in read_csv(root/"event_metrics.csv") if r["method"]==ROBUST}
    if set(old)!=set(patches):
        raise ValueError("Missing historical Robust CCLA event metrics")
    return bias["events"],old


def tail_bootstrap(ious,weights):
    order = np.argsort(ious,kind="stable")
    w = weights[:,order]
    k = (weights.sum(axis=1)+2)//3
    used = np.clip(k[:,None]-(w.cumsum(axis=1)-w),0,w)
    return (used@ious[order])/k


def report_validation(out,cms,selected):
    methods = list(next(iter(cms.values())))
    events = list(cms)
    n = len(events)
    draws = np.random.default_rng(SEED).integers(n,size=(ITERATIONS,n))
    weights = (draws[...,None]==np.arange(n)).sum(axis=1)
    summary,details,boot = [],[],{}
    for method in methods:
        mats = np.stack([cms[e][method] for e in events])
        summary.append(dict(method=method,**per_event_metric(mats),worst_event_id=min(events,key=lambda e:scores(cms[e][method])["flood_iou"])))
        d = stats(mats,weights)
        d["lower_tail_iou"] = tail_bootstrap(np.array([scores(cm)["flood_iou"] for cm in mats]),weights)
        boot[method] = d
        for e,mat in zip(events,mats,strict=True):
            tp,fp,fn = fpr_counts(mat)
            details.append(dict(event_id=e,method=method,**scores(mat),tp=tp,fp=fp,fn=fn))
    index = {r["method"]:r for r in summary}
    ci,delta = [],[]
    for method,q in selected.items():
        for ref in METHODS_OLD:
            for metric,vector in boot[method].items():
                left = "flood_"+metric.removeprefix("pooled_flood_") if metric.startswith("pooled_flood_") else metric
                lo,hi,valid = interval(100*(vector-boot[ref][metric]))
                ci.append(dict(method=method,reference=ref,metric=metric,iterations=valid,
                    delta_pp=100*(index[method][left]-index[ref][left]),ci_low_pp=lo,ci_high_pp=hi))
        for e in events:
            a,b = cms[e][method],cms[e][q["init"]]
            d = np.subtract(fpr_counts(a),fpr_counts(b))
            delta.append(dict(event_id=e,method=method,initialization=q["init"],fallback=q["fallback"],
                delta_tp=int(d[0]),delta_fp=int(d[1]),delta_fn=int(d[2]),
                delta_iou_pp=100*(scores(a)["flood_iou"]-scores(b)["flood_iou"])))
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(dir=out) as temp:
        staging = Path(temp)/"validation"
        staging.mkdir()
        for name,rows in (("summary",summary),("event_metrics",details),("paired_bootstrap",ci),
                          ("graph_count_deltas",delta),("focus_events",[r for r in delta if r["event_id"] in ("279","1111003")])):
            save_csv(staging/f"{name}.csv",rows)
        np.savez_compressed(staging/"event_confusions.npz",event_ids=events,methods=methods,
                            confusion=np.stack([[cms[e][m] for m in methods] for e in events]))
        staging.rename(out/"validation")


def run_validation(run,out,robust,plan,scale,lock,workers):
    if (out/"validation/summary.csv").is_file():
        print("Reuse completed validation reports",flush=True)
        return
    if (lock["metrics_sha256"]!=digest(out/"meta_cv_summary.csv")
            or lock["meta_parameters_sha256"]!=digest(out/"meta_parameters.json")):
        raise ValueError("Locked graph configuration no longer matches the OOF metadata")
    # This function is entered only after reading the serialized OOF-selected graph lock.
    old_cms,_ = event_matrices(run,out/"scalar_inputs",plan)
    cache = run/"validation"
    m = read_manifest(cache,dict(split="validation",features_only=False,member_seeds=[42,43,44]))
    index,patches = lookup([(cache,m)])
    arrays = verified_arrays(index)
    del arrays
    rb,original_robust = robust_artifacts(run,robust,plan,patches)
    recorded = load_json(run/"results/bias_lock.json")["events"]
    scalar = load_json(out/"scalar_inputs/scalar_bias_lock.json")["chosen"][F1]["bias"]
    if scalar!=1.5:
        raise ValueError("OOF F1 scalar differs from the established +1.50 comparator")
    selected = lock["selected"]
    original = [dict(key=k,init=k,family="none",strength=0.,sigma=None) for k in (F1,MARGIN,ROBUST)]
    params = original + list(selected.values())
    ordered = [p | dict(eval_fold="validation",biases={
                F1:scalar,MARGIN:recorded[e]["patch_biases"][MARGIN][p["patch_id"]],ROBUST:rb[e][p["patch_id"]]})
               for e,ps in patches.items() for p in ps]
    ans = apply_tasks(out/"validation_blocks",index,ordered,dict(validation=scale),params,workers)
    graph = group_counts(ordered,ans,params)
    cms = {}
    for e,group in graph.items():
        cms[e] = {k:old_cms[e][k] for k in METHODS_OLD if k!=ROBUST}
        for j,k in enumerate((F1,MARGIN)):
            np.testing.assert_array_equal(group[j],cms[e][k],err_msg=f"{e} lambda=0 {k}")
        historical = original_robust[e]
        cm = group[2]
        np.testing.assert_array_equal(fpr_counts(cm),[int(historical[k]) for k in ("tp","fp","fn")])
        s = scores(cm)
        checked(list(s.values()),[float(historical[k]) for k in s],f"{e} replay saved Robust CCLA",TOL)
        cms[e][ROBUST] = cm
        for j,method in enumerate(selected,3):
            if selected[method]["fallback"]:
                np.testing.assert_array_equal(group[j],cms[e][selected[method]["init"]])
            np.testing.assert_array_equal(group[j].sum(axis=1),cms[e][GLOBAL].sum(axis=1))
            cms[e][method] = group[j]
    report_validation(out,cms,selected)
    print(f"Saved 959-patch validation: {out/'validation'}",flush=True)


def run(args):
    maxflow = maxflow_module()
    out,source,robust = args.out.resolve(),args.run.resolve(),args.robust_results.resolve()
    out.mkdir(parents=True,exist_ok=True)
    with (out/".benchmark.lock").open("a") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        frozen = out/"frozen_config.json"
        identity = dict(source=str(source),robust_results=str(robust),script_sha256=digest(Path(__file__)),
            source_run_sha256=digest(source/"frozen_config.json"),predictor_sha256=digest(source/"fit/predictor.json"),
            oof_curves_sha256=digest(source/"fit/oof_patch_curves.npz"),pymaxflow=maxflow.__version__,
            numpy=np.__version__,grid=choices(),patch_block=BLOCK,pairwise_fields=list(FIELDS),
            potts="patch-local four-connected input-valid neighbors; nonflood class from original logits",
            unary="softplus costs with per-node common offset removed; lambda=0 has original strict tie rule",
            sar="source-only scale from up to 50000 valid pixels/event; absent slope omitted from pair distance, rescale by available dimensions",
            selection="macro IoU gain with F1 drop <=0.5 pp and no lower-tail drop, independently by init and family",
            lower_tail="mean of ceil(E/3) lowest event IoUs",
            validation="locked configs only, 959 validation patches, no model inference",
            meta_cv="event-disjoint 18 source/9 held-out events; frozen segmentation OOF logits; not independent nested U-Net CV",
            bootstrap="10000 paired whole-event percentile resamples; seed 42")
        if frozen.exists():
            if load_json(frozen)!=identity:
                raise ValueError("Resume requires the same locked code, dependency versions and inputs")
        else:
            write_json(frozen,identity)
        scalar = out/"scalar_inputs"
        if not (scalar/"scalar_bias_lock.json").is_file():
            lock_oof(source,scalar)
        s_lock = load_json(scalar/"scalar_bias_lock.json")
        if s_lock["oof_curves_sha256"]!=identity["oof_curves_sha256"] or s_lock["predictor_sha256"]!=identity["predictor_sha256"]:
            raise ValueError("Saved scalar lock differs from the OOF inputs")
        plan = load_json(source/"frozen_config.json")
        graph_lock = out/"graph_cut_lock.json"
        if args.stage=="evaluate" and not graph_lock.is_file():
            raise ValueError("Complete meta-CV and lock graph settings first")
        if not graph_lock.exists():
            predictor,grid,curves,ids,events,rows,folds,index = read_oof(source)
            arrays = verified_arrays(index)
            setup = setup_meta(out,arrays,predictor,grid,curves,ids,events,rows,folds)
            del arrays
            run_meta(out,index,rows,ids,events,folds,grid,curves,setup,args.workers)
        if args.stage in ("all","evaluate"):
            lock = load_json(graph_lock)
            scale = load_json(out/"meta_parameters.json")["all_oof_sar_scale"]
            run_validation(source,out,robust,plan,scale,lock,args.workers)


def energy(pred, s, valid, strength, links):
    nonflood = np.maximum(s,0)
    flood = np.maximum(-s,0)
    value = np.where(pred,flood,nonflood)[valid].sum()
    right,down = links
    value += strength*(right[:,:-1]*(pred[:,:-1]!=pred[:,1:])).sum()
    value += strength*(down[:-1,:]*(pred[:-1,:]!=pred[1:,:])).sum()
    return value


class GraphCutTests(unittest.TestCase):
    def test_exhaustive_binary_potts(self):
        rng = np.random.default_rng(42)
        for seed in range(32):
            s = rng.normal(size=(2,3))
            valid = rng.random((2,3))>.2
            x = rng.normal(size=(2,3,5))
            x[0,0,4] = np.nan
            links = edge_weights(valid,x,0.5+seed%3*.5) if seed%2 else edge_weights(valid)
            lam = LAMBDAS[1+seed%5]
            pred = graph_cut(s,valid,lam,links)
            best = np.inf
            for state in product((False,True),repeat=int(valid.sum())):
                mask = np.zeros(s.shape,dtype=bool)
                mask[valid]=state
                best = min(best,energy(mask,s,valid,lam,links))
            self.assertAlmostEqual(float(energy(pred,s,valid,lam,links)),float(best),places=10)
            self.assertFalse(np.any(pred&~valid))

    def test_zero_lambda_and_mask_holes(self):
        m = np.array([[-1.,0.,.25],[1.,5.,-2.]])
        v = np.array([[True,True,True],[True,False,True]])
        np.testing.assert_array_equal(graph_cut(m,v,0),v&(m>0))
        r,d = edge_weights(v)
        self.assertEqual(r[1,:].sum(),0)
        self.assertEqual(d[0,1],0)

    def test_candidate_grid_and_bootstrap(self):
        self.assertEqual(len(choices()),42)
        a = np.array([.4,.5,.6,.7,.1,.2])
        picks = np.array([[0,1,2,3,4,5],[5,5,5,1,3,4]])
        w = (picks[...,None]==np.arange(len(a))).sum(axis=1)
        np.testing.assert_allclose(tail_bootstrap(a,w),np.sort(a[picks],axis=1)[:,:2].mean(axis=1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("meta","evaluate","all","self-test"))
    p.add_argument("--run",type=Path)
    p.add_argument("--robust-results",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--workers",type=int,default=6)
    a = p.parse_args()
    if a.stage=="self-test":
        maxflow_module()
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(GraphCutTests)
        raise SystemExit(0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1)
    if any(getattr(a,k) is None for k in ("run","out","robust_results")) or a.workers<1:
        p.error("benchmark requires --run, --robust-results, --out and a positive --workers")
    run(a)


if __name__=="__main__":
    main()
