#!/usr/bin/env python3
"""Exploratory leave-one-validation-event-out BBSE/EM flood-bias calibration."""
import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp, softmax

from evaluate_unet_eoa import (
    FLIP_TTA_BASELINE, digest, load_calibrated_selection, load_json, load_model,
    pre_variability_inputs, seed_everything, write_json,
)
from evaluate_unet_eoa_diagnostics import pr_curve, rank_event, rank_pooled
from evaluate_unet_eoa_flip_tta import FLIPS, FlipTTAEnsemble
from evaluate_unet_vanilla import write_csv
from kuro_dataset import KuroDataset, RawArchive

SEED = 42
NCLASS = 3
FLOOD = 2
IGNORE = 255
CAP = 100_000
MAX_EM_ITER = 300
EM_TOL = 1e-7
FLOOR = 1e-9
METHODS = ("no_bias", "global_bias", "bbse", "em", "oracle_DIAGNOSTIC_ONLY")
CLASSES = ("background", "permanent_water", "flood")
VAL_ARTIFACT = Path(__file__).resolve().parents[2] / "artifacts/a3-eoa-flip-tta-q8H1dB/results"


def reference(*, load_confusions=True):
    recipe = load_json(FLIP_TTA_BASELINE)
    path = Path(recipe["selection"])
    if digest(path) != recipe["selection_sha256"] or recipe["flip_axes"] != [list(a) for a in FLIPS]:
        raise ValueError("Frozen selection or Flip TTA protocol changed")
    if (digest(VAL_ARTIFACT / "protocol.json") != recipe["validation_protocol_sha256"] or
            digest(VAL_ARTIFACT / "bias_selection.json") != recipe["validation_bias_lock_sha256"]):
        raise ValueError("Validation bias lock does not match the frozen recipe")
    selection = load_json(path)
    if load_json(VAL_ARTIFACT / "protocol.json")["checkpoint_members"] != selection["sma"]:
        raise ValueError("Archived validation does not use the frozen members")
    with np.load(VAL_ARTIFACT / "validation_confusion_eoa_flip_tta.npz") as source:
        grid = source["biases"]
        expected = {e: source[e] if load_confusions else None for e in source.files if e != "biases"}
    if len(expected) != 6:
        raise ValueError("Expected the six official validation events")
    return recipe, selection, grid, expected


def file_paths(root, event):
    return root / f"{event}.logits.f32", root / f"{event}.labels.u8"


def cache_validation(out, recipe, selection, expected):
    import torch
    from torch.utils.data import DataLoader

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the frozen AMP validation inference")
    if out.exists():
        raise FileExistsError(out)
    verified = load_calibrated_selection(recipe, Path(recipe["selection"]))
    settings = verified["protocol"]["training"]
    archive = RawArchive(settings["raw"])
    dataset = KuroDataset(archive, "validation")
    if set(dataset.rows.actid.astype(str)) != set(expected):
        raise ValueError("Validation events changed")
    device = torch.device("cuda")
    seed_everything(SEED)
    models = [load_model(m, device) for m in verified["sma"]]
    model = FlipTTAEnsemble(models).to(device).eval()
    loader = DataLoader(dataset, batch_size=settings["batch_size"], num_workers=settings["workers"],
                        shuffle=False, pin_memory=True)

    out.mkdir(parents=True, exist_ok=False)
    counts = dict.fromkeys(expected, 0)
    with torch.inference_mode():
        for step, batch in enumerate(loader, 1):
            x = {key: batch[key].to(device, non_blocking=True) for key in ("sar","dem")}
            with torch.autocast("cuda"):
                outputs = model(pre_variability_inputs(x)[0])
            logits = outputs.float().permute(0,2,3,1).cpu().numpy()
            for i, event in enumerate(batch["event_id"]):
                event = str(event)
                # This mask is computed from physical inputs, independent of annotation IGNORE.
                included = batch["input_valid"][i].numpy()
                scores = logits[i][included].astype("<f4", copy=False)
                labels = batch["mask"][i].numpy()[included].astype("u1")
                if not np.isfinite(scores).all() or not np.isin(labels,[0,1,2,IGNORE]).all():
                    raise ValueError(f"Invalid validation values: {event}")
                x_path, y_path = file_paths(out, event)
                with x_path.open("ab") as stream:
                    scores.tofile(stream)
                with y_path.open("ab") as stream:
                    labels.tofile(stream)
                counts[event] += len(scores)
            if step % 100 == 0 or step == len(loader):
                print(f"validation cache: {step}/{len(loader)} batches",flush=True)
    write_json(out/"manifest.json",dict(
        split="validation", selection_sha256=recipe["selection_sha256"], members=selection["sma"],
        recipe_sha256=digest(FLIP_TTA_BASELINE), domain="input_valid_only", dtype="<f4",
        event_counts=counts,
        file_sha256={p.name:digest(p) for event in counts for p in file_paths(out,event)},
    ))


def open_cache(cache, recipe, selection, expected):
    manifest = load_json(cache/"manifest.json")
    check = (manifest["split"], manifest["selection_sha256"], manifest["recipe_sha256"],
             manifest["domain"], manifest["dtype"], manifest["members"])
    if check != ("validation",recipe["selection_sha256"],digest(FLIP_TTA_BASELINE),
                 "input_valid_only","<f4",selection["sma"]):
        raise ValueError("Cache provenance differs from the frozen validation recipe")
    if set(manifest["event_counts"]) != set(expected):
        raise ValueError("Unexpected cached event IDs")
    inputs, labels = {}, {}
    for event, size in manifest["event_counts"].items():
        x,y = file_paths(cache,event)
        for path, bytes_expected in ((x,12*size),(y,size)):
            if not size or path.stat().st_size != bytes_expected or digest(path) != manifest["file_sha256"][path.name]:
                raise ValueError(f"Incomplete or changed cache file: {path}")
        inputs[event] = np.memmap(x,mode="r",dtype="<f4",shape=(size,3))
        labels[event] = np.memmap(y,mode="r",dtype="u1",shape=(size,))
    return inputs,labels


def sample_indices(indices, rng):
    if len(indices) <= CAP:
        return indices
    return np.sort(rng.choice(indices,CAP,replace=False))


def labeled_source_statistics(logits, labels, rng):
    # Compute each event's stats independently. The caller excludes target stats from its fold.
    valid = np.flatnonzero(labels != IGNORE)
    if not len(valid):
        raise ValueError("No labeled pixels in a source event")
    y = labels[valid].astype(int)
    n = np.bincount(y,minlength=3)
    hard = logits[valid].argmax(1)
    confusion = np.bincount(hard*3+y,minlength=9).reshape(3,3)
    choice = sample_indices(valid,rng)
    return n,confusion,logits[choice],labels[choice],np.full(len(choice),len(valid)/len(choice))


def temperature_scale(source):
    x,y,w = (np.concatenate(parts) for parts in zip(*(r[2:] for r in source),strict=True))
    x = x.astype(np.float64)
    actual = x[np.arange(len(y)),y]

    def nll(log_t):
        inv = np.exp(-log_t)
        return float(np.average(logsumexp(inv*x,axis=1)-inv*actual,weights=w))

    result = minimize_scalar(nll,bounds=(np.log(.05),np.log(20.)),method="bounded",
                             options={"xatol":1e-5})
    if not result.success or not np.isfinite(result.fun):
        return 1., "temperature_fallback", nll(0.)
    return float(np.exp(result.x)), "ok", float(result.fun)


def bbse(confusion, target_hard_prior, source_prior):
    totals = confusion.sum(axis=0)
    if not totals.all():
        return source_prior.copy(), {"status":"source_class_missing","condition":None,"residual":None}
    matrix = confusion/totals
    condition = np.linalg.cond(matrix)
    if not np.isfinite(condition) or np.linalg.matrix_rank(matrix) < NCLASS:
        return source_prior.copy(), {"status":"singular_source_confusion","condition":None,
                                     "residual":float(np.linalg.norm(matrix@source_prior-target_hard_prior))}
    ridge = 1e-4 if condition > 1e4 else 0.

    def objective(p):
        residual = matrix@p-target_hard_prior
        return float(.5*(residual@residual+ridge*np.sum((p-source_prior)**2)))

    def jacobian(p):
        return matrix.T@(matrix@p-target_hard_prior)+ridge*(p-source_prior)

    result = minimize(objective,source_prior,jac=jacobian,method="SLSQP",bounds=[(0,1)]*3,
                      constraints=[{"type":"eq","fun":lambda p:p.sum()-1,
                                    "jac":lambda p:np.ones(3)}],
                      options={"ftol":1e-13,"maxiter":100})
    if not result.success or not np.isfinite(result.x).all():
        p,status = source_prior.copy(),"optimizer_fallback"
    else:
        p = np.maximum(result.x,0)
        p /= p.sum()
        status = "ill_conditioned_regularized" if ridge else "ok"
    return p,{"status":status,"condition":float(condition),"ridge":ridge,
              "residual":float(np.linalg.norm(matrix@p-target_hard_prior)),
              "at_boundary":bool(np.any(p<1e-8))}


def em(logits,temperature,source_prior):
    prior = np.maximum(source_prior,FLOOR)
    prior /= prior.sum()
    q = softmax(logits.astype(np.float64)/temperature,axis=1) / prior
    p = prior.copy()
    for iteration in range(1,MAX_EM_ITER+1):
        w = q*p
        denom = w.sum(axis=1,keepdims=True)
        if (denom <= 0).any() or not np.isfinite(denom).all():
            return prior,{"status":"numeric_fallback","iterations":iteration}
        new = (w/denom).mean(axis=0)
        new = np.maximum(new,FLOOR)
        new /= new.sum()
        if np.max(abs(new-p)) <= EM_TOL:
            return new,{"status":"ok","iterations":iteration,
                        "at_boundary":bool(np.any(new<1e-8))}
        p = new
    return p,{"status":"max_iterations","iterations":MAX_EM_ITER,
              "at_boundary":bool(np.any(p<1e-8))}


def flood_margin(logits):
    return logits[:,FLOOD] - np.maximum(logits[:,0],logits[:,1])


def bias_for_groups(values, groups):
    """Choose the smallest-magnitude float64 bias for a given number of score groups."""
    if groups == 0:
        return min(0., -float(values[-1]))
    lo = np.nextafter(-np.float64(values[-groups]), np.inf)
    hi = -float(values[-groups-1]) if groups < len(values) else np.inf
    return float(np.clip(0., lo, hi))


def prevalence_bias(margin, prevalence):
    """Find the closest attainable prevalence, with ties resolved by smallest |bias|."""
    if not len(margin) or not np.isfinite(prevalence) or not 0 <= prevalence <= 1:
        raise ValueError("Invalid target prevalence request")
    if not np.isfinite(margin).all():
        raise ValueError("Nonfinite flood margin")
    values, counts = np.unique(margin, return_counts=True)
    attainable = np.r_[0, np.cumsum(counts[::-1])]/len(margin)
    distances = abs(attainable-prevalence)
    winners = np.flatnonzero(np.isclose(distances, distances.min(), atol=1e-14, rtol=0))
    candidates = [(bias_for_groups(values, int(i)), int(i)) for i in winners]
    bias, i = min(candidates, key=lambda pair: (abs(pair[0]), pair[0]))
    observed = float(np.mean(np.greater(margin.astype(np.float64), -bias)))
    if not np.isclose(observed, attainable[i], atol=1e-14, rtol=0):
        raise ValueError("Flood bias failed to reproduce the attainable prevalence")
    return bias, observed


def oracle_from_ranks(ranked):
    """Find the exact per-event IoU upper bound over all attainable thresholds."""
    values, positive, total = ranked
    tp = np.r_[0, np.cumsum(positive[::-1])]
    predicted = np.r_[0, np.cumsum(total[::-1])]
    iou = tp / np.maximum(1, predicted+positive.sum()-tp)
    winners = np.flatnonzero(iou == iou.max())
    biases = [bias_for_groups(values, int(k)) for k in winners]
    return min(biases, key=lambda b: (abs(b), b)), float(iou.max())


def estimate(target_logits, source_statistics, seed):
    """This entry point has no target-label or annotation-mask argument."""
    n = np.sum([s[0] for s in source_statistics],axis=0)
    source_prior = n/n.sum()
    confusion = np.sum([s[1] for s in source_statistics],axis=0)
    temperature,status,nll = temperature_scale(source_statistics)
    observed = np.bincount(target_logits.argmax(1),minlength=3)/len(target_logits)
    subsample = sample_indices(np.arange(len(target_logits)),np.random.default_rng(seed))
    estimates = dict(bbse=bbse(confusion,observed,source_prior),
                     em=em(target_logits[subsample],temperature,source_prior))
    margin = flood_margin(target_logits)
    output={}
    for method,(prior,diagnostics) in estimates.items():
        bias,matched = prevalence_bias(margin,float(prior[FLOOD]))
        output[method]=dict(prior=prior.tolist(),bias=bias,predicted_prevalence_input_valid=matched,
                            diagnostics=diagnostics)
    return dict(source_prior=source_prior.tolist(),confusion_pred_by_true=confusion.tolist(),
                temperature=temperature,temperature_status=status,source_calibrated_nll=nll,
                unlabeled_predicted_prior=observed.tolist(),target_em_sample_size=len(subsample),
                methods=output)


def scores(cm):
    t = cm.diagonal()
    predicted, actual = cm.sum(0), cm.sum(1)
    union = predicted+actual-t
    return dict(pooled_miou=float(np.mean(t/np.maximum(union,1))),
                flood_iou=float(t[FLOOD]/max(union[FLOOD],1)),
                flood_f1=float(2*t[FLOOD]/max(predicted[FLOOD]+actual[FLOOD],1)),
                flood_precision=float(t[FLOOD]/max(predicted[FLOOD],1)),
                flood_recall=float(t[FLOOD]/max(actual[FLOOD],1)))


def predict_confusion(logits,labels,bias):
    margin=flood_margin(logits)
    predictions=np.where(margin.astype(np.float64)>-bias,FLOOD,logits[:,:FLOOD].argmax(1))
    return np.bincount(labels.astype(np.int64)*NCLASS+predictions,minlength=9).reshape(3,3)


def evaluate(cache,out,recipe,selection,grid,expected,*, include_otsu=False,
             include_entropy=False, view_cache=None):
    if out.exists():
        raise FileExistsError(out)
    x,y=open_cache(cache,recipe,selection,expected)
    events=sorted(x,key=int)
    extra = [name for name, enabled in (("otsu",include_otsu),("entropy",include_entropy),
                                       ("tta_consistency",view_cache is not None)) if enabled]
    names = (*METHODS[:-1], *extra, METHODS[-1])
    objective_rules, objective_rows = None, []
    if include_entropy or view_cache is not None:
        import unet_bias_objectives as objectives

        objective_rules = objectives.RULES
        views = objectives.load_views(view_cache,cache,recipe,x) if view_cache is not None else {}
    if include_otsu:
        from unet_otsu_delta import RULES, delta_histograms, histogram_rows, otsu, plot_histograms

        # The Otsu module receives only input-valid predictions, never y.
        histograms = delta_histograms(selection["protocol"]["training"]["raw"], x)
    source={event:labeled_source_statistics(x[event],y[event],np.random.default_rng(SEED+i))
            for i,event in enumerate(events)}
    locked={}
    for i,event in enumerate(events):
        ids=[e for e in events if e!=event]
        locked[event]=dict(source_events=ids,input_valid_pixels=len(x[event]),
                           **estimate(x[event],[source[e] for e in ids],SEED+i))
        if include_otsu:
            prevalence, diagnostics = otsu(*histograms[event])
            margin = flood_margin(x[event])
            if prevalence is None:
                bias = RULES["fallback_bias"]
                matched = float(np.mean(margin.astype(np.float64) > -bias))
            else:
                bias, matched = prevalence_bias(margin, prevalence)
            locked[event]["methods"]["otsu"] = dict(
                prior=[None,None,prevalence], bias=bias, diagnostics=diagnostics,
                predicted_prevalence_input_valid=matched,
            )
            print(f"Locked {event}: Otsu b={bias:.4f} ({diagnostics['status']})", flush=True)
        if objective_rules is not None:
            candidate_curves = {}
            if include_entropy:
                sample = x[event][objectives.pixel_indices(len(x[event]),event)]
                candidate_curves["entropy"] = objectives.entropy_curve(sample)
            if event in views:
                candidate_curves["tta_consistency"] = objectives.consistency_curve(views[event])
            margin = flood_margin(x[event])
            for name, curve in candidate_curves.items():
                result, points = objectives.choose_bias(curve,margin)
                locked[event]["methods"][name] = result
                objective_rows.extend(dict(event_id=event,method=name,**point) for point in points)
                print(f"Locked {event}: {name} b={result['bias']:.2f} "
                      f"({result['diagnostics']['status']})",flush=True)
        print(f"Locked {event}: BBSE b={locked[event]['methods']['bbse']['bias']:.4f}, "
              f"EM b={locked[event]['methods']['em']['bias']:.4f}",flush=True)
    out.mkdir(parents=True,exist_ok=False)
    if include_otsu:
        write_csv(out/"otsu_delta_vv_histogram.csv", list(histogram_rows(histograms)))
        plot_histograms(out/"otsu_delta_vv_histogram.png", histograms, locked)
        write_csv(out/"otsu_thresholds.csv", [
            dict(event_id=e, estimated_flood_prevalence=m["prior"][FLOOD], applied_bias=m["bias"],
                 **m["diagnostics"])
            for e in events for m in [locked[e]["methods"]["otsu"]]
        ])
    write_json(out/"bias_lock.json",dict(
        scope="validation exploratory; not a LOEO replacement",
        source="five other validation events, source labels and predictions",
        target_support="input_valid only, excludes annotation validity from bias estimation",
        global_bias=recipe["flood_logit_bias"],seed=SEED,sample_cap=CAP,
        threshold_rule="float64 comparison of float32 margin > -bias; nearest attainable prevalence",
        temperature_bounds=[.05,20.],em_floor=FLOOR,em_max_iter=MAX_EM_ITER,em_tol=EM_TOL,
        cache=str(cache),cache_manifest_sha256=digest(cache/"manifest.json"),
        checkpoint_selection_sha256=recipe["selection_sha256"],
        otsu_rules=RULES if include_otsu else None,objective_rules=objective_rules,
        view_cache_manifest_sha256=digest(view_cache/"manifest.json") if view_cache else None,
        events=locked))
    if objective_rows:
        write_csv(out/"objective_curves.csv",objective_rows)
        objectives.plot_curves(out,objective_rows)

    # Evaluation begins after writing the locked unlabeled estimates.
    eval_rows, matrix_rows = [], {name:[] for name in names}
    ap_groups = {name:[] for name in names}
    global_bias=recipe["flood_logit_bias"]
    zero,global_index=(int(np.flatnonzero(np.isclose(grid,b,rtol=0,atol=1e-12)).item())
                       for b in (0.,global_bias))
    for event in events:
        keep=y[event]!=IGNORE
        labels=y[event][keep]
        logits=x[event][keep]
        source_cm=expected[event]
        ranked=rank_event([(flood_margin(logits),labels==FLOOD)])
        oracle_bias, oracle_iou=oracle_from_ranks(ranked)
        event_ap=pr_curve(ranked)[-1]
        prior=np.bincount(labels,minlength=3)/len(labels)
        methods = locked[event]["methods"]
        applied = dict(no_bias=0.,global_bias=global_bias,oracle_DIAGNOSTIC_ONLY=oracle_bias,
                       **{name:result["bias"] for name,result in methods.items()})
        for name in names:
            bias = applied[name]
            cm=predict_confusion(logits,labels,bias)
            baseline_index={"no_bias":zero,"global_bias":global_index}.get(name)
            if baseline_index is not None and not np.array_equal(cm,source_cm[baseline_index]):
                raise ValueError(f"{event}/{name}: cached logits disagree with archived confusion")
            metric = scores(cm)
            if metric["flood_iou"] > oracle_iou + 1e-12:
                raise ValueError(f"{event}/{name}: observed IoU exceeds exact oracle")
            if name == "oracle_DIAGNOSTIC_ONLY" and not np.isclose(metric["flood_iou"], oracle_iou, atol=1e-12, rtol=0):
                raise ValueError(f"{event}: oracle bias does not reproduce its maximum")
            predicted_input=(methods[name]["predicted_prevalence_input_valid"] if name in methods else
                             float((flood_margin(x[event]).astype(np.float64)>-bias).mean()))
            values=dict(event_id=event,method=name,source_events=" | ".join(locked[event]["source_events"]),
                        evaluation_pixels=len(labels),input_valid_pixels=len(x[event]),
                        bias=bias,oracle_bias_iou_DIAGNOSTIC_ONLY=oracle_bias,
                        oracle_outside_archive_grid_DIAGNOSTIC_ONLY=not float(grid[0])<=oracle_bias<=float(grid[-1]),
                        predicted_prevalence_input_valid=predicted_input,
                        predicted_prevalence_scored=float(cm[:,FLOOD].sum()/len(labels)),
                        flood_ap=event_ap,status=methods[name]["diagnostics"]["status"] if name in methods else "",
                        **metric)
            for i,c in enumerate(CLASSES):
                values[f"actual_{c}_prevalence"]=float(prior[i])
                values[f"estimated_{c}_prevalence"]=methods[name]["prior"][i] if name in methods else None
            matrix_rows[name].append(cm)
            ap_groups[name].append((ranked[0].astype(np.float64)+bias,*ranked[1:]))
            eval_rows.append(values)
    result=[]
    for name in names:
        rows=[r for r in eval_rows if r["method"]==name]
        iou=[r["flood_iou"] for r in rows]
        errors=[abs(r["actual_flood_prevalence"]-r["estimated_flood_prevalence"]) for r in rows
                if r["estimated_flood_prevalence"] is not None]
        fallback_statuses = {"source_class_missing", "singular_source_confusion", "optimizer_fallback",
                             "numeric_fallback"}
        result.append(dict(method=name,n_events=len(rows),**scores(sum(matrix_rows[name])),
                           pooled_flood_ap=pr_curve(rank_pooled(ap_groups[name]))[-1],
                           event_macro_flood_ap=float(np.mean([r["flood_ap"] for r in rows])),
                           event_macro_flood_iou=float(np.mean(iou)),
                           event_flood_iou_std=float(np.std(iou,ddof=0)),
                           flood_prevalence_mae=float(np.mean(errors)) if errors else None,
                           n_prevalence_estimates=len(errors),
                           n_fallbacks=sum(r["status"].startswith("fallback_") or r["status"] in fallback_statuses
                                           for r in rows)))
    write_csv(out/"event_metrics.csv",eval_rows)
    write_csv(out/"summary.csv",result)
    write_json(out/"evaluation_config.json",dict(
        split="validation",experiment="exploratory; checkpoints/global bias historically selected on all validation events",
        source_caveat="each fold excludes its target from calibration and confusion; this is not train-on-source-only LOEO",
        bias_lock_sha256=digest(out/"bias_lock.json"), actual_prevalence="annotation-valid pixels only",
        ap="exact flood margin rank, grouped ties; per-event unchanged by a constant bias; pooled AP uses adapted margins",
        oracle="per-event maximum Flood IoU over exact attainable thresholds, diagnostics only",
        ground_truth_used="source statistics for other events; evaluation and oracle for target",
        evaluator_sha256=digest(Path(__file__)),
        otsu_sha256=digest(Path(__file__).with_name("unet_otsu_delta.py")) if include_otsu else None,
        objective_sha256=digest(Path(__file__).with_name("unet_bias_objectives.py"))
        if objective_rules is not None else None))
    print(f"Saved validation-only metrics: {out}",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    cache=sub.add_parser("cache",help="Save frozen logits on the official validation split")
    cache.add_argument("--out",required=True,type=Path)
    view=sub.add_parser("cache-views",help="Cache individual aligned Flip TTA logits on validation only")
    view.add_argument("--cache",required=True,type=Path)
    view.add_argument("--out",required=True,type=Path)
    ev=sub.add_parser("evaluate",help="Use cached logits for BBSE/EM and validation comparison")
    ev.add_argument("--cache",required=True,type=Path)
    ev.add_argument("--out",required=True,type=Path)
    ev.add_argument("--otsu",action="store_true",help="Add label-free Delta-VV Otsu using the validation cache")
    ev.add_argument("--entropy",action="store_true",help="Add label-free predictive entropy minimization")
    ev.add_argument("--view-cache",type=Path,help="Add flip consistency using the aligned view cache")
    args=parser.parse_args()
    recipe,selection,grid,expected=reference()
    if args.command=="cache":
        cache_validation(args.out.resolve(),recipe,selection,expected)
    elif args.command=="cache-views":
        from unet_bias_objectives import cache_views

        cache_views(args.out.resolve(),args.cache.resolve(),recipe,selection,expected)
    else:
        evaluate(args.cache.resolve(),args.out.resolve(),recipe,selection,grid,expected,
                 include_otsu=args.otsu,include_entropy=args.entropy,
                 view_cache=args.view_cache.resolve() if args.view_cache else None)


if __name__=="__main__":
    main()
