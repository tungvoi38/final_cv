#!/usr/bin/env python3
"""Decompose the frozen Spatial Oracle by gating and within-flood bias optimization."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adapt_unet_event_bias import FLOOD, IGNORE, digest, flood_margin, load_json, oracle_from_ranks, reference, scores, write_json
from adapt_unet_iw_iou import cache_file, open_predictions
from audit_unet_bias_failures import checked, read_csv, save_csv
from diagnose_unet_spatial_oracle import (
    EVENT, GLOBAL, PATCH, average_precision, predicted_confusion, shifted_rank,
)
from evaluate_unet_eoa_diagnostics import rank_event, rank_pooled

GATE_GLOBAL = "oracle_patch_gating_global_bias"
GATE_FIXED = "oracle_patch_gating_fixed_event_bias_auxiliary"
GATE_REFIT = "oracle_patch_gating_refit_event_oracle"
METHODS = (GLOBAL, EVENT, GATE_GLOBAL, GATE_FIXED, GATE_REFIT, PATCH)
CORE_METRICS = ("pooled_miou", "flood_f1", "flood_iou", "macro_flood_iou")


def by_key(rows, fields):
    result = {tuple(r[f] for f in fields):r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicated artifact rows for {fields}")
    return result


def load_spatial(root, cache, recipe, metadata):
    config = load_json(root/"evaluation_config.json")
    patch_path = root/"patch_manifest.json"
    manifest = load_json(patch_path)
    cache_hash = digest(cache/"manifest.json")
    if (config["split"] != "validation" or manifest["split"] != "validation"
            or config["checkpoint_selection_sha256"] != recipe["selection_sha256"]
            or config["global_bias"] != recipe["flood_logit_bias"]
            or config["mean_cache_sha256"] != cache_hash or manifest["mean_cache_sha256"] != cache_hash
            or config["patch_manifest_sha256"] != digest(patch_path)
            or manifest["event_counts"] != metadata["event_counts"]
            or set(manifest["patches"]) != set(metadata["event_counts"])):
        raise ValueError("Spatial Oracle artifacts do not match the frozen validation cache")
    rows = [r for r in read_csv(root/"event_metrics.csv") if r["method"] in (GLOBAL, EVENT, PATCH)]
    groups = [r for r in read_csv(root/"group_biases.csv") if r["method"] == PATCH]
    old = by_key(rows, ("event_id", "method"))
    old_groups = by_key(groups, ("event_id", "group_id"))
    if set(old) != {(e,m) for e in manifest["patches"] for m in (GLOBAL, EVENT, PATCH)}:
        raise ValueError("Incomplete historical Spatial Oracle event coverage")
    if set(old_groups) != {(e,p["patch_id"]) for e,patches in manifest["patches"].items() for p in patches}:
        raise ValueError("Historical patch biases do not cover the patch manifest")
    return manifest["patches"], old, old_groups


def flood_counts(cm):
    return int(cm[2,2]), int(cm[:2,2].sum()), int(cm[2,:2].sum())


def load_patch_ranks(logits, labels, patches, old_groups, event):
    expected = 0
    records = []
    for p in patches:
        patch_id,start,stop = p["patch_id"],p["start"],p["stop"]
        if start != expected or not start <= stop <= len(labels):
            raise ValueError(f"Invalid patch range: {event}/{patch_id}")
        expected = stop
        annot = labels[start:stop]
        keep = annot != IGNORE
        y = annot[keep]
        if not np.isin(y,[0,1,2]).all():
            raise ValueError(f"Invalid cached labels: {event}/{patch_id}")
        z = logits[start:stop][keep]
        margin = flood_margin(z)
        if not np.isfinite(margin).all():
            raise ValueError(f"Nonfinite margin: {event}/{patch_id}")
        nonflood = z[:,:FLOOD].argmax(axis=1)
        truth = y==FLOOD
        positive = int(truth.sum())
        old = old_groups[event,patch_id]
        np.testing.assert_array_equal(
            [stop-start,len(y),positive],
            [int(old[k]) for k in ("input_valid_pixels","scored_pixels","true_flood_pixels")],
            err_msg=f"{event}/{patch_id}: historical patch alignment")
        state = "no_scored_pixels" if not len(y) else "scored" if positive else "no_flood"
        if state != old["status"] or (not old["bias"]) != (not len(y)):
            raise ValueError(f"Historical patch status differs: {event}/{patch_id}")
        rank = rank_event([(margin,truth)]) if len(y) else None
        records.append((patch_id,y,margin,nonflood,rank,positive,old))
    if expected != len(labels):
        raise ValueError(f"Incomplete patch ranges for {event}")
    return records


def gated_rank(rank):
    # Gate is a hard abstention from class flood. All blocked scores tie at the bottom.
    return np.array([-np.inf]), np.array([0],dtype=np.int64), np.array([rank[2].sum()],dtype=np.int64)


def audit_event(event, logits, labels, patches, old, old_groups, global_bias):
    records = load_patch_ranks(logits,labels,patches,old_groups,event)
    active = [r[4] for r in records if r[5]]
    refit, refit_iou = oracle_from_ranks(rank_pooled(active)) if active else (None,0.)
    event_bias = float(old[event,EVENT]["scalar_bias"])
    fixed = {GLOBAL:global_bias,EVENT:event_bias,GATE_GLOBAL:global_bias,GATE_FIXED:event_bias,GATE_REFIT:refit}
    pooled = {m:np.zeros((3,3),dtype=np.int64) for m in METHODS}
    free = {m:np.zeros((3,3),dtype=np.int64) for m in METHODS}
    ranks = {m:[] for m in METHODS}
    patch_rows, raw_ranks = [], []
    inventory = dict(patches_total=len(records),patches_no_scored=0,patches_flood_free=0,patches_flood_containing=0,
                     flood_free_scored_pixels=0,flood_containing_scored_pixels=0)
    for patch_id,y,margin,nonflood,r,positive,previous in records:
        if not len(y):
            inventory["patches_no_scored"] += 1
            continue
        patch_type = "flood_containing" if positive else "flood_free"
        inventory[f"patches_{patch_type}"] += 1
        inventory[f"{patch_type}_scored_pixels"] += len(y)
        raw_ranks.append(r)
        for method in METHODS:
            bias = float(previous["bias"]) if method == PATCH else fixed[method]
            gated = method in (GATE_GLOBAL,GATE_FIXED,GATE_REFIT) and not positive
            if gated:
                cm = np.bincount(3*y.astype(np.int64)+nonflood,minlength=9).reshape(3,3)
                adjusted = gated_rank(r)
            else:
                cm = predicted_confusion(margin,nonflood,y,bias)
                adjusted = shifted_rank(r,bias)
            tp,fp,fn = flood_counts(cm)
            if method == PATCH:
                np.testing.assert_array_equal([tp,fp,fn],[int(previous[k]) for k in ("tp","fp","fn")],
                                              err_msg=f"{event}/{patch_id} archived Patch Oracle counts")
            pooled[method] += cm
            ranks[method].append(adjusted)
            if not positive:
                free[method] += cm
            patch_rows.append(dict(event_id=event,patch_id=patch_id,patch_type=patch_type,
                                   method=method,scored_pixels=len(y),true_flood_pixels=positive,
                                   gated=gated,bias=None if gated else bias,tp=tp,fp=fp,fn=fn))
    if not raw_ranks:
        raise ValueError(f"No scored pixels in {event}")
    raw_rank = rank_pooled(raw_ranks)
    raw_ap = average_precision(raw_rank)
    rows = []
    for method in METHODS:
        cm = pooled[method]
        tp,fp,fn = flood_counts(cm)
        free_fp = flood_counts(free[method])[1]
        metric = scores(cm)
        rank = rank_pooled(ranks[method])
        ap = average_precision(rank)
        if method in (GLOBAL,EVENT,PATCH):
            historical = old[event,method]
            checked([tp,fp,fn],[int(historical[k]) for k in ("tp","fp","fn")],f"{event} {method} TP/FP/FN")
            checked([*metric.values(),ap],[float(historical[k]) for k in (*metric,"flood_ap_adjusted")],
                    f"{event} {method} archived metrics")
        rows.append(dict(event_id=event,method=method,scored_pixels=int(cm.sum()),tp=tp,fp=fp,fn=fn,
                         fp_flood_free=free_fp,fp_flood_containing=fp-free_fp,**inventory,**metric,
                         flood_ap_raw=raw_ap,flood_ap_adjusted=ap,
                         scalar_bias=None if method==PATCH else fixed[method]))
        ranks[method] = rank
    checked(scores(pooled[GATE_REFIT])["flood_iou"],refit_iou,f"{event} gated optimum")
    for a,b in ((GLOBAL,GATE_GLOBAL),(EVENT,GATE_FIXED)):
        before,after = flood_counts(pooled[a]),flood_counts(pooled[b])
        checked([after[0],after[2],before[1]-after[1]],
                [before[0],before[2],flood_counts(free[a])[1]],f"{event} {b} gate-only effect")
    progression = [scores(pooled[m])["flood_iou"] for m in (EVENT,GATE_FIXED,GATE_REFIT,PATCH)]
    if np.any(np.diff(progression)<-1e-12):
        raise ValueError(f"Flood IoU decreased on a nested oracle: {event}")
    return pooled, ranks, raw_rank, rows, patch_rows


def decompose(rows, metrics, scope, event_id=None):
    values = {row["method"]:row for row in rows}
    output = []
    for metric in metrics:
        global_,ev,gated_global,gated_fixed,refit,patch = [values[m][metric] for m in METHODS]
        headroom = patch-ev
        gate_only = gated_fixed-ev
        shared_refit = refit-gated_fixed
        local = patch-refit
        global_headroom = patch-global_
        def share(part, whole):
            return 100*part/whole if abs(whole)>1e-12 else None
        output.append(dict(scope=scope,event_id=event_id,metric=metric,
            global_bias_pct=100*global_,event_oracle_pct=100*ev,gated_global_pct=100*gated_global,
            gated_fixed_event_pct=100*gated_fixed,gated_refit_event_pct=100*refit,patch_oracle_pct=100*patch,
            total_headroom_vs_event_pp=100*headroom,gate_only_pp=100*gate_only,
            shared_event_bias_refit_pp=100*shared_refit,within_flood_patch_bias_pp=100*local,
            gate_only_share_pct=share(gate_only,headroom),gate_plus_refit_share_pct=share(refit-ev,headroom),
            global_gate_only_pp=100*(gated_global-global_),
            global_gate_only_share_pct=share(gated_global-global_,global_headroom)))
    return output


def plot_decomposition(out, components):
    data = [row for row in components if (row["scope"]=="event" and row["metric"]=="flood_iou")
            or row["metric"]=="macro_flood_iou"]
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(9,4.8),layout="constrained")
    baseline = np.zeros(len(data))
    for key,label,color in (
        ("gate_only_pp","Gate flood-free patches", "#27678c"),
        ("shared_event_bias_refit_pp","Refit shared event bias", "#56a393"),
        ("within_flood_patch_bias_pp","Tune bias within each flood patch", "#bc7039"),
    ):
        values = np.array([r[key] for r in data])
        ax.barh(y,values,left=baseline,label=label,color=color)
        baseline += values
    ax.set(yticks=y,yticklabels=[row["event_id"] or "Macro average" for row in data],
           xlabel="Flood IoU gain from Event Oracle (pp)")
    ax.invert_yaxis()
    ax.legend(loc="upper center",bbox_to_anchor=(.5,-.12),ncols=2,fontsize=8)
    fig.savefig(out/"gating_decomposition.png",dpi=165)
    plt.close(fig)


def run(spatial, cache, out):
    if out.exists():
        raise FileExistsError(out)
    recipe, selection, _, expected = reference(load_confusions=False)
    events = sorted(expected,key=int)
    metadata, logits = open_predictions(cache,recipe,selection,events)
    patches, historical, old_groups = load_spatial(spatial,cache,recipe,metadata)
    totals = {m:np.zeros((3,3),dtype=np.int64) for m in METHODS}
    ranks = {m:[] for m in METHODS}
    raw_ranks, events_rows, patch_rows, components = [], [], [], []
    for event in events:
        labels = cache_file(cache,metadata,event,labels=True)
        cms, rank, raw_rank, event_rows, p_rows = audit_event(
            event,logits[event],labels,patches[event],historical,old_groups,recipe["flood_logit_bias"])
        for method in METHODS:
            totals[method] += cms[method]
            ranks[method].append(rank[method])
        raw_ranks.append(raw_rank)
        components.extend(decompose(event_rows,("flood_iou","flood_f1","pooled_miou"),"event",event))
        events_rows.extend(event_rows)
        patch_rows.extend(p_rows)
        print(f"Verified {event}: " + ", ".join(f"{r['method']}={r['flood_iou']:.5f}" for r in event_rows),flush=True)
    raw_ap = average_precision(rank_pooled(raw_ranks))
    summary = []
    for method in METHODS:
        rows = [r for r in events_rows if r["method"]==method]
        tp,fp,fn = flood_counts(totals[method])
        ious = [r["flood_iou"] for r in rows]
        summary.append(dict(method=method,events=len(events),tp=tp,fp=fp,fn=fn,
             fp_flood_free=sum(r["fp_flood_free"] for r in rows),
             fp_flood_containing=sum(r["fp_flood_containing"] for r in rows),
             **scores(totals[method]),macro_flood_iou=float(np.mean(ious)),event_iou_std=float(np.std(ious)),
             worst_event_iou=min(ious),flood_ap_raw=raw_ap,
             flood_ap_adjusted=average_precision(rank_pooled(ranks.pop(method)))))
    components.extend(decompose(summary,CORE_METRICS,"pooled"))
    archived = by_key(read_csv(spatial/"summary.csv"),("method",))
    for row in summary:
        method = row["method"]
        if method in (GLOBAL,EVENT,PATCH):
            keys = ("pooled_miou","flood_iou","flood_f1","macro_flood_iou","event_iou_std","flood_ap_adjusted")
            archived_keys = ("pooled_miou","pooled_flood_iou","pooled_flood_f1","event_macro_flood_iou",
                             "event_iou_std","pooled_flood_ap_adjusted")
            checked([row[k] for k in keys],[float(archived[method,][k]) for k in archived_keys],
                    f"{method} archived pooled metrics")
    out.mkdir(parents=True,exist_ok=False)
    for name, rows in (
        ("summary",summary),("event_metrics",events_rows),("event_279",[r for r in events_rows if r["event_id"]=="279"]),
        ("patch_counts",patch_rows),("decomposition",components),
    ):
        save_csv(out/f"{name}.csv",rows)
    plot_decomposition(out,components)
    write_json(out/"evaluation_config.json",dict(
        split="validation",scope="ground-truth oracle diagnostic; never use as deployable performance",
        frozen_selection_sha256=recipe["selection_sha256"],validation_cache_sha256=digest(cache/"manifest.json"),
        spatial_results=str(spatial.resolve()),
        spatial_sha256={name:digest(spatial/name) for name in ("evaluation_config.json","patch_manifest.json",
                                                                "group_biases.csv","event_metrics.csv","summary.csv")},
        pure_gate="hold the historical Event Oracle bias fixed on patches containing flood and hard-block flood at other patches",
        gate_refit="choose one shared exact IoU-maximizing bias over patches containing flood, leaving all other patches blocked",
        full_patch="replay saved joint event-IoU-optimal patch biases",
        decomposition="Event Oracle -> fixed-bias gate -> gated refitted shared bias -> Full Patch Oracle",
        pooled_flood_ap="exact grouped-tie AP; gated scores tie at negative infinity; its gain uses ground truth",
        case_without_scored_pixels="excluded from counts and marked in the inventory",
        ratio_when_zero_headroom="null",
        code_sha256=digest(Path(__file__))))
    print(f"Saved Patch Oracle decomposition: {out}",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("spatial-results","cache","out"):
        parser.add_argument(f"--{name}",type=Path,required=True)
    args = parser.parse_args()
    run(args.spatial_results,args.cache,args.out)


if __name__ == "__main__":
    main()
