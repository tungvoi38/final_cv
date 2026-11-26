"""Evaluation and event-level sensitivity reports for a frozen CCLA ensemble."""
import numpy as np

from adapt_unet_event_bias import digest, load_json, scores, write_json
from audit_unet_bias_failures import save_csv
from ccla_crossfit_analysis import (
    EVENT, GLOBAL, METHODS, PATCH, VARIANTS, lock_biases, metrics_and_bootstrap, score_event,
)
from ccla_crossfit_cache import values
from diagnose_unet_spatial_oracle import average_precision
from evaluate_unet_eoa_diagnostics import rank_pooled

PAIRS = [(m, r) for m in VARIANTS for r in METHODS if m != r]


def outcome_reports(rows):
    events = sorted({r["event_id"] for r in rows},key=int)
    lookup = {(r["method"],r["event_id"]):r["flood_iou"] for r in rows}
    details, summary = [],[]
    for method,ref in PAIRS:
        totals = dict(win=0,tie=0,loss=0)
        for event in events:
            a,b = lookup[method,event],lookup[ref,event]
            sign = "win" if a-b>1e-12 else "loss" if b-a>1e-12 else "tie"
            totals[sign] += 1
            details.append(dict(method=method,reference=ref,event_id=event,flood_iou=a,
                                reference_iou=b,change_pp=100*(a-b),outcome=sign))
        summary.append(dict(method=method,reference=ref,events=len(events),**totals))
    return summary,details


def stat_rows(cms, ranks, include):
    macro, result = {},{}
    for method in METHODS:
        matrices = [cms[e][method] for e in include]
        pooled = scores(sum(matrices))
        iou = [scores(cm)["flood_iou"] for cm in matrices]
        macro[method] = float(np.mean(iou))
        result[method] = dict(
            pooled_miou=pooled["pooled_miou"],pooled_flood_iou=pooled["flood_iou"],
            pooled_flood_f1=pooled["flood_f1"],pooled_flood_precision=pooled["flood_precision"],
            pooled_flood_recall=pooled["flood_recall"],event_macro_flood_iou=macro[method],
            event_flood_iou_std=float(np.std(iou)),worst_event_iou=min(iou),
            worst_event=include[int(np.argmin(iou))],
            pooled_ap=average_precision(rank_pooled([ranks[e][method] for e in include])))
    for m,row in result.items():
        row.update(mean_event_oracle_gap_pp=100*(macro[EVENT]-macro[m]),
                   mean_patch_oracle_gap_pp=100*(macro[PATCH]-macro[m]))
    return result


def sensitivity(cms, ranks):
    events = list(cms)
    rows, paired = [],[]
    for omitted in (None,*events):
        include = [e for e in events if e!=omitted]
        stat = stat_rows(cms,ranks,include)
        for method,values_ in stat.items():
            rows.append(dict(omitted_event=omitted or "none",events=len(include),method=method,**values_))
        for method,ref in PAIRS:
            for key,a in stat[method].items():
                b = stat[ref][key]
                if key=="worst_event" or a is None or b is None:
                    continue
                scale = 1 if key.endswith("_pp") else 100
                paired.append(dict(omitted_event=omitted or "none",events=len(include),method=method,
                    reference=ref,metric=key,delta_pp=scale*(a-b)))
    return rows,paired


def evaluate_ensemble(oof,validation,predictor,out,identity):
    if out.exists():
        raise FileExistsError(out)
    locks,index,patches,logits = lock_biases(oof,validation,predictor)
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/"bias_lock.json",identity | dict(events=locks,
               source_reference="27 OOF ensemble training events, source macro Flood IoU",
               iw="OOF training labels and unlabeled held-out covariates",historical_global_bias=.95))
    locks = load_json(out/"bias_lock.json")["events"]
    cms,ranks,rows,oracles = {},{},[],[]
    for e,lock in locks.items():
        y = values(index,e,"labels")
        cms[e],ranks[e],scored,patch_oracle = score_event(e,logits[e],y,patches[e],lock)
        rows.extend(scored)
        oracles.extend(patch_oracle)
        print(f"Scored validation event {e}",flush=True)
    _,ci,paired = metrics_and_bootstrap(cms)
    wtl,delta = outcome_reports(rows)
    loo,loo_paired = sensitivity(cms,ranks)
    summary = [{key:value for key,value in r.items() if key!="omitted_event"} for r in loo
               if r["omitted_event"]=="none"]
    for name,data in (("summary",summary),("event_metrics",rows),("event_279",[r for r in rows if r["event_id"]=="279"]),
                      ("patch_oracle_biases",oracles),("bootstrap",ci),("paired_bootstrap",paired),
                      ("win_tie_loss",wtl),("event_differences",delta),("leave_one_event_out",loo),
                      ("leave_one_event_out_paired",loo_paired)):
        save_csv(out/f"{name}.csv",data)
    np.savez_compressed(out/"event_confusions.npz",event_ids=list(cms),methods=METHODS,
                        confusion=np.stack([[cms[e][m] for m in METHODS] for e in cms]))
    write_json(out/"evaluation_config.json",identity | dict(split="validation",patches=959,
        protocol="three seed epoch-20 SMA, mean FP32 logits after per-member Flip TTA",
        ap="exact pooled AP with grouped ties; recalculated in every omitted-event subset",
        bootstrap="10000 paired whole-event percentile replicates, seed 42; core confusion metrics; AP point estimates",
        sensitivity="fixed locked biases and oracles; omit events only from metric aggregation",
        oracle="diagnostic only, ground-truth event scalar and joint event-IoU patch optimizer",
        bias_lock_sha256=digest(out/"bias_lock.json")))
    print(f"Saved three-seed CCLA results: {out}",flush=True)
