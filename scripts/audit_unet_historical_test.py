#!/usr/bin/env python3
"""Paired event-level audit of already scored historical test predictions."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adapt_unet_event_bias import digest, load_json, scores, write_json
from adapt_unet_graph_cut import tail_bootstrap
from audit_ccla_scalar_bias import stats
from audit_unet_bias_failures import checked, read_csv, save_csv
from bootstrap_unet_events import ITERATIONS, SEED, interval
from evaluate_unet_best_methods_test import EVENT, F1, GLOBAL, IW_MARGIN, MARGIN, REF, SCOPE, footprint
from kuro_dataset import read_metadata

PRIMARY = ((IW_MARGIN,GLOBAL),(IW_MARGIN,REF),(MARGIN,F1))
PAIRS = (*PRIMARY,(IW_MARGIN,F1))
METRICS = {
    "pooled_miou":"Pooled mIoU","pooled_flood_f1":"Pooled Flood F1",
    "event_macro_flood_iou":"Macro Flood IoU","event_flood_iou_std":"Event IoU std",
    "worst_event_iou":"Worst-event IoU","lower_tail_iou":"Lower-tail IoU",
}
NAMES = {IW_MARGIN:"IW margin",GLOBAL:"Global +0.95",REF:"Source Reference +3.70",
         MARGIN:"CCLA margin",F1:"OOF Scalar F1 +1.50",EVENT:"Event Oracle"}
TOL = 1e-12
ORACLE_METRICS = ("event_macro_flood_iou","event_flood_iou_std","worst_event_iou")


def read_study(results):
    root = results.parent
    cfg = load_json(results/"evaluation_config.json")
    paths = {n:root/f"{n}.json" for n in ("frozen_config","test_predictions_lock","test_bias_lock",
                                         "test_manifest","footprint_audit")}
    if (cfg["scope"]!=SCOPE or cfg["split"]!="test" or cfg["patches"]!=2000 or cfg["events"]!=10
            or cfg["test_predictions_lock_sha256"]!=digest(paths["test_predictions_lock"])
            or cfg["fingerprint_sha256"]!=digest(paths["test_manifest"])
            or cfg["footprint_audit_sha256"]!=digest(paths["footprint_audit"])):
        raise ValueError("Archived results do not match the historical test locks")
    frozen = load_json(paths["frozen_config"])
    locks = load_json(paths["test_predictions_lock"])
    bias = load_json(paths["test_bias_lock"])
    if (locks["bias_lock_sha256"]!=digest(paths["test_bias_lock"]) or locks["test_labels_used"]
            or bias["test_labels_used"] or locks["frozen_config_sha256"]!=digest(paths["frozen_config"])):
        raise ValueError("Provenance of label-free predictions changed")
    scope = load_json(paths["test_manifest"])
    needed = {m for pair in PAIRS for m in pair}|{EVENT}
    with np.load(results/"event_confusions.npz",allow_pickle=False) as data:
        events, methods, cms = data["event_ids"].tolist(),data["methods"].tolist(),data["confusion"]
    if (len(events)!=10 or len(set(events))!=10 or set(events)!=set(scope["events"]) or "411" not in events
            or set(bias["events"])!=set(events) or not needed.issubset(methods)
            or len(methods)!=len(set(methods)) or cms.shape!=(10,len(methods),3,3)
            or np.any(cms<0) or not np.issubdtype(cms.dtype,np.integer)):
        raise ValueError("Invalid paired confusion matrix archive")
    mats = {m:cms[:,methods.index(m)] for m in needed}
    data = read_csv(results/"event_metrics.csv")
    rows = {(r["event_id"],r["method"]):r for r in data}
    if len(rows)!=len(data):
        raise ValueError("Duplicate test metric rows")
    coverage = {r["event_id"]:r for r in read_csv(results/"test_coverage.csv")}
    ref_bias = {r["event_id"]:float(r["bias"]) for r in read_csv(results/"oracle_DIAGNOSTIC_ONLY/event_biases.csv")}
    for i,e in enumerate(events):
        support = int(coverage[e]["scored_pixels"])
        poses = int(coverage[e]["flood_pixels"])
        for m in needed:
            cm = mats[m][i]
            r = rows[e,m]
            tp,fp,fn = int(cm[2,2]),int(cm[:2,2].sum()),int(cm[2,:2].sum())
            np.testing.assert_array_equal([support,poses,tp,fp,fn],
                                          [cm.sum(),tp+fn,int(r["tp"]),int(r["fp"]),int(r["fn"])])
            expected = scores(cm)
            checked(list(expected.values()),[float(r[k]) for k in expected],f"{e}/{m} saved scores")
    for e,lock in bias["events"].items():
        checked([lock["scalar_biases"][m] for m in (GLOBAL,REF,F1)],[.95,3.70,1.50],f"{e} scalar bias")
    return paths,frozen,scope,events,mats,bias["events"],coverage,ref_bias


def verify_footprint(paths,frozen,scope):
    old = load_json(paths["footprint_audit"])
    frame = read_metadata(Path(frozen["raw"]))  # Metadata only, no rasters or labels.
    test = frame[frame["tortilla:data_split"]=="test"]
    if test["tortilla:id"].astype(str).tolist()!=scope["patch_ids"]:
        raise ValueError("Official test patch traversal changed")
    pairs,counts = footprint(frame)
    if counts!=old["overlap_pair_counts"] or pairs!=old["pairs"]:
        raise ValueError("Archived footprint audit disagrees with pinned metadata")
    overlap = {r["test_patch"] for r in pairs if r["other_split"]=="train"}
    return dict(projection=old["projection"],counts=counts,pairs=pairs,
        train_test_overlap_patches=len(overlap),footprint_clean_test_patches=2000-len(overlap),
        footprint_clean_scope="identical to the official 2000 patches" if not overlap else
                              "not recoverable from event-level confusions alone",
        parent_scene_independence="unverified")


def all_stats(mats,weights):
    out = {}
    for m,cm in mats.items():
        d = stats(cm,weights)
        ious = np.array([scores(c)["flood_iou"] for c in cm])
        d["lower_tail_iou"] = tail_bootstrap(ious,weights)
        out[m] = d
    return out


def bias_fields(e,m,locks,ref):
    if m==EVENT:
        b = [ref[e]]
    else:
        q = locks[e]
        b = [q["scalar_biases"][m]] if m in q["scalar_biases"] else list(q["patch_biases"][m].values())
    return dict(bias_type="patch" if m==MARGIN else "scalar",bias_n=len(b),bias_median=float(np.median(b)),
                bias_q10=float(np.quantile(b,.1)),bias_q90=float(np.quantile(b,.9)))


def analyze(events,mats,locks,coverage,oracle):
    n = len(events)
    omits = np.vstack([np.ones(n,dtype=int),1-np.eye(n,dtype=int)])
    point = all_stats(mats,omits)
    picks = np.random.default_rng(SEED).integers(n,size=(ITERATIONS,n))
    boot = all_stats(mats,(picks[...,None]==np.arange(n)).sum(axis=1))
    summaries = [dict(method=m,**{k:float(v[0]) for k,v in point[m].items()})
                 for m in sorted(point)]
    detailed,ci,sensitivity,wins,gap,aggregate = [],[],[],[],[],[]
    for method,ref in PAIRS:
        a,b = mats[method],mats[ref]
        apoint,bpoint = point[method],point[ref]
        primary = (method,ref) in PRIMARY
        ident = dict(method=method,reference=ref,comparison="primary" if primary else "secondary")
        outcomes = dict(win=0,tie=0,loss=0)
        for i,e in enumerate(events):
            x,y = scores(a[i]),scores(b[i])
            delta = x["flood_iou"]-y["flood_iou"]
            outcome = "win" if delta>TOL else "loss" if delta<-TOL else "tie"
            outcomes[outcome]+=1
            ac,bc = a[i],b[i]
            t0,f0,n0 = int(bc[2,2]),int(bc[:2,2].sum()),int(bc[2,:2].sum())
            t1,f1,n1 = int(ac[2,2]),int(ac[:2,2].sum()),int(ac[2,:2].sum())
            detailed.append(dict(**ident,event_id=e,patches=int(coverage[e]["patches"]),scored_pixels=int(coverage[e]["scored_pixels"]),
                outcome=outcome,method_iou=x["flood_iou"],reference_iou=y["flood_iou"],delta_iou_pp=100*delta,
                event_oracle_iou=scores(mats[EVENT][i])["flood_iou"],event_oracle_bias=oracle[e],
                **{f"method_{k}":v for k,v in bias_fields(e,method,locks,oracle).items()},
                **{f"reference_{k}":v for k,v in bias_fields(e,ref,locks,oracle).items()},
                method_precision=x["flood_precision"],reference_precision=y["flood_precision"],
                precision_delta_pp=100*(x["flood_precision"]-y["flood_precision"]),method_recall=x["flood_recall"],
                reference_recall=y["flood_recall"],recall_delta_pp=100*(x["flood_recall"]-y["flood_recall"]),
                method_tp=t1,method_fp=f1,method_fn=n1,reference_tp=t0,reference_fp=f0,reference_fn=n0,
                delta_tp=t1-t0,delta_fp=f1-f0,delta_fn=n1-n0))
        wins.append(ident|dict(events=n,**outcomes))
        sa,sb = a.sum(0),b.sum(0)
        aggregate.append(ident|dict(delta_tp=int(sa[2,2]-sb[2,2]),delta_fp=int(sa[:2,2].sum()-sb[:2,2].sum()),
                 delta_fn=int(sa[2,:2].sum()-sb[2,:2].sum()),
                 delta_precision_pp=100*(apoint["pooled_flood_precision"][0]-bpoint["pooled_flood_precision"][0]),
                 delta_recall_pp=100*(apoint["pooled_flood_recall"][0]-bpoint["pooled_flood_recall"][0])))
        for metric in METRICS:
            da = apoint[metric]-bpoint[metric]
            lo,hi,k = interval(100*(boot[method][metric]-boot[ref][metric]))
            ci.append(ident|dict(metric=metric,iterations=k,delta_pp=100*da[0],ci_low_pp=lo,ci_high_pp=hi))
            for j,e in enumerate(("none",*events)):
                sensitivity.append(ident|dict(omitted_event=e,events=n-int(j>0),metric=metric,
                    method_value=apoint[metric][j],reference_value=bpoint[metric][j],
                    delta_pp=100*da[j],change_vs_full_pp=100*(da[j]-da[0])))
        for j,omit in ((0,"none"),(events.index("411")+1,"411")):
            for metric in ORACLE_METRICS:
                o,av,bv = point[EVENT][metric][j],apoint[metric][j],bpoint[metric][j]
                # Std is a distance to the reference, since Event Oracle maximizes each event's IoU.
                ga,gb = (abs(av-o),abs(bv-o)) if metric=="event_flood_iou_std" else (o-av,o-bv)
                gap.append(ident|dict(omitted_event=omit,metric=metric,oracle_value=o,method_value=av,reference_value=bv,
                    method_gap_pp=100*ga,reference_gap_pp=100*gb,closed_gap_pp=100*(gb-ga),
                    closed_gap_pct=100*(gb-ga)/gb if gb>TOL else None,
                    gap_definition="absolute distance" if metric=="event_flood_iou_std" else "oracle minus method"))
    return dict(summary=summaries,paired_bootstrap=ci,win_tie_loss=wins,leave_one_event_out=sensitivity,
                event_differences=detailed,event_411=[r for r in detailed if r["event_id"]=="411"],
                event_oracle_gap=gap,pooled_confusion_deltas=aggregate)


def description(method,ref):
    return f"{NAMES[method]} vs {NAMES[ref]}"


def plots(out,events,result):
    fig,axes = plt.subplots(2,3,figsize=(14,6.8),sharey=True,layout="constrained")
    labels = [description(*pair) for pair in PRIMARY]
    for ax,(metric,label) in zip(axes.flat,METRICS.items(),strict=True):
        rows = [r for r in result["paired_bootstrap"] if r["metric"]==metric and r["comparison"]=="primary"]
        for i,r in enumerate(rows):
            ax.hlines(i,r["ci_low_pp"],r["ci_high_pp"],color="#356b83")
            ax.plot(r["delta_pp"],i,"o",color="#356b83")
        ax.axvline(0,ls="--",color=".5")
        ax.set(title=label, xlabel="Difference (pp)",yticks=range(len(labels)),yticklabels=labels)
    axes.flat[0].invert_yaxis()
    fig.savefig(out/"paired_bootstrap.png",dpi=170)
    plt.close(fig)

    fig,axes = plt.subplots(2,2,figsize=(12,6.5),layout="constrained")
    columns = ("none",*events)
    lookup_ = {(r["method"],r["reference"],r["omitted_event"],r["metric"]):r["delta_pp"]
               for r in result["leave_one_event_out"]}
    for ax,key in zip(axes.flat,("pooled_miou","pooled_flood_f1","event_macro_flood_iou","lower_tail_iou"),strict=True):
        for pair in PRIMARY:
            ax.plot(columns,[lookup_[*pair,e,key] for e in columns],"o-",lw=1,label=description(*pair))
        ax.axhline(0,ls="--",color=".55")
        ax.axvspan(columns.index("411")-.35,columns.index("411")+.35,color="#dbba7d",alpha=.32)
        ax.set(title=METRICS[key],xlabel="Omitted event",ylabel="Difference (pp)")
        ax.tick_params(axis="x",rotation=35,labelsize=8)
    axes[0,0].legend(fontsize=7)
    fig.savefig(out/"leave_one_event_out.png",dpi=170)
    plt.close(fig)

    fig,axes = plt.subplots(3,1,figsize=(11,7.5),sharex=True,layout="constrained")
    for ax,pair in zip(axes,PRIMARY,strict=True):
        rows = [r for r in result["event_differences"] if (r["method"],r["reference"])==pair]
        delta = [r["delta_iou_pp"] for r in rows]
        ax.bar(events,delta,color=["#257c72" if v>=0 else "#b15547" for v in delta])
        ax.axhline(0,color=".55",lw=.8)
        ax.set(title=description(*pair),ylabel="Flood IoU Δ (pp)")
    axes[-1].set_xlabel("Historical test event")
    fig.savefig(out/"event_differences.png",dpi=170)
    plt.close(fig)


def markdown(out, result, geom):
    ci = {(r["method"],r["reference"],r["metric"]):r for r in result["paired_bootstrap"]}
    loo = {(r["method"],r["reference"],r["omitted_event"],r["metric"]):r for r in result["leave_one_event_out"]}
    gap = {(r["method"],r["reference"],r["omitted_event"],r["metric"]):r for r in result["event_oracle_gap"]}
    d411 = {(r["method"],r["reference"]):r for r in result["event_411"]}
    wtl = {(r["method"],r["reference"]):r for r in result["win_tie_loss"]}
    lines = [
        "# Audit historical test", "",
        "Historical-test exploratory evaluation trên 2.000 patch của 10 event. Audit chỉ đọc metadata, confusion matrix, metrics và bias đã lưu.",
        "", "## Paired event bootstrap", "",
        "| So sánh | Metric | Chênh lệch (pp) | CI 95% (pp) | Bỏ event 411 (pp) |",
        "|---|---|---:|---:|---:|",
    ]
    for pair in PAIRS:
        for key,label in METRICS.items():
            c = ci[*pair,key]
            omit = loo[*pair,"411",key]
            lines.append(f"| {description(*pair)} | {label} | {c['delta_pp']:+.3f} | "
                         f"[{c['ci_low_pp']:+.3f}, {c['ci_high_pp']:+.3f}] | {omit['delta_pp']:+.3f} |")
    lines += ["","CI dùng 10.000 mẫu bootstrap ghép cặp theo event. Lower-tail lấy trung bình ceil(E/3) event yếu nhất. "
              "Bỏ 411 làm kích thước lower-tail đổi từ 4 còn 3. Phân tích bỏ event giữ nguyên mọi bias.","",
              "## Event 411 và khoảng cách tới Event Oracle",""]
    for pair in PRIMARY:
        row,w = d411[pair],wtl[pair]
        lines.append(f"- **{description(*pair)}** thắng/hòa/thua {w['win']}/{w['tie']}/{w['loss']}. "
                     f"Ở 411, bias median {row['reference_bias_median']:+.2f} → {row['method_bias_median']:+.2f}, "
                     f"ΔIoU {row['delta_iou_pp']:+.2f} pp, ΔTP/FP/FN "
                     f"{row['delta_tp']:+d}/{row['delta_fp']:+d}/{row['delta_fn']:+d}.")
        g = [gap[*pair,"none",k]["closed_gap_pct"] for k in ORACLE_METRICS]
        q = [f"{v:+.1f}%" if v is not None else "NA" for v in g]
        lines.append(f"  Tỷ lệ thu hẹp khoảng cách oracle cho macro IoU, std và worst IoU lần lượt là {q[0]}, {q[1]} và {q[2]}.")
    lines += ["","Tỷ lệ âm biểu thị khoảng cách tăng. Với std, khoảng cách là trị tuyệt đối tới std của Event Oracle. "
              "Event Oracle tối ưu IoU từng event và std của nó chỉ là mốc chẩn đoán.","",
              "## Footprint","",
              f"Train–test có {geom['counts']['train']} cặp giao footprint. Validation–test có {geom['counts']['validation']}. "
              f"Tập sạch footprint theo train gồm {geom['footprint_clean_test_patches']} patch. "
              f"Có {geom['counts']['test']} cặp giao footprint nội bộ test được giữ theo split chính thức. "
              "Tính độc lập theo parent scene chưa được xác minh.","",
              "## Kết luận",""]
    vs_macro = ci[IW_MARGIN,REF,"event_macro_flood_iou"]
    vs_worst = ci[IW_MARGIN,REF,"worst_event_iou"]
    vs_f1 = ci[IW_MARGIN,F1,"pooled_flood_f1"]
    loc = ci[MARGIN,F1,"pooled_flood_f1"]
    lines += [
        f"- IW-IoU so với OOF Source Reference chênh {vs_macro['delta_pp']:+.2f} pp macro IoU và "
        f"{vs_worst['delta_pp']:+.2f} pp worst-event IoU. Trên tập lịch sử này, kết quả chưa cho thấy IW-IoU vượt "
        "scalar tối ưu macro IoU một cách nhất quán.",
        f"- So với scalar OOF tối ưu F1, IW-IoU chênh {vs_f1['delta_pp']:+.2f} pp pooled F1. "
        f"CCLA chênh {loc['delta_pp']:+.2f} pp, CI [{loc['ci_low_pp']:+.2f}, {loc['ci_high_pp']:+.2f}] pp.",
        "", "[CSV theo event](event_differences.csv) · [Precision/Recall và TP/FP/FN gộp](pooled_confusion_deltas.csv) "
        "· [Khoảng cách oracle](event_oracle_gap.csv) · [Paired bootstrap](paired_bootstrap.csv)"
    ]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    result = a.results.resolve()
    paths,frozen,scope,events,mats,bias,coverage,oracle = read_study(result)
    geom = verify_footprint(paths,frozen,scope)
    rows = analyze(events,mats,bias,coverage,oracle)
    a.out.mkdir(parents=True,exist_ok=False)
    for name,data in rows.items():
        save_csv(a.out/f"{name}.csv",data)
    write_json(a.out/"footprint_check.json",geom)
    write_json(a.out/"audit_config.json",dict(scope=SCOPE,results=str(result),events=events,
        input_sha256={str(path):digest(path) for path in (*paths.values(),result/"evaluation_config.json",
        result/"event_confusions.npz",result/"event_metrics.csv", result/"test_coverage.csv",
        result/"oracle_DIAGNOSTIC_ONLY/event_biases.csv")},
        bootstrap_iterations=ITERATIONS,seed=SEED,bootstrap="paired whole events, percentile 95%",
        pairwise="candidate minus reference; lower std is preferred",metric_omission="fixed saved predictions only",
        primary=[list(p) for p in PRIMARY],secondary=[IW_MARGIN,F1],script_sha256=digest(Path(__file__))))
    plots(a.out,events,rows)
    markdown(a.out,rows,geom)
    print(f"Saved historical-test audit: {a.out}",flush=True)


if __name__=="__main__":
    main()
