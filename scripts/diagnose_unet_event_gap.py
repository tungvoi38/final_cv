#!/usr/bin/env python3
"""Exploratory validation–historical-test gap audit. Raster and saved metrics only."""
import argparse
from itertools import combinations
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rasterio.warp import transform
from scipy.stats import spearmanr

from bootstrap_unet_events import read_csv, read_json, verify, recover_flood_counts
from diagnose_sar_scattering import texture, slope_degrees, save_csv
from kuro_dataset import KuroDataset, RawArchive, RAW_SHA256, REVISION

REPO = Path(__file__).resolve().parents[1]
SPLITS = ("validation", "test")
CLASSES = ("background", "permanent_water", "flood")
SAR_NAMES = [f"{time}_{band}_db" for time in ("pre1", "pre2", "post") for band in ("vv", "vh")]
RADIOMETRY = SAR_NAMES + ["delta_vv_db", "delta_vh_db", "post_vh_minus_vv_db"]
TEXTURE = [f"post_{band}_variance_{k}x{k}_db2" for k in (3, 5) for band in ("vv", "vh")]
FEATURES = RADIOMETRY + ["slope_deg"] + TEXTURE
EDGES = {key: np.linspace(-400, 400, 16001) for key in RADIOMETRY}
EDGES["slope_deg"] = np.linspace(0, 90, 361)
EDGES.update({key: np.linspace(0, 18, 721) for key in TEXTURE})
TARGETS = ("flood_ap", "flood_iou", "flood_f1", "precision", "recall", "false_positive_rate",
           "oracle_bias_iou_DIAGNOSTIC_ONLY")


def divide(a, b):
    return a / b if b else None


def rates(tp, fp, fn, total):
    return dict(flood_iou=divide(tp, tp+fp+fn), flood_f1=divide(2*tp,2*tp+fp+fn),
                precision=divide(tp,tp+fp), recall=divide(tp,tp+fn),
                false_positive_rate=divide(fp,total-tp-fn))


def summarize_events(root, recipe_path):
    recipe = read_json(recipe_path)
    bias = recipe["flood_logit_bias"]
    val = root / "a3-eoa-flip-tta-q8H1dB/results"
    test = root / "a3-eoa-flood-diagnostics-YtaAJg/results"
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if (sha(val / "protocol.json") != recipe["validation_protocol_sha256"]
            or sha(val / "bias_selection.json") != recipe["validation_bias_lock_sha256"]
            or read_json(test / "protocol.json")["recipe_sha256"] != sha(recipe_path)
            or any(read_json(p / "protocol.json")["selection_sha256"] != recipe["selection_sha256"]
                   for p in (val, test))):
        raise ValueError("Artifacts do not match the frozen Flip TTA selection")
    metrics = {}
    val_npz = val / "validation_confusion_eoa_flip_tta.npz"
    with np.load(val_npz) as data:
        grid = data["biases"]
        idx = {b:int(np.flatnonzero(np.isclose(grid,b,rtol=0,atol=1e-12)).item()) for b in (0.,bias)}
        for event in set(data.files)-{"biases"}:
            cm = data[event]
            total = int(cm[idx[bias]].sum())
            tc = cm[idx[bias]].sum(-1)
            tp = cm[:,2,2]
            union = cm[:,2,:].sum(-1) + cm[:,:,2].sum(-1) - tp
            iou = tp / np.maximum(union,1)
            winner = min(np.flatnonzero(iou==iou.max()), key=lambda i:(abs(grid[i]),grid[i]))
            for b, k in idx.items():
                t = int(tp[k])
                row = dict(split="validation",event_id=event,bias=b,tp=t,
                           fp=int(cm[k,:,2].sum()-t),fn=int(cm[k,2,:].sum()-t),
                           valid_pixels=total, expected_classes=tc.tolist(), flood_ap=None,
                           oracle_bias_iou_DIAGNOSTIC_ONLY=float(grid[winner]),
                           oracle_flood_iou_DIAGNOSTIC_ONLY=float(iou[winner]), metric_source=str(val_npz))
                metrics[("validation",event,b)] = row
    pr = {r["scope"]:float(r["average_precision"]) for r in read_csv(test/"test_flood_pr_summary.csv")}
    historical = read_csv(test/"test_event_oracle_DIAGNOSTIC_ONLY.csv")
    for r in historical:
        event=r["event_id"]
        n,g=int(r["valid_pixels"]),int(r["flood_pixels"])
        if float(r["validation_global_bias"]) != bias:
            raise ValueError("Validation-selected bias mismatch")
        for b, label in ((0.,"zero"),(bias,"validation_bias")):
            path=test/f"pr_exact/{event}.npz"
            tp,fp,fn=recover_flood_counts(path,b,g,n)
            verify(tp/(tp+fp+fn),float(r[f"flood_iou_{label}"]),str(path))
            metrics[("test",event,b)] = dict(split="test",event_id=event,bias=b,tp=tp,fp=fp,fn=fn,
                valid_pixels=n,expected_classes=None,flood_ap=pr[event],
                oracle_bias_iou_DIAGNOSTIC_ONLY=float(r["oracle_bias_iou_DIAGNOSTIC_ONLY"]),
                oracle_flood_iou_DIAGNOSTIC_ONLY=float(r["oracle_flood_iou_DIAGNOSTIC_ONLY"]),metric_source=str(path))
    return metrics,bias


def collect_histograms(archive):
    counts, coverage = {}, {}
    hist = {}
    for split in SPLITS:
        ds=KuroDataset(archive,split)
        affine=np.stack(ds.rows["stac:geotransform"])
        if not (ds.rows["stac:crs"].eq("EPSG:3857").all()
                and np.all(affine[:,[1,2,4,5]]==[10,0,0,-10])):
            raise ValueError("Unexpected native raster grid")
        _, lat = transform("EPSG:3857","EPSG:4326",affine[:,0]+1120,affine[:,3]-1120)
        for i in range(len(ds)):
            sample=ds.physical(i)
            key=(split,sample["event_id"])
            valid=sample["valid_mask"]
            label=sample["mask"][valid]
            counts.setdefault(key,np.zeros(3,dtype=np.int64))
            counts[key] += np.bincount(label,minlength=3)
            ground=10*np.cos(np.radians(lat[i]))
            coverage.setdefault(key,[]).append(float(ground))
            if key not in hist:
                hist[key]={name:np.zeros((3,len(edges)-1),dtype=np.int64) for name,edges in EDGES.items()}
            sar=sample["sar_db"]
            post=sar[2]
            delta=post-(sar[0]+sar[1])/2
            values=[*sar.reshape(6,224,224),*delta,post[1]-post[0],
                    slope_degrees(sample["dem_m"][0],sample["input_valid"],ground)]
            values.extend(np.log1p(texture(post[band],valid,k)) for k in (3,5) for band in (0,1))
            for name, value in zip(FEATURES,values,strict=True):
                v=value[valid]
                exists=np.isfinite(v)
                edges=EDGES[name]
                m=len(edges)-1
                if exists.any() and (v[exists].min()<edges[0] or v[exists].max()>edges[-1]):
                    raise ValueError(f"Histogram bounds for {name} in {key}")
                bins=np.minimum(np.searchsorted(edges,v[exists],side="right")-1,m-1)
                hist[key][name] += np.bincount(label[exists].astype(np.int64)*m+bins,minlength=3*m).reshape(3,m)
            if (i+1)%250==0 or i+1==len(ds):
                print(f"Read {split} raster {i+1}/{len(ds)}",flush=True)
    if tuple(sum(s==split for s,_ in counts) for split in SPLITS)!=(6,10):
        raise ValueError("Unexpected event coverage")
    return counts,coverage,hist


def quantiles(hist, edges, as_texture):
    n=hist.sum()
    if not n:
        return [None]*5
    t=(edges[:-1]+edges[1:])/2
    x=t[np.searchsorted(hist.cumsum(),np.array([.05,.25,.5,.75,.95])*n,side="left")]
    return (np.expm1(x) if as_texture else x).tolist()


def w1(a,b,edges):
    na,nb=a.sum(),b.sum()
    if not na or not nb:
        return None
    return float(np.dot(np.abs(np.cumsum(a/na-b/nb)[:-1]),np.diff(edges)[:-1]))


def compare_histograms(hist):
    keys=sorted(hist,key=lambda k:(k[0],int(k[1])))
    summary,pairs,split_summary=[],[],[]
    reference={}
    opposite={}
    for name in FEATURES:
        edges=EDGES[name]
        metric_space="log1p(dB²)" if name in TEXTURE else "degree" if name=="slope_deg" else "dB"
        for c,cls in enumerate(CLASSES):
            by_split={split:[hist[k][name][c] for k in keys if k[0]==split] for split in SPLITS}
            prob={split:[b/b.sum() for b in hs if b.sum()] for split,hs in by_split.items()}
            means={s:np.mean(v,axis=0) if v else np.zeros(len(edges)-1) for s,v in prob.items()}
            for k in keys:
                a=hist[k][name][c]
                q=quantiles(a,edges,name in TEXTURE)
                other="test" if k[0]=="validation" else "validation"
                opposite[k,cls,name]=w1(a,means[other],edges)
                summary.append(dict(split=k[0],event_id=k[1],cls=cls,feature=name,n=int(a.sum()),
                                    q05=q[0],q25=q[1],median=q[2],q75=q[3],q95=q[4],
                                    w1_to_other_split_event_balanced=opposite[k,cls,name],w1_units=metric_space))
                reference[k,cls,name]=q[2]
            for a,b in combinations(keys,2):
                value=w1(hist[a][name][c],hist[b][name][c],edges)
                pairs.append(dict(split_a=a[0],event_a=a[1],split_b=b[0],event_b=b[1],cls=cls,
                                  feature=name,wasserstein=value,units=metric_space,
                                  n_a=int(hist[a][name][c].sum()),n_b=int(hist[b][name][c].sum())))
            values={s:np.sum(v,axis=0) for s,v in by_split.items()}
            for mode,scores in (("equal_event_weight",means),("pooled_pixels",values)):
                split_summary.append(dict(cls=cls,feature=name,aggregation=mode,
                                          wasserstein=w1(scores["validation"],scores["test"],edges),
                                          units=metric_space,validation_events=len(prob["validation"]),
                                          test_events=len(prob["test"])))
    return summary,pairs,split_summary,reference,opposite


def merge_perf(saved,counts,coverage,reference,opposite):
    events=[]
    for (split,event,bias), original in sorted(saved.items(),key=lambda r:(r[0][0],int(r[0][1]),r[0][2])):
        row=original.copy()
        key=(split,event)
        c=counts[key]
        expected=row.pop("expected_classes")
        if (row["valid_pixels"],row["tp"]+row["fn"])!=(int(c.sum()),int(c[2])):
            raise ValueError(f"Raw mask does not match saved event metrics: {key}")
        if expected is not None and expected!=c.tolist():
            raise ValueError(f"Raw classes do not match saved multiclass confusion: {key}")
        for label, count in zip(CLASSES,c,strict=True):
            row[f"{label}_pixels"]=int(count)
            row[f"{label}_prevalence"]=float(count/c.sum())
        row["patches"]=len(coverage[key])
        row["ground_pixel_spacing_m"]=float(np.median(coverage[key]))
        row.update(rates(row["tp"],row["fp"],row["fn"],row["valid_pixels"]))
        for label in CLASSES:
            for name in FEATURES:
                row[f"{label}_{name}_median"]=reference[key,label,name]
                row[f"{label}_{name}_w1_to_other_split"]=opposite[key,label,name]
        events.append(row)
    return events


def split_gap(rows,bias):
    summary=[]
    for split in SPLITS:
        for b in (0.,bias):
            group=[r for r in rows if (r["split"],r["bias"])==(split,b)]
            tp,fp,fn,n=(sum(r[key] for r in group) for key in ("tp","fp","fn","valid_pixels"))
            data=dict(split=split,bias=b,events=len(group),valid_pixels=n,tp=tp,fp=fp,fn=fn,
                      **{f"pooled_{k}":v for k,v in rates(tp,fp,fn,n).items()})
            for label in CLASSES:
                data[f"pooled_{label}_prevalence"]=sum(r[f"{label}_pixels"] for r in group)/n
                data[f"median_event_{label}_prevalence"]=float(np.median([r[f"{label}_prevalence"] for r in group]))
            for name in TARGETS[:6]:
                good=[r[name] for r in group if r[name] is not None]
                data[f"event_mean_{name}"]=float(np.mean(good)) if good else None
            summary.append(data)
    decomposition=[]
    def expected_f1(p,tpr,fpr):
        return 2*p*tpr/(p*(1+tpr)+(1-p)*fpr)
    for b in (0.,bias):
        v,t=[next(r for r in summary if r["split"]==split and r["bias"]==b) for split in SPLITS]
        pv,pt=[r["pooled_flood_prevalence"] for r in (v,t)]
        rv,rt=[(r["pooled_recall"],r["pooled_false_positive_rate"]) for r in (v,t)]
        vv,tt=expected_f1(pv,*rv),expected_f1(pt,*rt)
        vt,tv=expected_f1(pv,*rt),expected_f1(pt,*rv)
        verify([vv,tt],[v["pooled_flood_f1"],t["pooled_flood_f1"]],"split F1")
        decomposition.append(dict(bias=b,observed_gap_pp=100*(tt-vv),
            prevalence_term_pp=50*((tv-vv)+(tt-vt)),error_rate_term_pp=50*((vt-vv)+(tt-tv)),
            f1_val_prevalence_test_rates=vt,f1_test_prevalence_val_rates=tv,
            note="Descriptive symmetric substitution; no causal interpretation"))
    return summary,decomposition


def pooled_reference(ablation, recipe_path, bias, overall):
    recipe=read_json(recipe_path)
    selection_path=Path(recipe["selection"])
    if hashlib.sha256(selection_path.read_bytes()).hexdigest()!=recipe["selection_sha256"]:
        raise ValueError("Ablation selection differs from the frozen recipe")
    checkpoints=" | ".join(m["sha256"] for m in read_json(selection_path)["sma"])
    rows=[]
    for r in read_csv(ablation):
        if r["method"] not in ("EoA + Flip TTA","EoA + Flip TTA + Global Bias 0.95"):
            continue
        split={"validation":"validation","test_historical":"test"}[r["split"]]
        b=float(r["bias"])
        if r["checkpoint_sha256"]!=checkpoints or r["flip_tta"]!="True" or b not in (0.,bias):
            raise ValueError("Pooled reference differs from the frozen checkpoints or protocol")
        prior=next(v for v in overall if (v["split"],float(v["bias"]))==(split,b))
        verify(float(r["pooled_flood_f1"]),float(prior["pooled_flood_f1"]),str(ablation))
        rows.append(dict(split=split,bias=b,pooled_miou=float(r["pooled_miou"]),
                         pooled_flood_ap=float(r["pooled_flood_ap"]) if r["pooled_flood_ap"] else None,
                         flood_ap_by_event_available=(split=="test"),source=str(ablation)))
    if len({(r["split"],r["bias"]) for r in rows})!=4 or len(rows)!=4:
        raise ValueError("Incomplete pooled reference")
    return rows


def correlations(rows,bias):
    data=[r for r in rows if r["bias"]==bias]
    features=["patches","ground_pixel_spacing_m",*(f"{c}_prevalence" for c in CLASSES)]
    features += [f"{c}_{f}_{k}" for c in CLASSES for f in FEATURES for k in ("median","w1_to_other_split")]
    result=[]
    for scope in (*SPLITS,"combined"):
        group=[r for r in data if r["split"]==scope or scope=="combined"]
        for feature in features:
            for target in TARGETS:
                pairs=[(r[feature],r[target]) for r in group if r[feature] is not None and r[target] is not None]
                rho=None
                if len(pairs)>=3:
                    a,b=zip(*pairs)
                    if len(set(a))>1 and len(set(b))>1:
                        rho=float(spearmanr(a,b).statistic)
                result.append(dict(scope=scope,feature=feature,metric=target,n_events=len(pairs),
                                   spearman_rho=rho,status="exploratory" if rho is not None else "not_available"))
    return result


def plots(out,rows,bias):
    data=[r for r in rows if r["bias"]==bias]
    palettes={"validation":"#2c638b","test":"#b95420"}
    fig,axs=plt.subplots(2,4,figsize=(15,7),layout="constrained")
    box=[("flood_prevalence","Flood prevalence (%)",100),("flood_iou","Flood IoU (%)",100),
         ("precision","Precision (%)",100),("recall","Recall (%)",100),
         ("flood_post_vv_db_median","Flood post-VV (dB)",1),("flood_delta_vv_db_median","Flood ΔVV (dB)",1),
         ("flood_post_vh_minus_vv_db_median","Flood VH − VV (dB)",1),
         ("flood_slope_deg_median","Flood slope (°)",1)]
    for ax,(key,label,scale) in zip(axs.flat,box,strict=True):
        for i,s in enumerate(SPLITS,1):
            y=[r[key]*scale for r in data if r["split"]==s and r[key] is not None]
            ax.boxplot(y,positions=[i],widths=.42,showfliers=False)
            ax.scatter(i+np.linspace(-.13,.13,len(y)),y,s=16,color=palettes[s],zorder=3)
        ax.set(xticks=[1,2],xticklabels=["Validation (6)","Test (10)"],ylabel=label)
        ax.grid(axis="y",alpha=.22)
    fig.suptitle("Exploratory • one point per event • locked bias +0.95")
    fig.savefig(out/"event_split_boxplots.png",dpi=170)
    plt.close(fig)

    scatter=[("flood_prevalence","flood_f1","Flood prevalence (%)","Flood F1 (%)",100,100),
             ("flood_post_vv_db_median","recall","Flood post-VV (dB)","Recall (%)",1,100),
             ("flood_delta_vv_db_median","flood_ap","Flood ΔVV (dB)","AP (%)",1,100),
             ("flood_prevalence","oracle_bias_iou_DIAGNOSTIC_ONLY","Flood prevalence (%)","Oracle bias (diagnostic)",100,1),
             ("flood_post_vv_variance_5x5_db2_median","flood_iou","Flood VV local variance 5×5 (dB²)","Flood IoU (%)",1,100),
             ("flood_slope_deg_median","precision","Flood slope (°)","Precision (%)",1,100)]
    fig,axs=plt.subplots(2,3,figsize=(13,8),layout="constrained")
    for ax,(x,y,xlabel,ylabel,sx,sy) in zip(axs.flat,scatter,strict=True):
        for split in SPLITS:
            points=[r for r in data if r["split"]==split and r[x] is not None and r[y] is not None]
            ax.scatter([r[x]*sx for r in points],[r[y]*sy for r in points],color=palettes[split],
                       label=f"{split} ({len(points)})")
            for r in points:
                if r["event_id"] in ("411","445"):
                    ax.annotate(r["event_id"],(r[x]*sx,r[y]*sy),xytext=(4,3),textcoords="offset points",fontsize=9)
        ax.set(xlabel=xlabel,ylabel=ylabel)
        ax.grid(alpha=.22)
        ax.legend(fontsize=8)
    fig.suptitle("Exploratory event associations • per-event validation AP unavailable")
    fig.savefig(out/"event_scatter.png",dpi=170)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--artifacts",type=Path,default=REPO.parent/"artifacts")
    p.add_argument("--ablation",type=Path,default=REPO.parent/"run/a3-11ch-ablation-yd6pBu/results/ablation.csv")
    args=p.parse_args()
    if args.out.exists():
        p.error("--out must be a new directory")
    recipe=REPO/"configs/experiments/unet_a3_eoa_flip_tta.json"
    saved,bias=summarize_events(args.artifacts.resolve(), recipe)
    archive=RawArchive(args.raw)
    counts,coverage,hist=collect_histograms(archive)
    if set(counts)!={(s,e) for s,e,b in saved}:
        raise ValueError("Unexpected mismatch between raw and performance event IDs")
    summary,pairs,split,ref,other=compare_histograms(hist)
    events=merge_perf(saved,counts,coverage,ref,other)
    overall,decomposition=split_gap(events,bias)
    reference_test=read_csv(args.artifacts/"a3-eoa-flood-diagnostics-YtaAJg/results/test_paper_metrics.csv")
    for bias_point, point_name in ((0.,"b_zero"),(bias,"global_bias_selected_on_validation")):
        row=next(r for r in overall if r["split"]=="test" and r["bias"]==bias_point)
        expected=next(r for r in reference_test if r["operating_point"]==point_name)
        verify(row["pooled_flood_f1"],float(expected["pooled_flood_f1"]),"test F1")
    args.out.mkdir(parents=True,exist_ok=False)
    for file, rows in (
        ("event_performance_features.csv",events),("event_class_distributions.csv",summary),
        ("wasserstein_event_pairs.csv",pairs),("wasserstein_splits.csv",split),
        ("split_summary.csv",overall),("flood_f1_gap_decomposition.csv",decomposition),
        ("event_spearman.csv",correlations(events,bias)),
        ("split_pooled_reference.csv",pooled_reference(args.ablation.resolve(),recipe,bias,overall)),
    ):
        save_csv(args.out/file,rows)
    plots(args.out,events,bias)
    (args.out/"config.json").write_text(json.dumps(dict(
        scope="exploratory historical validation and test only, no model inference",
        dataset_sha256=RAW_SHA256,revision=REVISION,raw=str(args.raw.resolve()),bias=bias,
        SAR="dB of linear Sigma0; delta=post minus arithmetic mean of the two pre images in dB",
        ratio="post VH minus VV in dB",texture="native 3x3 and 5x5 complete valid windows; stats in dB²; W1 in log1p(dB²)",
        slope="native DEM centered differences, projected 10m corrected by cos(latitude)",
        W1="approximate 1D W1 of equally spaced binned representations, dB step .05, degrees .25, log-texture .025",
        event_weighting="all pixels for within-event histogram, equal per-event normalized mass for balanced split W1",
        correlation="one row per event at the locked bias, per split and combined; exploratory rho; no pixel p-values",
        oracle="event-label-sensitive diagnostic only; no bias or model change",
        limitations="per-event validation AP unavailable; full test multiclass confusion per event unavailable; native effective resolution and acquisition geometry may differ",
        sources=str(args.artifacts.resolve()),ablation=str(args.ablation.resolve()),
    ),indent=2)+"\n",encoding="utf-8")
    print(f"Saved exploratory analysis to {args.out}",flush=True)


if __name__=="__main__":
    main()
