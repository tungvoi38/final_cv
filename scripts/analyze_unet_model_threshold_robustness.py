#!/usr/bin/env python3
"""Threshold robustness of frozen A3 model variants on aligned evaluation patches."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from analyze_unet_threshold_sensitivity import (
    LOCKED, cumulative_confusions, digest, load_json, plt, save_csv, save_figure, score_groups, scores, write_json,
)
from ccla_crossfit_cache import cache_path, collate, flipped_single, lookup, values
from ccla_crossfit_protocol import select_rows, split_rows
from ccla_eoa_cache import ensemble_mean, models
from evaluate_unet_best_methods_test import EVENT, NOBIAS, REF, labels_for_event
from evaluate_unet_eoa import pre_variability_inputs, seed_everything
from evaluate_unet_eoa_diagnostics import pr_curve, rank_pooled
from kuro_dataset import RawArchive, normalize_physical_inputs, preprocess, read_sample
from train_unet import IGNORE
from adapt_unet_event_bias import flood_margin, oracle_from_ranks

FLIP = "eoa_flip_tta"
IDENTITY = [*(f"vanilla_erm_{s}" for s in (42,43,44)),*(f"single_sma_{s}" for s in (42,43,44)),"eoa"]
MARGIN_DTYPE = np.dtype([("margin","<f4"),("other","u1"),("label","u1")])
GRID = np.round(np.arange(-400,401)*.05,2)


class LabeledPatches(Dataset):
    def __init__(self, archive, rows):
        self.archive, self.rows = archive, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        self.archive.check_unchanged()
        row = self.rows.iloc[index]
        p = preprocess(*read_sample(self.archive.path,row,include_labels=True),include_labels=True)
        valid = p["input_valid"]
        sar, dem = normalize_physical_inputs(p["sar_db"],p["dem_m"],valid)
        y = p["mask"]
        return torch.from_numpy(sar),torch.from_numpy(dem),str(row.actid),y


def evaluation_loader(archive, rows):
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise ValueError("Missing inference requires exactly one visible CUDA GPU")
    seed_everything(42)
    return DataLoader(LabeledPatches(archive,rows),batch_size=2,num_workers=2,shuffle=False,
                      pin_memory=True,collate_fn=collate)


def evaluation_inputs(batch):
    return pre_variability_inputs({k:t.cuda(non_blocking=True) for k,t in batch.items()})[0]


def score_array(z, labels):
    if not len(labels) or not np.isfinite(z).all() or not np.isin(labels,[0,1,2]).all():
        raise ValueError("Invalid scored logits or labels")
    return flood_margin(z), z[:,:2].argmax(1).astype(np.uint8), labels


@torch.inference_mode()
def identity_cache(archive, rows, online, sma, path):
    # One data pass and one identity forward per checkpoint. EoA reuses the three SMA outputs.
    loader = evaluation_loader(archive,rows)
    nets = models(online,[42,43,44])+models(sma,[42,43,44])
    for step,(batch,items) in enumerate(loader,1):
        with torch.autocast("cuda"):
            x = evaluation_inputs(batch)
            logits = [m(x).float().permute(0,2,3,1).cpu().numpy() for m in nets]
        for i,(event, y) in enumerate(items):
            y = y.astype(np.uint8,copy=False)
            keep = y!=IGNORE
            if not keep.any():
                continue
            y = y[keep]
            out = [z[i][keep] for z in logits]
            out.append(ensemble_mean(*out[3:]))
            data = np.empty(len(y),dtype=MARGIN_DTYPE)
            for name,z in zip(IDENTITY,out,strict=True):
                data["margin"],data["other"],data["label"] = score_array(z,y)
                with (path/f"{name}-{event}.bin").open("ab") as f:
                    data.tofile(f)
        if step%100==0 or step==len(loader):
            print(f"Identity model forwards: {step}/{len(loader)}",flush=True)


def frozen_setup(a):
    cfg = load_json(a.threshold_results/"analysis_config.json")
    source, test = Path(cfg["oof_run"]),Path(cfg["test_run"])
    plan, tc = load_json(source/"frozen_config.json"),load_json(test/"frozen_config.json")
    selection = load_json(a.selection)
    online, sma = selection["online"],tc["members"]
    if ([m["sha256"] for m in sma]!=[m["sha256"] for m in plan["full_train"]]
            or any(m["epoch"]!=20 or m["weights_key"]!="model" for m in sma)
            or any(m["weights_key"]!="online_model" or m["input_channels"]!=11 for m in online)
            or [m["run_id"] for m in online]!=[m["run_id"] for m in selection["sma"]]
            or any(Path(o["checkpoint"]).parent != Path(s["checkpoint"]).parent.parent
                   for o,s in zip(online,sma,strict=True))):
        raise ValueError("Unexpected frozen online and fixed-epoch SMA provenance")
    return cfg,source,test,plan,tc,online,sma


def flip_inputs(cfg, split):
    p, = cfg["splits"][split]["cache_manifests"].keys()
    path = Path(p)
    if digest(path)!=cfg["splits"][split]["cache_manifests"][p]:
        raise ValueError(f"Archived cache manifest changed: {path}")
    manifest = load_json(path)
    return lookup([(path.parent,manifest)])


def analyze_event(margin, other, y):
    rank = score_groups(margin,y==2)
    oracle, iou = oracle_from_ranks(rank) if np.any(y==2) else (0.,0.)
    lo = -float(rank[0][-1])
    hi = np.nextafter(-np.float64(rank[0][0]),np.inf)
    grid = np.unique(np.r_[GRID,lo,hi,0.,LOCKED[REF],oracle])
    cms = cumulative_confusions(margin,other,y,grid)
    ap = pr_curve(rank)[3] if np.any(y==2) else None
    return rank,ap,oracle,iou,grid,cms


def point(grid,cms,b):
    return cms[np.searchsorted(grid,b)]


def summarize_cms(cms):
    event_iou = np.array([scores(c)["flood_iou"] for c in cms])
    return scores(sum(cms)) | dict(event_macro_flood_iou=float(event_iou.mean()),
                                  event_flood_iou_std=float(event_iou.std()),
                                  worst_event_iou=float(event_iou.min()))


def render(out, events, curves):
    fig,axes = plt.subplots(2,4,figsize=(13,6),layout="constrained",sharex=True,sharey=True)
    for ax,(model, specs) in zip(axes.flat,curves.items(),strict=True):
        for e,(grid,cm) in specs.items():
            tp = cm[:,2,2]
            iou = tp/np.maximum(cm[:,2,:].sum(1)+cm[:,:,2].sum(1)-tp,1)
            highlight = e in ("411","445")
            ax.plot(grid,100*iou,lw=1.1 if highlight else .65,alpha=.9 if highlight else .45,
                    color={"411":"#d55e00","445":"#009e73"}.get(e,"#0072b2"),label=e if highlight else None)
        ax.axvline(0,lw=.8,color="#555555",ls="--")
        ax.set(title=model.replace("_"," "),xlim=(-8,10),ylim=(0,100))
        ax.grid(alpha=.15)
    if "411" in events:
        axes[0,0].legend(frameon=False,fontsize=8)
    fig.supxlabel("Flood-logit bias")
    fig.supylabel("Flood IoU (%)")
    save_figure(fig,out,"model_sensitivity")


def analyze_split(a, cfg, test, tc, archive, online, sma, split):
    index, patches = flip_inputs(cfg,split)
    events = sorted(index,key=int)
    ids = [p["patch_id"] for e in events for p in patches[e]]
    rows = select_rows(split_rows(archive.frame,"test" if split=="historical_test" else split),ids)
    out = a.out/split
    out.mkdir(exist_ok=False)
    conf_path = a.threshold_results/split/"event_threshold_confusions.npz"
    if digest(conf_path)!=cfg["splits"][split]["archive_sha256"]:
        raise ValueError(f"Changed threshold artifacts: {split}")
    with np.load(conf_path,allow_pickle=False) as file:
        ci = {e:i for i,e in enumerate(file["event_ids"].tolist())}
        archived = file["confusions"][:,np.searchsorted(file["grid"],0.)]
    curves, summaries, event_rows, curve_rows, distribution = {}, [], [], [], []
    with TemporaryDirectory(dir=a.scratch,prefix=f"threshold-{split}-") as temp:
        directory = Path(temp)
        identity_cache(archive,rows,online,sma,directory)
        for name in [*IDENTITY,FLIP]:
            model_ranks, cs, bs = [], {}, []
            curves[name] = {}
            allowed = {NOBIAS:0.}
            if name==FLIP:
                allowed[REF] = LOCKED[REF]
            points = {k:[] for k in (*allowed,EVENT)}
            for e in events:
                if name!=FLIP:
                    buf = np.memmap(directory/f"{name}-{e}.bin",mode="r",dtype=MARGIN_DTYPE)
                    margin,other,y = (buf[k] for k in ("margin","other","label"))
                else:
                    z = values(index,e,"logits")
                    if split=="historical_test":
                        y0 = labels_for_event(archive,select_rows(rows,[p["patch_id"] for p in patches[e]]),
                            patches[e],cache_path(index[e][0],e,"masks"),len(z))
                    else:
                        y0 = values(index,e,"labels")
                    keep = y0!=IGNORE
                    margin,other,y = score_array(z[keep],y0[keep])
                result,ap,b,iou,grid,mat = analyze_event(margin,other,y)
                del margin,other,y
                z0 = point(grid,mat,0.)
                np.testing.assert_array_equal(z0.sum(1),archived[ci[e]].sum(1),err_msg=f"{name}/{split}/{e} support")
                if name==FLIP:
                    np.testing.assert_array_equal(z0,archived[ci[e]],err_msg=f"{split}/{e} Flip cache")
                curves[name][e] = grid,mat
                model_ranks.append(result)
                bs.append(b)
                cs[e] = scores(z0)
                for label,bias in (allowed | {EVENT:b}).items():
                    cm = point(grid,mat,bias)
                    points[label].append(cm)
                    event_rows.append(dict(model=name,event_id=e,operating_point=label,bias=bias,
                        diagnostic_only=label==EVENT,flood_ap=ap,**scores(cm)))
                for bias,cm in zip(grid,mat,strict=True):
                    curve_rows.append(dict(model=name,event_id=e,bias=bias,event_oracle=bias==b,**scores(cm)))
            ap_pooled = pr_curve(rank_pooled(model_ranks))[3]
            for k,cms in points.items():
                summaries.append(dict(model=name,operating_point=k,bias=None if k==EVENT else allowed[k],
                    diagnostic_only=k==EVENT,pooled_flood_ap=ap_pooled,**summarize_cms(cms)))
            q1,med,q3 = np.quantile(bs,[.25,.5,.75])
            gains = [scores(c)["flood_iou"]-cs[e]["flood_iou"] for e,c in zip(events,points[EVENT],strict=True)]
            distribution.append(dict(model=name,events=len(events),bias_mean=np.mean(bs),bias_median=med,
                bias_std=np.std(bs),bias_iqr=q3-q1,bias_min=min(bs),bias_max=max(bs),
                oracle_mean_gain_pp=100*np.mean(gains),events_gain_gt_1pp=sum(x>.01 for x in gains)))
            print(f"{split}: scored {name}",flush=True)
    for name,data in (("summary",summaries),("event_metrics",event_rows),("event_curves",curve_rows),
                      ("oracle_distribution",distribution)):
        save_csv(out/f"{name}.csv",data)
    render(out,events,curves)
    return distribution


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threshold-results",type=Path,required=True)
    p.add_argument("--selection",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--scratch",type=Path,default=None)
    p.add_argument("--splits",nargs="+",choices=("validation","historical_test"),
                   default=["validation","historical_test"])
    a = p.parse_args()
    if len(set(a.splits))!=len(a.splits):
        p.error("Each split can appear once")
    cfg,test_src,test,plan,tc,online,sma = frozen_setup(a)
    a.out.mkdir(parents=True,exist_ok=False)
    if a.scratch is None:
        a.scratch = a.out
    a.scratch.mkdir(parents=True,exist_ok=True)
    write_json(a.out/"analysis_config.json",dict(scope=cfg["scope"],splits=a.splits,online=online,sma=sma,
        validation_and_test_scope="same 959 and 2000 patches as the archived threshold analysis",
        threshold_config_sha256=digest(a.threshold_results/"analysis_config.json"),
        selection_sha256=digest(a.selection),script_sha256=digest(Path(__file__)),
        balanced_bias={"model":FLIP,"value":LOCKED[REF],"source":"same ensemble recipe on OOF events"},
        missing_balanced_bias=IDENTITY))
    archive = RawArchive(plan["raw"])
    lines = ["# Experiment 1 — Model threshold robustness", "",
        "Vanilla ERM uses per-seed validation-selected raw online checkpoints. Single SMA, EoA and EoA + Flip TTA "
        "use fixed epoch 20 for each seed, matching the frozen cross-fit study. Current models share the same patch "
        "scope and class masks within each split. Online/SMA differences may include checkpoint epoch. "
        "SMA-only and vanilla model sweeps require supplemental frozen inference. EoA + Flip TTA reuses its original cache.", "",
        "AP pools continuous flood margins with exact ties grouped and is unchanged by a single scalar threshold. "
        "Per-event oracles maximize Flood IoU and use the split's labels only for retrospective diagnostics. "
        "OOF event-balanced +3.70 is compatible with EoA + Flip TTA. Identity variants lack comparable event-disjoint "
        "source predictions, so their balanced points are unavailable. Historical test rows are exploratory.", ""]
    for split in a.splits:
        ds = analyze_split(a,cfg,test,tc,archive,online,sma,split)
        lines += [f"## {split}", "", "| Model | Oracle bias range | IoU gain (pp) | Events with >1 pp gain |",
                  "|---|---:|---:|---:|"]
        for row in ds:
            lines.append(f"| {row['model']} | {row['bias_min']:+.2f} to {row['bias_max']:+.2f} | "
                         f"{row['oracle_mean_gain_pp']:.2f} | {row['events_gain_gt_1pp']}/{row['events']} |")
        lines += ["",f"[Full metrics]({split}/summary.csv) · [Per-event curves]({split}/event_curves.csv)",""]
    (a.out/"REPORT.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__":
    main()
