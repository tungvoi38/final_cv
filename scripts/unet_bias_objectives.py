"""Label-free scalar-bias objectives and validated per-flip cache for A3 EoA."""
from pathlib import Path

import numpy as np
from scipy.special import expit, xlogy

SEED = 42
SAMPLE_CAP = 32_768
GRID = np.round(np.arange(-3., 3.001, .05), 2)
RULES = {
    "seed": SEED, "max_pixels_per_event": SAMPLE_CAP,
    "bias_grid": [float(GRID[0]), float(GRID[-1]), .05],
    "noncollapse": "1 <= predicted flood pixels, predicted flood fraction <= 0.5, and 0.5*p0 <= p <= 2*p0",
    "p0": "zero-bias argmax flood fraction on every input-valid pixel of the frozen mean logits",
    "fallback_bias": 0.,
    "tie_rule": "objective tolerance 1e-12, then nearest zero, then lower bias",
    "entropy": "mean categorical predictive entropy of frozen Flip TTA mean logits",
    "tta_consistency": "mean Jensen-Shannon divergence of four spatially aligned, per-view EoA predictions",
}


def pixel_indices(size, event):
    if not size:
        raise ValueError(f"No input-valid pixels in {event}")
    if size <= SAMPLE_CAP:
        return np.arange(size)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, int(event)]))
    return np.sort(rng.choice(size, SAMPLE_CAP, replace=False))


def flood_terms(logits):
    z = np.asarray(logits, dtype=np.float64)
    nonflood_norm = np.logaddexp(z[..., 0], z[..., 1])
    flood_odds = z[..., 2] - nonflood_norm
    return flood_odds, np.exp(z[..., :2] - nonflood_norm[..., None])


def shannon(probabilities):
    return -xlogy(probabilities, probabilities).sum(axis=-1)


def entropy_curve(logits):
    odds, nonflood = flood_terms(logits)
    nonflood_entropy = shannon(nonflood)
    result = []
    for b in GRID:
        p, q = expit(odds+b), expit(-odds-b)
        h = -xlogy(p,p) - xlogy(q,q) + q*nonflood_entropy
        result.append(h.mean())
    return np.array(result)


def consistency_curve(views):
    if views.ndim != 3 or views.shape[1:] != (4,3):
        raise ValueError("Expected aligned per-view logits [sample,4,3]")
    odds, nonflood = flood_terms(views)
    result=[]
    for b in GRID:
        p, q = expit(odds+b), expit(-odds-b)
        distributions = np.concatenate((q[...,None]*nonflood, p[...,None]), axis=2)
        js = shannon(distributions.mean(axis=1)) - shannon(distributions).mean(axis=1)
        result.append(np.maximum(js,0).mean())
    return np.array(result)


def choose_bias(objective, full_margin):
    if not len(full_margin) or not np.isfinite(full_margin).all() or not np.isfinite(objective).all():
        raise ValueError("Nonfinite or missing unlabeled objective statistics")
    n = len(full_margin)
    ordered = np.sort(full_margin.astype(np.float64))
    prevalence = (n - np.searchsorted(ordered, -GRID, side="right"))/n
    zero = int(np.flatnonzero(GRID == 0).item())
    p0 = float(prevalence[zero])
    lower, upper = max(1/n,.5*p0), min(.5,2*p0)
    feasible = (prevalence >= lower-1e-15) & (prevalence <= upper+1e-15) & (prevalence>0)
    choices = np.flatnonzero(feasible)
    if not len(choices):
        selected, status = zero, "fallback_no_noncollapsed_candidate"
    else:
        best = objective[choices].min()
        ties = choices[np.isclose(objective[choices],best,atol=1e-12,rtol=0)]
        selected = min(ties,key=lambda i:(abs(GRID[i]), GRID[i]))
        status = "ok"
    diagnostics = dict(status=status, zero_bias_prevalence=p0,
                       minimum_flood_prevalence=lower, maximum_flood_prevalence=upper,
                       feasible_biases=len(choices), objective_zero=float(objective[zero]),
                       objective_selected=float(objective[selected]),
                       at_grid_boundary=bool(selected in (0,len(GRID)-1)))
    curve = [dict(bias=float(b),objective=float(o),flood_prevalence=float(p),
                  feasible=bool(valid),selected=(i==selected))
             for i,(b,o,p,valid) in enumerate(zip(GRID,objective,prevalence,feasible,strict=True))]
    return dict(prior=[None,None,None],bias=float(GRID[selected]),
                predicted_prevalence_input_valid=float(prevalence[selected]),diagnostics=diagnostics),curve


def aligned_view_logits(models, inputs):
    import torch
    from evaluate_unet_eoa_flip_tta import FLIPS

    per_view = []
    for axes in FLIPS:
        view = torch.flip(inputs,axes) if axes else inputs
        a,b,c = [model(view).float() for model in models]
        average = (a+b+c)/3
        per_view.append(torch.flip(average,axes) if axes else average)
    return torch.stack(per_view,dim=1)  # B,4,3,H,W


def cache_views(out, cache, recipe, selection, expected):
    import torch
    from torch.utils.data import DataLoader
    from adapt_unet_event_bias import open_cache
    from evaluate_unet_eoa import (digest, load_calibrated_selection, load_model,
                                   pre_variability_inputs, seed_everything, write_json)
    from kuro_dataset import KuroDataset, RawArchive

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the frozen AMP validation forwards")
    if out.exists():
        raise FileExistsError(out)
    merged,_ = open_cache(cache,recipe,selection,expected)
    members = load_calibrated_selection(recipe,Path(recipe["selection"]))["sma"]
    cfg = selection["protocol"]["training"]
    ds = KuroDataset(RawArchive(cfg["raw"]),"validation")
    if set(ds.rows.actid.astype(str)) != set(merged):
        raise ValueError("Validation event IDs differ from the mean cache")
    loader = DataLoader(ds,batch_size=cfg["batch_size"],num_workers=cfg["workers"],shuffle=False,pin_memory=True)
    seed_everything(SEED)
    device=torch.device("cuda")
    models = [load_model(member,device) for member in members]
    indices = {event:pixel_indices(len(z),event) for event,z in merged.items()}
    offsets = dict.fromkeys(merged,0)
    collected={event:[] for event in merged}
    with torch.inference_mode():
        for step,batch in enumerate(loader,1):
            assignments=[]
            for i,raw_event in enumerate(batch["event_id"]):
                event=str(raw_event)
                valid = np.flatnonzero(batch["input_valid"][i].numpy().ravel())
                start,stop=offsets[event],offsets[event]+len(valid)
                if stop>len(merged[event]):
                    raise ValueError(f"Validation/raw counts exceed mean cache in {event}")
                left,right=np.searchsorted(indices[event],[start,stop])
                requested=indices[event][left:right]-start
                assignments.append((i,event,valid[requested]))
                offsets[event]=stop
            if any(len(requested) for _,_,requested in assignments):
                # Keep the original batch geometry for AMP convolution reproducibility.
                x={key:batch[key].to(device,non_blocking=True) for key in ("sar","dem")}
                with torch.autocast("cuda"):
                    logits=aligned_view_logits(models,pre_variability_inputs(x)[0])
                for i,event,requested in assignments:
                    if len(requested):
                        index=torch.as_tensor(requested,device=device,dtype=torch.long)
                        aligned=logits[i].reshape(4,3,-1).index_select(-1,index)
                        collected[event].append(aligned.permute(2,0,1).cpu().numpy())
            if step%100==0 or step==len(loader):
                print(f"validation individual views: {step}/{len(loader)} batches",flush=True)
    out.mkdir(parents=True,exist_ok=False)
    manifest={}
    for event, chunks in collected.items():
        views=np.concatenate(chunks)
        if offsets[event]!=len(merged[event]) or views.shape!=(len(indices[event]),4,3):
            raise ValueError(f"Missing validation views for {event}")
        result=views.mean(axis=1,dtype=np.float64)
        expected_mean=merged[event][indices[event]]
        if not np.isfinite(views).all() or not np.allclose(result,expected_mean,rtol=1e-6,atol=1e-5):
            raise ValueError(f"Aligned view average disagrees with frozen mean cache for {event}")
        path=out/f"{event}.npz"
        np.savez_compressed(path,indices=indices[event],logits=views)
        manifest[event]=dict(sha256=digest(path),n_pixels=len(views),
                             max_mean_logit_abs_error=float(np.abs(result-expected_mean).max()))
    write_json(out/"manifest.json",dict(split="validation",selection_sha256=recipe["selection_sha256"],
               mean_cache_manifest_sha256=digest(cache/"manifest.json"),domain="input_valid_only",
               flip_axes=recipe["flip_axes"],sampling_seed=SEED,sample_cap=SAMPLE_CAP,
               events=manifest))
    print(f"Saved aligned per-view validation cache: {out}",flush=True)


def load_views(directory, cache, recipe, merged):
    from evaluate_unet_eoa import digest, load_json

    manifest=load_json(directory/"manifest.json")
    if (manifest["split"], manifest["selection_sha256"], manifest["mean_cache_manifest_sha256"],
        manifest["domain"], manifest["flip_axes"], manifest["sampling_seed"], manifest["sample_cap"]) != (
            "validation",recipe["selection_sha256"],digest(cache/"manifest.json"),"input_valid_only",
            recipe["flip_axes"],SEED,SAMPLE_CAP):
        raise ValueError("Per-view cache provenance does not match the frozen validation mean cache")
    if set(manifest["events"])!=set(merged):
        raise ValueError("Per-view cache has different validation event IDs")
    data={}
    for event in merged:
        path=directory/f"{event}.npz"
        if digest(path)!=manifest["events"][event]["sha256"]:
            raise ValueError(f"Changed view artifact: {path}")
        idx=pixel_indices(len(merged[event]),event)
        with np.load(path) as artifact:
            logits,offsets=artifact["logits"],artifact["indices"]
        if (not np.array_equal(idx,offsets) or logits.shape!=(len(idx),4,3)
                or not np.isfinite(logits).all()
                or not np.allclose(logits.mean(axis=1,dtype=np.float64),merged[event][idx],rtol=1e-6,atol=1e-5)):
            raise ValueError(f"View cache misalignment: {path}")
        data[event]=logits
    return data


def plot_curves(out,curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for method in dict.fromkeys(r["method"] for r in curves):
        events=sorted({r["event_id"] for r in curves if r["method"]==method},key=int)
        fig,axes=plt.subplots(2,3,figsize=(12,6),sharex=True,layout="constrained")
        for ax,event in zip(axes.flat,events,strict=True):
            rows=[r for r in curves if r["method"]==method and r["event_id"]==event]
            x=[r["bias"] for r in rows]
            y=[r["objective"] for r in rows]
            ax.plot(x,y,color=".45",label="All biases")
            f=[r for r in rows if r["feasible"]]
            chosen,=[r for r in rows if r["selected"]]
            ax.scatter([r["bias"] for r in f],[r["objective"] for r in f],s=7,label="Feasible")
            ax.scatter(chosen["bias"],chosen["objective"],color="#ae4b26",s=27,label="Applied bias")
            ax.set_title(f"{event} · b={chosen['bias']:+.2f}")
        axes[0,0].legend(fontsize=7)
        fig.supxlabel("Flood bias")
        fig.supylabel("Entropy (nats)" if method=="entropy" else "Flip Jensen–Shannon divergence (nats)")
        fig.suptitle("Exploratory validation")
        fig.savefig(out/f"{method}_curves.png",dpi=165)
        plt.close(fig)
