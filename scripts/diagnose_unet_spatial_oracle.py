#!/usr/bin/env python3
"""Ground-truth spatial bias upper bounds on frozen EoA validation predictions."""
import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from adapt_unet_event_bias import (
    FLOOD, IGNORE, SEED, VAL_ARTIFACT, bias_for_groups, digest, flood_margin,
    load_json, oracle_from_ranks, reference, scores, write_json,
)
from adapt_unet_expected_iou import CHUNK, sar_features
from adapt_unet_iw_iou import cache_file, open_predictions, supplement
from audit_unet_bias_failures import checked, read_csv, save_csv
from evaluate_unet_eoa_diagnostics import pr_curve, rank_event, rank_pooled
from kuro_dataset import KuroDataset, RAW_SHA256, preprocess, read_sample
from unet_conservative_expected_iou import get_sar

GLOBAL, EVENT, PATCH = "global_bias", "event_oracle", "patch_oracle"
KS = (2, 3, 4)
REGIMES = {k:f"sar_regime_k{k}_oracle" for k in KS}
METHODS = (GLOBAL, EVENT, PATCH, *REGIMES.values())
FEATURES = ("post_vv_db", "delta_vv_db", "delta_vh_db", "post_vh_minus_vv_db", "slope_degrees")
FIT_CAP = 100_000
N_INIT = 10


def patch_ranges(raw, predictions, sar, path, cache_hash, sar_hash):
    counts = {event:len(values) for event, values in predictions.items()}
    identity = dict(version=1, split="validation", raw_sha256=RAW_SHA256,
                    mean_cache_sha256=cache_hash, sar_manifest_sha256=sar_hash, event_counts=counts)
    if not path.exists():
        dataset = KuroDataset(raw, "validation")
        if set(dataset.rows.actid.astype(str)) != set(predictions):
            raise ValueError("Raw validation events differ from frozen logits")
        rows = {e:[] for e in predictions}
        offsets = dict.fromkeys(counts, 0)
        for i, (_, row) in enumerate(dataset.rows.iterrows()):
            dataset.archive.check_unchanged()
            sample = preprocess(*read_sample(dataset.archive.path, row, include_labels=False), include_labels=False)
            event = str(row.actid)
            features = sar_features(sample["sar_db"], sample["input_valid"])
            start = offsets[event]
            stop = start + len(features)
            if stop > counts[event] or not np.array_equal(features, sar[event][start:stop]):
                raise ValueError(f"SAR/logit patch alignment mismatch at row {i}: {event}")
            rows[event].append(dict(patch_id=str(row["tortilla:id"]), start=start, stop=stop))
            offsets[event] = stop
            if (i+1) % 250 == 0 or i+1 == len(dataset):
                print(f"Aligned patch ranges: {i+1}/{len(dataset)}", flush=True)
        if offsets != counts:
            raise ValueError("Patch ranges did not cover the frozen logit cache")
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, identity | {"patches":rows})
    value = load_json(path)
    if any(value[k] != v for k,v in identity.items()):
        raise ValueError("Patch manifest belongs to a different validation cache")
    ids = set()
    for event, rows in value["patches"].items():
        offset = 0
        for row in rows:
            if row["patch_id"] in ids or not 0 <= offset == row["start"] <= row["stop"] <= counts[event]:
                raise ValueError(f"Invalid patch manifest for {event}")
            ids.add(row["patch_id"])
            offset = row["stop"]
        if offset != counts[event]:
            raise ValueError(f"Incomplete patch manifest for {event}")
    if set(value["patches"]) != set(counts):
        raise ValueError("Missing validation event in patch manifest")
    return value["patches"]


def feature_rows(sar, extra, indices):
    vv, dvv, dvh = sar[indices].T
    vh, slope = extra[indices].T
    return np.column_stack((vv,dvv,dvh,vh-vv,slope))


def fill_slope(features, median):
    features[~np.isfinite(features[:, -1]), -1] = median
    return features


def fit_regimes(sar, extra):
    parts, counts = [], {}
    for event, data in sar.items():
        rng = np.random.default_rng(np.random.SeedSequence([SEED,int(event)]))
        sample = rng.choice(len(data), size=min(FIT_CAP,len(data)), replace=False)
        parts.append(feature_rows(data,extra[event],sample))
        counts[event] = len(sample)
    data = np.concatenate(parts)
    seen = data[np.isfinite(data[:, -1]), -1]
    if not len(seen):
        raise ValueError("No valid slope in the unlabeled fit sample")
    median = float(np.median(seen))
    scaler = StandardScaler().fit(fill_slope(data,median))
    x = scaler.transform(data)
    models = {}
    for k in KS:
        model = KMeans(n_clusters=k, n_init=N_INIT, random_state=SEED).fit(x)
        if len(np.unique(model.labels_)) != k:
            raise ValueError(f"K-means returned fewer than {k} regimes")
        models[k] = model
        print(f"Fit validation SAR regimes K={k} on {len(x)} unlabeled pixels", flush=True)
    return scaler, median, models, dict(
        features=list(FEATURES), fit_pixels_by_event=counts, slope_imputation_median=median,
        feature_mean=scaler.mean_.tolist(), feature_scale=scaler.scale_.tolist(),
        clusters={str(k):dict(centroids_standardized=m.cluster_centers_.tolist(),
                              centroids_physical=scaler.inverse_transform(m.cluster_centers_).tolist(),
                              fit_size=np.bincount(m.labels_,minlength=k).tolist(), inertia=float(m.inertia_))
                  for k,m in models.items()})


def assign_regimes(sar, extra, scaler, median, models):
    result = {k:np.empty(len(sar),dtype=np.int8) for k in models}
    for i in range(0,len(sar),CHUNK):
        sl = slice(i,i+CHUNK)
        x = scaler.transform(fill_slope(feature_rows(sar,extra,sl),median))
        for k, model in models.items():
            result[k][sl] = model.predict(x)
    return result


@dataclass
class Group:
    identifier: str
    selector: object
    input_pixels: int
    rank: tuple | None
    tp: np.ndarray
    fp: np.ndarray


def group(identifier, selector, raw_margin, flood, input_pixels):
    values, truth = raw_margin[selector], flood[selector]
    if len(values):
        rank = rank_event([(values,truth)])
        positive, total = rank[1:]
        # A group without any true flood can always choose zero alarms. Keep its rank for AP.
        if positive.sum():
            tp = np.r_[0,positive[::-1].cumsum()]
            fp = np.r_[0,(total-positive)[::-1].cumsum()]
        else:
            tp = fp = np.zeros(1,dtype=np.int64)
    else:
        rank = None
        tp = fp = np.zeros(1,dtype=np.int64)
    return Group(identifier, selector, input_pixels, rank, tp, fp)


def optimize_event(groups, positives):
    """Dinkelbach over exact integer counts with the all-negative option in each group."""
    num, den = 0, max(positives, 1)
    for iteration in range(1, 101):
        # Index order breaks objective ties toward fewer predictions (and no false alarms when no flood).
        choices = [int(np.argmax(den*g.tp - num*g.fp)) for g in groups]
        tp = sum(int(g.tp[i]) for g,i in zip(groups,choices,strict=True))
        fp = sum(int(g.fp[i]) for g,i in zip(groups,choices,strict=True))
        divisor = max(positives+fp, 1)
        if den*tp == num*divisor:
            return choices, tp, fp, iteration
        if den*tp < num*divisor:
            raise ValueError("Spatial oracle objective unexpectedly decreased")
        num, den = tp, divisor
    raise RuntimeError("Spatial oracle failed to reach an exact fixed point")


def selected_bias(g, i):
    return bias_for_groups(g.rank[0], i) if g.rank is not None else None


def predicted_confusion(raw_margin, nonflood, labels, bias):
    pred = np.where(raw_margin.astype(np.float64)>-bias,FLOOD,nonflood)
    return np.bincount(3*labels.astype(np.int64)+pred,minlength=9).reshape(3,3)


def shifted_rank(rank, bias):
    return rank[0].astype(np.float64)+bias, rank[1], rank[2]


def average_precision(rank):
    return float(pr_curve(rank)[-1]) if rank[1].sum() else None


def apply_groups(event, method, groups, margin, nonflood, y):
    positives = int(np.count_nonzero(y==FLOOD))
    choices, tp, fp, iterations = optimize_event(groups, positives)
    cm = np.zeros((3,3),dtype=np.int64)
    biased_ranks, details = [], []
    for g,i in zip(groups,choices,strict=True):
        n = 0 if g.rank is None else int(g.rank[2].sum())
        target_pos = 0 if g.rank is None else int(g.rank[1].sum())
        bias = selected_bias(g,i)
        group_tp, group_fp = int(g.tp[i]),int(g.fp[i])
        if bias is not None:
            sub = predicted_confusion(margin[g.selector],nonflood[g.selector],y[g.selector],bias)
            np.testing.assert_array_equal([sub[FLOOD,FLOOD],sub[:FLOOD,FLOOD].sum()],
                                          [group_tp,group_fp],err_msg=f"{event}/{g.identifier} bias")
            cm += sub
            biased_ranks.append(shifted_rank(g.rank,bias))
        details.append(dict(event_id=event, method=method, group_id=g.identifier, bias=bias,
             input_valid_pixels=g.input_pixels, scored_pixels=n, ignored_pixels=g.input_pixels-n,
             true_flood_pixels=target_pos, no_flood=target_pos==0 if n else None,
             status="no_scored_pixels" if n==0 else "no_flood" if target_pos==0 else "scored",
             tp=group_tp,fp=group_fp,fn=target_pos-group_tp,iterations=iterations))
    np.testing.assert_array_equal([cm.sum(),cm[FLOOD,FLOOD],cm[:FLOOD,FLOOD].sum()],
                                  [len(y),tp,fp],err_msg=f"{event}/{method} total")
    return cm, rank_pooled(biased_ranks), details


def event_metrics(event, logits, labels, patch_rows, regimes, historical, global_bias, global_cm):
    keep = labels != IGNORE
    y = labels[keep]
    if not len(y):
        raise ValueError(f"No scored pixels for event {event}")
    if not np.isin(y,[0,1,2]).all():
        raise ValueError(f"Unexpected label in {event}")
    z = logits[keep]
    margin = flood_margin(z)
    nonflood = z[:,:FLOOD].argmax(1)
    del z
    flood = y==FLOOD
    if not np.isfinite(margin).all():
        raise ValueError(f"Nonfinite flood margin in {event}")

    raw_rank = rank_event([(margin,flood)])
    event_bias, event_iou = oracle_from_ranks(raw_rank) if flood.any() else (0., 0.)
    checked([event_bias,event_iou], [float(historical["exact_oracle_bias"]),float(historical["exact_oracle_iou"])],
            f"{event} archived Event Oracle")
    by_method, ranks, details, inventory = {}, {}, [], {}
    for method,bias in ((GLOBAL,global_bias),(EVENT,event_bias)):
        by_method[method] = predicted_confusion(margin,nonflood,y,bias)
        ranks[method] = rank_pooled([shifted_rank(raw_rank,bias)])
        inventory[method] = dict(groups=1, groups_no_flood=int(not flood.any()), groups_no_scored=0)
    np.testing.assert_array_equal(by_method[GLOBAL],global_cm,err_msg=f"{event} archived Global Bias")
    checked(scores(by_method[EVENT])["flood_iou"],event_iou,f"{event} Event Oracle confusion")

    offset = 0
    patches = []
    for item in patch_rows:
        start,stop = item["start"],item["stop"]
        count = int(keep[start:stop].sum())
        patches.append(group(item["patch_id"],slice(offset,offset+count),margin,flood,stop-start))
        offset += count
    if offset != len(y):
        raise ValueError(f"Scored patch groups do not cover {event}")
    by_method[PATCH],ranks[PATCH],rows = apply_groups(event,PATCH,patches,margin,nonflood,y)
    del patches
    details.extend(rows)
    for k,method in REGIMES.items():
        all_ids = regimes[k]
        scored_ids = all_ids[keep]
        counts = np.bincount(all_ids,minlength=k)
        groups = [group(str(i),scored_ids==i,margin,flood,int(counts[i])) for i in range(k)]
        by_method[method],ranks[method],rows = apply_groups(event,method,groups,margin,nonflood,y)
        details.extend(rows)
        del groups
    result = []
    raw_ap = average_precision(raw_rank)
    for method in METHODS:
        cm = by_method[method]
        metrics = scores(cm)
        delta = metrics["flood_iou"]-event_iou
        if method != GLOBAL and delta < -1e-12:
            raise ValueError(f"{event}/{method} is below Event Oracle despite nested bias feasibility")
        groups = [r for r in details if r["method"]==method]
        counts = inventory.get(method) or dict(
            groups=len(groups), groups_no_flood=sum(r["status"]=="no_flood" for r in groups),
            groups_no_scored=sum(r["status"]=="no_scored_pixels" for r in groups))
        result.append(dict(event_id=event,method=method,scored_pixels=len(y),
            true_flood_pixels=int(flood.sum()), **counts, **metrics, flood_ap_adjusted=average_precision(ranks[method]),
            flood_ap_raw=raw_ap,event_oracle_iou=event_iou,flood_iou_delta_vs_event_oracle_pp=100*delta,
            tp=int(cm[2,2]),fp=int(cm[:2,2].sum()),fn=int(cm[2,:2].sum()),
            scalar_bias=global_bias if method==GLOBAL else event_bias if method==EVENT else None))
    for row in details:
        row["scored_fraction_of_event"] = row["scored_pixels"]/len(y)
    return by_method, ranks, raw_rank, result, details


def plot_summary(out, events, event_rows):
    labels = ["Global bias","Event oracle","Patch oracle",*[f"SAR regime K={k}" for k in KS]]
    lookup = {(r["event_id"],r["method"]):r for r in event_rows}
    values = np.array([[100*lookup[event,m]["flood_iou"] for event in events] for m in METHODS])
    fig, axs = plt.subplots(1,2,figsize=(13,4.5),layout="constrained")
    for j,label in enumerate(labels):
        axs[0].plot(events,values[j],"o-",label=label,lw=1.1)
    axs[0].set(xlabel="Validation event",ylabel="Flood IoU (%)")
    axs[0].legend(fontsize=8)
    delta = values[2:] - values[1]
    im = axs[1].imshow(delta, aspect="auto", cmap="Blues", vmin=0, vmax=max(1,delta.max()))
    for (i,j),v in np.ndenumerate(delta):
        axs[1].text(j,i,f"{v:+.1f}",ha="center",va="center",fontsize=8,
                    color="white" if v>.65*delta.max() else "black")
    axs[1].set(xticks=np.arange(len(events)),xticklabels=events,yticks=np.arange(len(delta)),
               yticklabels=labels[2:],xlabel="Validation event")
    fig.colorbar(im, ax=axs[1], label="Gain vs Event Oracle (pp)")
    fig.suptitle("Frozen EoA · Spatial bias oracles (ground truth diagnostic)")
    fig.savefig(out/"event_iou.png",dpi=170)
    plt.close(fig)


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    for cache in (args.sar_cache,args.extra_cache):
        if not (cache/"manifest.json").is_file():
            raise FileNotFoundError(f"Expected already saved SAR cache: {cache}")
    recipe, selection, grid, expected = reference()
    events = sorted(expected,key=int)
    manifest, x = open_predictions(args.cache,recipe,selection,events)
    raw = selection["protocol"]["training"]["raw"]
    sar = get_sar(raw,x,args.cache,args.sar_cache)
    extra = supplement(raw,x,sar,args.cache,args.extra_cache)
    args.out.mkdir(parents=True,exist_ok=False)
    patch_manifest = args.patch_manifest or args.out/"patch_manifest.json"
    cache_hash = digest(args.cache/"manifest.json")
    patches = patch_ranges(raw,x,sar,patch_manifest,cache_hash,digest(args.sar_cache/"manifest.json"))
    scaler, median, models, cluster_config = fit_regimes(sar,extra)
    write_json(args.out/"sar_regimes.json",cluster_config)
    old = read_csv(args.oracle_reference)
    historical = {row["event_id"]:row for row in old}
    if len(old)!=len(events) or set(historical)!=set(events):
        raise ValueError("Historical Event Oracle does not cover exactly the validation events")
    bias = recipe["flood_logit_bias"]
    ix, = np.flatnonzero(np.isclose(grid,bias,atol=1e-12,rtol=0))
    pooled = {m:np.zeros((3,3),dtype=np.int64) for m in METHODS}
    ranks = {m:[] for m in METHODS}
    raw_ranks, event_rows, group_rows = [], [], []
    for event in events:
        regimes = assign_regimes(sar[event],extra[event],scaler,median,models)
        labels = cache_file(args.cache,manifest,event,labels=True)
        matrices, adjusted, raw_rank, results, groups = event_metrics(
            event,x[event],labels,patches[event],regimes,historical[event],bias,expected[event][ix])
        for method in METHODS:
            pooled[method] += matrices[method]
            ranks[method].append(adjusted[method])
        raw_ranks.append(raw_rank)
        event_rows.extend(results)
        group_rows.extend(groups)
        print(f"Scored {event}: " + ", ".join(f"{r['method']}={r['flood_iou']:.4f}" for r in results),flush=True)
    raw_ap = average_precision(rank_pooled(raw_ranks))
    oracle_miou = scores(pooled[EVENT])["pooled_miou"]
    oracle_f1 = scores(pooled[EVENT])["flood_f1"]
    summary = []
    for method in METHODS:
        sub = [r for r in event_rows if r["method"]==method]
        ious = [r["flood_iou"] for r in sub]
        metrics = scores(pooled[method])
        ap = average_precision(rank_pooled(ranks.pop(method)))
        summary.append(dict(method=method,events=len(events),pooled_miou=metrics["pooled_miou"],
            pooled_flood_iou=metrics["flood_iou"],pooled_flood_f1=metrics["flood_f1"],
            event_macro_flood_iou=float(np.mean(ious)),event_iou_std=float(np.std(ious,ddof=0)),
            pooled_flood_ap_adjusted=ap,pooled_flood_ap_raw=raw_ap,
            event_macro_iou_delta_vs_event_oracle_pp=float(np.mean([r["flood_iou_delta_vs_event_oracle_pp"] for r in sub])),
            pooled_miou_delta_vs_event_oracle_pp=100*(metrics["pooled_miou"]-oracle_miou),
            pooled_f1_delta_vs_event_oracle_pp=100*(metrics["flood_f1"]-oracle_f1),
            groups=sum(r["groups"] for r in sub),groups_no_flood=sum(r["groups_no_flood"] for r in sub),
            groups_no_scored=sum(r["groups_no_scored"] for r in sub)))
    for name, rows in (("summary",summary),("event_metrics",event_rows),("group_biases",group_rows)):
        save_csv(args.out/f"{name}.csv",rows)
    plot_summary(args.out,events,event_rows)
    write_json(args.out/"evaluation_config.json",dict(
        split="validation",scope="ground-truth diagnostic only; spatial oracle biases are not deployable",
        checkpoint_selection_sha256=recipe["selection_sha256"],mean_cache_sha256=cache_hash,
        sar_cache_sha256=digest(args.sar_cache/"manifest.json"),extra_cache_sha256=digest(args.extra_cache/"manifest.json"),
        patch_manifest=str(patch_manifest.resolve()),patch_manifest_sha256=digest(patch_manifest),
        oracle_reference_sha256=digest(args.oracle_reference),
        validation_confusion_sha256=digest(VAL_ARTIFACT/"validation_confusion_eoa_flip_tta.npz"),
        global_bias=bias,seed=SEED,kmeans_k=list(KS),kmeans_n_init=N_INIT,kmeans_fit_cap_per_event=FIT_CAP,
        regime_fit="shared scaler and centers across unlabeled input-valid validation events; fixed K values",
        optimizer="exact candidate score group thresholds; jointly maximize event TP/(total true flood + FP)",
        oracle_ties="among equal parametric group objectives, prefer fewer predictions; within the threshold interval "
                    "pick the attainable bias nearest zero; historical Event Oracle tie rule is retained",
        no_flood_group="zero flood alarms; no-scored groups have null bias and no contribution; all-negative event IoU zero",
        threshold_rule="float64 strict margin > -bias, ties nonflood",
        average_precision="step-integral AP using grouped score ties; raw margin or margin plus local bias in float64; "
                          "oracle biases use evaluation labels and optimize IoU, not AP",
        population_iou_std_ddof=0,code_sha256=digest(Path(__file__))))
    print(f"Saved spatial oracle validation diagnostics: {args.out}",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache","sar-cache","extra-cache","oracle-reference","out"):
        parser.add_argument(f"--{name}",required=True,type=Path)
    parser.add_argument("--patch-manifest",type=Path,help="Reuse or write the cached patch-index manifest")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
