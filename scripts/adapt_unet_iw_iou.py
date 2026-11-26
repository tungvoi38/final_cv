#!/usr/bin/env python3
"""Exploratory importance-weighted Flood IoU on frozen validation predictions."""
import argparse
from pathlib import Path

import numpy as np
from rasterio.warp import transform
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from adapt_unet_event_bias import (
    FLOOD, FLIP_TTA_BASELINE, IGNORE, VAL_ARTIFACT, digest, file_paths, flood_margin,
    load_json, oracle_from_ranks, reference, write_json,
)
from adapt_unet_expected_iou import CHUNK, SEED
from audit_unet_bias_failures import checked, read_csv, save_csv
from diagnose_sar_scattering import slope_degrees
from evaluate_unet_eoa_diagnostics import rank_event
from kuro_dataset import KuroDataset, RAW_SHA256, preprocess, read_sample
from unet_conservative_expected_iou import (
    ORACLE, REF, best_index, get_sar, reference_from_sources,
)

POOL = "source_pooled_iou"
MARGIN = "iw_iou_margin"
SAR = "iw_iou_margin_sar"
METHODS = (POOL, REF, MARGIN, SAR, ORACLE)
FEATURES = ("margin", "post_vv_db", "post_vh_db", "delta_vv_db", "delta_vh_db",
            "post_vh_minus_vv_db", "slope_degrees")
FIT_CAP = 100_000
RATIO_CLIP = (0.05, 20.0)
OVERLAP = (0.1, 0.9)
C = 1.0
SUPPLEMENT = "iw-post-vh-native-slope-v1"


def cache_file(cache, manifest, event, *, labels=False):
    n = manifest["event_counts"][event]
    path = file_paths(cache, event)[int(labels)]
    width = n if labels else 12 * n
    if not n or path.stat().st_size != width or digest(path) != manifest["file_sha256"][path.name]:
        raise ValueError(f"Incomplete or changed cache file: {path}")
    return np.memmap(path, mode="r", dtype="u1" if labels else "<f4",
                     shape=(n,) if labels else (n, 3))


def open_predictions(cache, recipe, selection, events):
    manifest = load_json(cache / "manifest.json")
    expected = dict(split="validation", selection_sha256=recipe["selection_sha256"],
                    recipe_sha256=digest(FLIP_TTA_BASELINE), domain="input_valid_only",
                    dtype="<f4", members=selection["sma"])
    if any(manifest[key] != value for key, value in expected.items()) or set(manifest["event_counts"]) != set(events):
        raise ValueError("Expected predictions from the frozen six-event validation cache")
    return manifest, {event: cache_file(cache, manifest, event) for event in events}


def supplement(raw, x, sar, cache, directory):
    """Persist only the two missing physical covariates in the frozen pixel order."""
    counts = {event: len(z) for event, z in x.items()}
    identity = dict(version=SUPPLEMENT, split="validation", raw_sha256=RAW_SHA256,
                    mean_cache_sha256=digest(cache / "manifest.json"), event_counts=counts,
                    features=["post_vh_db", "slope_degrees"])
    if not directory.exists():
        data = KuroDataset(raw, "validation")
        rows = data.rows
        affine = np.stack(rows["stac:geotransform"])
        if (set(rows.actid.astype(str)) != set(x) or not rows["stac:crs"].eq("EPSG:3857").all()
                or not np.allclose(affine[:, [1, 2, 4, 5]], [10, 0, 0, -10], rtol=0, atol=1e-6)):
            raise ValueError("Unexpected validation events or native grid")
        centers = affine[:, [0, 3]] + [1120, -1120]
        _, lat = transform("EPSG:3857", "EPSG:4326", *centers.T)
        ground_m = 10 * np.cos(np.radians(lat))
        directory.mkdir(parents=True, exist_ok=False)
        arrays = {event: np.lib.format.open_memmap(directory / f"{event}.npy", mode="w+",
                  dtype=np.float32, shape=(n, 2)) for event, n in counts.items()}
        offsets = dict.fromkeys(x, 0)
        for i, (_, row) in enumerate(rows.iterrows()):
            data.archive.check_unchanged()
            sample = preprocess(*read_sample(data.archive.path, row, include_labels=False),
                                include_labels=False)
            valid = sample["input_valid"]
            event = str(row.actid)
            start = offsets[event]
            stop = start + int(valid.sum())
            radar = sample["sar_db"][:, :, valid]
            delta = radar[2] - (radar[0] + radar[1]) * .5
            if stop > counts[event] or not np.array_equal(
                    np.column_stack((radar[2, 0], delta[0], delta[1])), sar[event][start:stop]):
                raise ValueError(f"SAR cache alignment mismatch at validation patch {i}")
            arrays[event][start:stop, 0] = radar[2, 1]
            arrays[event][start:stop, 1] = slope_degrees(sample["dem_m"][0], valid, ground_m[i])[valid]
            offsets[event] = stop
            if (i + 1) % 250 == 0 or i + 1 == len(rows):
                print(f"Aligned extra covariates: {i+1}/{len(rows)} validation patches", flush=True)
        if offsets != counts:
            raise ValueError("Incomplete extra covariates")
        for array in arrays.values():
            array.flush()
        write_json(directory / "manifest.json", identity | {
            "file_sha256": {event: digest(directory / f"{event}.npy") for event in x}})
    manifest = load_json(directory / "manifest.json")
    if any(manifest[key] != value for key, value in identity.items()):
        raise ValueError("Post-VH/slope cache provenance changed")
    result = {}
    for event, n in counts.items():
        path = directory / f"{event}.npy"
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.shape != (n, 2) or values.dtype != np.float32 or digest(path) != manifest["file_sha256"][event]:
            raise ValueError(f"Invalid supplement file: {path}")
        result[event] = values
    return result


def feature_rows(x, sar, extra, event, indices):
    margin = flood_margin(x[event][indices])
    vv, delta_vv, delta_vh = sar[event][indices].T
    vh, slope = extra[event][indices].T
    return np.column_stack((margin, vv, vh, delta_vv, delta_vh, vh - vv, slope)).astype(np.float64)


def draw(indices, event, *, target):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, int(event), int(target)]))
    if isinstance(indices, int):
        return (np.sort(rng.choice(indices, FIT_CAP, replace=False))
                if indices > FIT_CAP else np.arange(indices))
    return np.sort(rng.choice(indices, FIT_CAP, replace=False)) if len(indices) > FIT_CAP else indices


def fit_domain(source_arrays, target, width):
    # Unit total mass per source event, then equal mass for source and target domains.
    ns = sum(map(len, source_arrays))
    nt = len(target)
    source_weights = np.concatenate([np.full(len(a), .5 / (len(source_arrays) * len(a))) for a in source_arrays])
    weights = np.r_[source_weights, np.full(nt, .5 / nt)]
    labels = np.r_[np.zeros(ns, dtype=np.uint8), np.ones(nt, dtype=np.uint8)]
    x = np.concatenate((*source_arrays, target))[:, :width]
    imputed = None
    if width == len(FEATURES):
        observed = np.isfinite(x[:ns, -1])
        if not observed.any():
            raise ValueError("No observed source slope in the fit sample")
        imputed = float(np.average(x[:ns, -1][observed], weights=source_weights[observed]))
        x[~np.isfinite(x[:, -1]), -1] = imputed
    scaler = StandardScaler().fit(x, sample_weight=weights)
    lr = LogisticRegression(C=C, solver="lbfgs", max_iter=1000, tol=1e-8)
    lr.fit(scaler.transform(x), labels, sample_weight=weights * len(labels))
    if lr.n_iter_[0] >= lr.max_iter:
        raise RuntimeError("Domain Logistic Regression did not converge")
    # Use the effective weighted prior. Raw counts describe the sampled rows.
    prior = float(np.average(labels, weights=weights))
    prior_offset = np.log1p(-prior) - np.log(prior)
    beta = lr.coef_[0] / scaler.scale_
    intercept = float(lr.intercept_[0] - beta @ scaler.mean_)
    return dict(width=width, beta=beta.tolist(), intercept=intercept, source_slope_imputation=imputed,
                weighted_target_prior=prior, sampled_target_prior=nt / (ns + nt),
                prior_log_odds_correction=float(prior_offset), source_fit_rows=ns, target_fit_rows=nt,
                fit_iterations=int(lr.n_iter_[0]))


def log_ratio(features, model):
    x = features[:, :model["width"]]
    if model["source_slope_imputation"] is not None:
        x = x.copy()
        x[~np.isfinite(x[:, -1]), -1] = model["source_slope_imputation"]
    result = x @ model["beta"] + model["intercept"] + model["prior_log_odds_correction"]
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite density-ratio logits")
    return result


def counts_from_bins(positive, negative):
    tp = positive.cumsum()[:-1]
    fp = negative.cumsum()[:-1]
    fn = positive[::-1].cumsum()[::-1][1:]
    union = tp + fp + fn
    return tp, fp, fn, np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)


def bins_for_margin(margin, grid):
    # Flood wins only if its biased logit is strictly above both other classes.
    return np.searchsorted(grid, -margin.astype(np.float64), side="right")


def weighted_source_curve(sources, source_y, n_by_event, features, model, grid):
    nb = len(grid) + 1
    positive, negative = np.zeros((2, nb), dtype=np.float64)
    sum_w, sum_w2, base_w2, raw_w, raw_w2 = 0., 0., 0., 0., 0.
    source_overlap, clip_low, clip_high = 0., 0., 0.
    low, high = np.log(RATIO_CLIP)
    # Scale all raw weights equally to keep the pre-clip ESS computation finite.
    log_scale = -np.inf
    for event in sources:
        base = 1 / (len(sources) * n_by_event[event])
        base_w2 += n_by_event[event] * base**2
        y = source_y[event]
        for start in range(0, len(y), CHUNK):
            indices = np.flatnonzero(y[start:start + CHUNK] != IGNORE) + start
            if not len(indices):
                continue
            x = features(event, indices)
            lr = log_ratio(x, model)
            propensity = expit(lr)
            source_overlap += base * np.count_nonzero((propensity >= OVERLAP[0]) & (propensity <= OVERLAP[1]))
            clip_low += base * np.count_nonzero(lr < low)
            clip_high += base * np.count_nonzero(lr > high)
            w = base * np.exp(np.clip(lr, low, high))
            sum_w += w.sum()
            sum_w2 += w @ w
            new_scale = max(log_scale, float(lr.max()))
            previous_scale = np.exp(log_scale - new_scale)
            raw = base * np.exp(lr - new_scale)
            raw_w = raw_w * previous_scale + raw.sum()
            raw_w2 = raw_w2 * previous_scale**2 + raw @ raw
            log_scale = new_scale
            idx = bins_for_margin(x[:, 0], grid)
            flood = y[indices] == FLOOD
            positive += np.bincount(idx[flood], weights=w[flood], minlength=nb)
            negative += np.bincount(idx[~flood], weights=w[~flood], minlength=nb)
    ess = sum_w**2 / sum_w2
    return counts_from_bins(positive, negative), dict(
        ess_clipped=float(ess), ess_unclipped=float(raw_w**2 / raw_w2),
        ess_base=float(1 / base_w2), ess_fraction_of_base=float(ess * base_w2),
        source_overlap_mass=float(source_overlap), source_clipped_low_mass=float(clip_low),
        source_clipped_high_mass=float(clip_high), mean_clipped_ratio=float(sum_w))


def flood_counts(confusion):
    tp = confusion[:, FLOOD, FLOOD]
    fp = confusion[:, :, FLOOD].sum(axis=1) - tp
    fn = confusion[:, FLOOD, :].sum(axis=1) - tp
    return tp, fp, fn


def iou(tp, fp, fn):
    return tp / np.maximum(tp + fp + fn, 1)


def lock_fold(target, events, manifest, cache, features, grid):
    sources = [event for event in events if event != target]
    source_y = {e: cache_file(cache, manifest, e, labels=True) for e in sources}
    with np.load(VAL_ARTIFACT / "validation_confusion_eoa_flip_tta.npz") as archive:
        confusions = {e: archive[e] for e in sources}
    ref, ref_curve = reference_from_sources(sources, confusions, grid)
    pooled_counts = flood_counts(sum(confusions.values()))
    pooled = iou(*pooled_counts)
    curves = {POOL: (*pooled_counts, pooled), REF: (None, None, None, ref_curve)}
    indices = {POOL: best_index(pooled, grid), REF: ref}

    source_sizes, source_samples = {}, []
    for event in sources:
        eligible = np.flatnonzero(source_y[event] != IGNORE)
        source_sizes[event] = len(eligible)
        if not len(eligible):
            raise ValueError(f"No scored source pixels for {event}")
        source_samples.append(features(event, draw(eligible, event, target=target)))
    target_indices = draw(manifest["event_counts"][target], target, target=target)
    target_sample = features(target, target_indices)
    diagnostics = {}
    for method, width in ((MARGIN, 1), (SAR, len(FEATURES))):
        model = fit_domain(source_samples, target_sample, width)
        curves[method], stats = weighted_source_curve(sources, source_y, source_sizes, features, model, grid)
        p = expit(log_ratio(target_sample, model))
        stats.update(target_overlap_fraction=float(np.mean((p >= OVERLAP[0]) & (p <= OVERLAP[1]))),
                     source_scored_rows=sum(source_sizes.values()),
                     target_input_valid_rows=manifest["event_counts"][target])
        if method == SAR:
            stats.update(target_fit_slope_missing_fraction=float(np.mean(~np.isfinite(target_sample[:, -1]))),
                         source_fit_slope_missing_mass=float(np.mean([
                             np.mean(~np.isfinite(a[:, -1])) for a in source_samples])))
        diagnostics[method] = model | stats
        indices[method] = best_index(curves[method][-1], grid)

    lock = dict(source_events=sources, source_scored_pixels=source_sizes, biases={
        method: dict(bias=float(grid[k]), objective=float(curves[method][-1][k]))
        for method, k in indices.items()}, domain_classifiers=diagnostics)
    return lock, curves


def evaluate_locked(target, cache, manifest, x, grid, lock, historical):
    y = cache_file(cache, manifest, target, labels=True)
    keep = y != IGNORE
    rank = rank_event([(flood_margin(x)[keep], y[keep] == FLOOD)])
    oracle_bias, oracle_iou = oracle_from_ranks(rank)
    checked([oracle_bias, oracle_iou],
            [float(historical["exact_oracle_bias"]), float(historical["exact_oracle_iou"])],
            f"{target} exact oracle")
    with np.load(VAL_ARTIFACT / "validation_confusion_eoa_flip_tta.npz") as archive:
        archive_tp, archive_fp, archive_fn = flood_counts(archive[target])
    score, positive, totals = rank
    thresholds = np.r_[grid, oracle_bias]
    k = np.searchsorted(score.astype(np.float64), -thresholds, side="right")
    tp = np.r_[positive[::-1].cumsum()[::-1], 0][k]
    pred = np.r_[totals[::-1].cumsum()[::-1], 0][k]
    fp, fn = pred - tp, positive.sum() - tp
    actual = iou(tp, fp, fn)
    for measured, expected in zip((tp[:-1], fp[:-1], fn[:-1]),
                                  (archive_tp, archive_fp, archive_fn), strict=True):
        np.testing.assert_array_equal(measured, expected, err_msg=f"Archived grid mismatch: {target}")
    checked(actual[-1], oracle_iou, f"{target} exact oracle operating point")
    g = best_index(actual[:-1], grid)
    checked([grid[g], actual[g]], [float(historical["grid_oracle_bias"]),
                                  float(historical["grid_oracle_iou"])], f"{target} grid oracle")
    rows = []
    for method in METHODS:
        bias = oracle_bias if method == ORACLE else lock["biases"][method]["bias"]
        b = len(grid) if method == ORACLE else int(np.searchsorted(grid, bias))
        rows.append(dict(event_id=target, method=method, bias=bias,
                         tp=int(tp[b]), fp=int(fp[b]), fn=int(fn[b]),
                         flood_iou=float(actual[b]), flood_f1=float(2*tp[b] / max(2*tp[b] + fp[b] + fn[b], 1)),
                         flood_prevalence=float(positive.sum() / totals.sum()),
                         predicted_flood_prevalence=float(pred[b] / totals.sum()),
                         exact_oracle_bias=oracle_bias, exact_oracle_iou=oracle_iou,
                         exact_oracle_gap_pp=float(100 * (oracle_iou - actual[b])),
                         grid_oracle_bias=float(grid[g]), grid_oracle_iou=float(actual[g]),
                         grid_oracle_gap_pp=None if method == ORACLE else float(100 * (actual[g] - actual[b]))))
    return rows, actual[:-1], dict(event_id=target, exact_oracle_bias=oracle_bias, exact_oracle_iou=oracle_iou,
                                  grid_oracle_bias=float(grid[g]), grid_oracle_iou=float(actual[g]),
                                  exact_bias_outside_grid=not grid[0] <= oracle_bias <= grid[-1],
                                  matches_historical=True)


def summarize(rows):
    results = []
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        tp, fp, fn = [sum(row[key] for row in group) for key in ("tp", "fp", "fn")]
        values = [row["flood_iou"] for row in group]
        worst = min(group, key=lambda row: row["flood_iou"])
        results.append(dict(method=method, events=len(group),
                            pooled_flood_f1=2 * tp / max(2 * tp + fp + fn, 1),
                            event_macro_flood_iou=float(np.mean(values)),
                            event_flood_iou_std=float(np.std(values, ddof=0)),
                            worst_event_iou=worst["flood_iou"], worst_event_id=worst["event_id"],
                            mean_exact_oracle_gap_pp=float(np.mean([r["exact_oracle_gap_pp"] for r in group])),
                            mean_grid_oracle_gap_pp=None if method == ORACLE else
                            float(np.mean([r["grid_oracle_gap_pp"] for r in group]))))
    return results


def run(args):
    recipe, selection, grid, events = reference(load_confusions=False)
    events = sorted(events, key=int)
    manifest, x = open_predictions(args.cache, recipe, selection, events)
    if args.out.exists():
        raise FileExistsError(args.out)
    raw = selection["protocol"]["training"]["raw"]
    sar = get_sar(raw, x, args.cache, args.sar_cache)
    extra = supplement(raw, x, sar, args.cache, args.extra_cache)
    args.out.mkdir(parents=True, exist_ok=False)

    def features(event, indices):
        return feature_rows(x, sar, extra, event, indices)

    locks, curves = {}, {}
    for target in events:
        locks[target], curves[target] = lock_fold(target, events, manifest, args.cache, features, grid)
        write_json(args.out / f"lock_{target}.json", locks[target])
        print(f"Locked {target}: " + ", ".join(
            f"{name}={item['bias']:+.2f}" for name, item in locks[target]["biases"].items()), flush=True)
    lock_path = args.out / "bias_lock.json"
    write_json(lock_path, dict(split="validation", candidate_grid=grid.tolist(), events=locks))
    previous = {r["event_id"]: r for r in read_csv(args.oracle_reference)}
    if set(previous) != set(events):
        raise ValueError("Oracle reference does not cover the frozen validation events")

    all_rows, all_curves, audit, fit_stats = [], [], [], []
    for target in events:
        lock = load_json(args.out / f"lock_{target}.json")
        rows, actual_curve, oracle = evaluate_locked(target, args.cache, manifest, x[target], grid,
                                                     lock, previous[target])
        all_rows.extend(rows)
        audit.append(oracle)
        for method, values in curves[target].items():
            tp, fp, fn, objective = values
            selected = lock["biases"][method]["bias"]
            for i, b in enumerate(grid):
                all_curves.append(dict(event_id=target, method=method, bias=float(b),
                                       selected=float(b) == selected, source_objective_iou=float(objective[i]),
                                       actual_target_iou=float(actual_curve[i]),
                                       source_tp=None if tp is None else float(tp[i]),
                                       source_fp=None if fp is None else float(fp[i]),
                                       source_fn=None if fn is None else float(fn[i])))
        for method, stats in lock["domain_classifiers"].items():
            fit_stats.append(dict(event_id=target, method=method,
                                  **{k: v for k, v in stats.items() if k not in ("beta", "intercept")}))
    save_csv(args.out / "summary.csv", summarize(all_rows))
    save_csv(args.out / "event_metrics.csv", all_rows)
    save_csv(args.out / "event_279.csv", [r for r in all_rows if r["event_id"] == "279"])
    save_csv(args.out / "curves.csv", all_curves)
    save_csv(args.out / "density_ratio_diagnostics.csv", fit_stats)
    save_csv(args.out / "oracle_audit.csv", audit)
    write_json(args.out / "evaluation_config.json", dict(
        split="validation", scope="exploratory; frozen checkpoints selected on these six events historically",
        selection_sha256=recipe["selection_sha256"], mean_cache_sha256=digest(args.cache / "manifest.json"),
        sar_cache_sha256=digest(args.sar_cache / "manifest.json"),
        extra_cache_sha256=digest(args.extra_cache / "manifest.json"),
        validation_confusions_sha256=digest(VAL_ARTIFACT / "validation_confusion_eoa_flip_tta.npz"),
        bias_lock_sha256=digest(lock_path), historical_oracle_sha256=digest(args.oracle_reference),
        features={MARGIN: [FEATURES[0]], SAR: list(FEATURES)}, C=C, fit_cap_per_event=FIT_CAP, seed=SEED,
        slope="centered DEM differences, native 10 m EPSG:3857 spacing times cos(patch center latitude)",
        slope_missing="impute with weighted source-fit mean over observed slopes",
        domain_fit="equal mass for each source event within source domain; total source and target mass each 0.5",
        density_ratio="target-vs-source odds times weighted training prior(source)/prior(target)",
        domain_fit_support="scored source pixels and input-valid target pixels; uniform samples within each event",
        target_labels="read only in the evaluation of their own fold after the serialized bias lock",
        ratio_clip=list(RATIO_CLIP), weighted_objective="sum clipped density ratio times equal-event base mass "
        "over all scored source pixels, followed by TP/(TP+FP+FN)",
        overlap="equal-prior propensity interval; fitted domain classifier; full scored source and sampled input-valid target",
        overlap_interval=list(OVERLAP), tie="bias nearest zero then lower bias", event_std_ddof=0,
        exact_oracle="all attainable target thresholds on scored pixels; diagnostic only",
        code_sha256=digest(Path(__file__))))
    print(f"Saved IW-IoU evaluation: {args.out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "sar-cache", "extra-cache", "oracle-reference", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
