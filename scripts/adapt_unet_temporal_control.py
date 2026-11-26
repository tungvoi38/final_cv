#!/usr/bin/env python3
"""Exploratory temporal negative-control bias on the frozen EoA validation split."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
import torch
from torch.utils.data import DataLoader, Dataset

from adapt_unet_event_bias import (
    FLOOD, IGNORE, SEED, VAL_ARTIFACT, digest, flood_margin, load_json,
    predict_confusion, reference, scores, write_json,
)
from adapt_unet_expected_iou import CHUNK, sar_features
from adapt_unet_iw_iou import cache_file, open_predictions
from audit_unet_bias_failures import checked, save_csv
from audit_unet_iw_iou import GLOBAL, MARGIN, ORACLE, REF, SAR, load_artifacts
from evaluate_unet_eoa import (
    load_calibrated_selection, load_model, pre_variability_inputs, seed_everything,
)
from evaluate_unet_eoa_flip_tta import FlipTTAEnsemble
from kuro_dataset import KuroDataset, RAW_SHA256, normalize_physical_inputs, preprocess, read_sample
from unet_conservative_expected_iou import get_sar, reference_from_sources

CONTROL = "temporal_negative_control"
METHODS = (GLOBAL, REF, MARGIN, SAR, CONTROL, ORACLE)
CONFUSIONS = VAL_ARTIFACT / "validation_confusion_eoa_flip_tta.npz"
VERSION = "pre1_pre2_pre2_v1"
DIAG_CAP = 100_000
ABS_TOL = .005    # 0.5 percentage points of FPR
REL_TOL = .25    # 25% of the source budget
NAMES = ("control_all", "control_matched_non_flood", "factual_non_flood")


class ControlPatches(Dataset):
    """Read pre SAR and DEM without annotation TIFFs; retain factual input support."""

    def __init__(self, raw):
        self.data = KuroDataset(raw, "validation")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        self.data.archive.check_unchanged()
        row = self.data.rows.iloc[i]
        sample = preprocess(
            *read_sample(self.data.archive.path, row, include_labels=False), include_labels=False
        )
        valid = sample["input_valid"]
        alignment = sar_features(sample["sar_db"], valid)
        sample["sar_db"][2] = sample["sar_db"][1]
        sar, dem = normalize_physical_inputs(sample["sar_db"], sample["dem_m"], valid)
        return str(row.actid), sar, dem, valid, alignment


def collate(samples):
    events, sar, dem, valid, alignment = zip(*samples, strict=True)
    return events, dict(sar=torch.from_numpy(np.stack(sar)), dem=torch.from_numpy(np.stack(dem))), valid, alignment


def cache_identity(cache, sar_cache, recipe, predictions):
    return dict(version=VERSION, split="validation", raw_sha256=RAW_SHA256,
                selection_sha256=recipe["selection_sha256"], flip_axes=recipe["flip_axes"],
                factual_manifest_sha256=digest(cache/"manifest.json"),
                sar_manifest_sha256=digest(sar_cache/"manifest.json"),
                event_counts={e:len(values) for e, values in predictions.items()},
                dtype="<f4", inference="CUDA autocast float16; averaging and margin float32")


@torch.inference_mode()
def build_control(args, recipe, selection, factual, identity, *, smoke=False):
    path = args.control_cache
    if not smoke and path.exists():
        raise FileExistsError(path)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Use an available CUDA GPU to match the existing FP16 inference protocol")
    verified = load_calibrated_selection(recipe, Path(recipe["selection"]))
    settings = selection["protocol"]["training"]
    alignment_cache = get_sar(settings["raw"], factual, args.cache, args.sar_cache)
    patches = ControlPatches(settings["raw"])
    counts = identity["event_counts"]
    if set(patches.data.rows.actid.astype(str)) != set(counts):
        raise ValueError("Raw validation events differ from cached events")
    loader = DataLoader(patches, batch_size=1 if smoke else settings["batch_size"],
                        num_workers=0 if smoke else args.workers, shuffle=False, collate_fn=collate)
    seed_everything(SEED)
    ensemble = FlipTTAEnsemble([load_model(member, device) for member in verified["sma"]]).eval()
    offsets = dict.fromkeys(counts, 0)
    if not smoke:
        path.mkdir(parents=True, exist_ok=False)
        files = {e: np.memmap(path/f"{e}.margin.f32", dtype="<f4", mode="w+", shape=(n,))
                 for e, n in counts.items()}
    for step, (events, batch, valid, alignment) in enumerate(loader, 1):
        inputs, _ = pre_variability_inputs({key:value.to(device) for key, value in batch.items()})
        if smoke:
            torch.testing.assert_close(inputs[:, 4:6], inputs[:, 2:4], rtol=0, atol=0)
            difference = inputs[:, 2:4] - inputs[:, :2]
            torch.testing.assert_close(inputs[:, 7:9], difference*.5, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(inputs[:, 9:11], difference.abs(), rtol=0, atol=0)
        with torch.autocast("cuda"):
            logits = ensemble(inputs).float()
        if tuple(logits.shape) != (len(events), 3, 224, 224):
            raise ValueError("Unexpected EoA logits shape")
        margins = (logits[:, FLOOD] - logits[:, :FLOOD].amax(dim=1)).cpu().numpy()
        for i, event in enumerate(events):
            values = margins[i][valid[i]]
            start = offsets[event]
            stop = start + len(values)
            if (stop > counts[event] or not np.isfinite(values).all()
                    or not np.array_equal(alignment[i], alignment_cache[event][start:stop])):
                raise ValueError(f"Nonfinite margin or pixel alignment mismatch: {event}, batch {step}")
            if not smoke:
                files[event][start:stop] = values
            offsets[event] = stop
        if smoke:
            print(f"PASS: {inputs.shape[1]} channels, frozen EoA + Flip TTA, aligned event {events[0]}")
            return
        if step % 100 == 0 or step == len(loader):
            print(f"Negative-control validation inference: {step}/{len(loader)} batches", flush=True)
    if offsets != counts:
        raise ValueError("Incomplete control cache")
    for array in files.values():
        array.flush()
    write_json(path/"manifest.json", identity | {"inference_script_sha256": digest(Path(__file__)),
               "file_sha256": {e:digest(path/f"{e}.margin.f32") for e in counts}})


def open_control(path, identity):
    manifest = load_json(path/"manifest.json")
    if any(manifest[k] != v for k, v in identity.items()):
        raise ValueError("Negative-control and factual cache provenance differ")
    result = {}
    for event, n in identity["event_counts"].items():
        file = path/f"{event}.margin.f32"
        if not n or file.stat().st_size != 4*n or digest(file) != manifest["file_sha256"][event]:
            raise ValueError(f"Invalid negative-control scores: {event}")
        result[event] = np.memmap(file, dtype="<f4", mode="r", shape=(n,))
    return result


def fpr(confusion):
    return float(confusion[:FLOOD, FLOOD].sum()/confusion[:FLOOD, :].sum())


def upper_tail_bias(margin, budget):
    """Largest attainable decision set with control alarm rate at most budget."""
    if not len(margin) or not 0 <= budget <= 1:
        raise ValueError("Control quantile needs nonempty margins and a valid FPR budget")
    alarms = int(np.floor(budget * len(margin)))
    if alarms == len(margin):
        return float(np.nextafter(-np.float64(np.min(margin)), np.inf))
    index = len(margin) - alarms - 1
    # At the empirical threshold, ties go to the non-flood class.
    return -float(np.partition(margin, index)[index])


def lock_biases(out, grid, control, control_manifest_hash, factual_manifest_hash):
    locked = {}
    for event, values in control.items():
        sources = [e for e in control if e != event]
        with np.load(CONFUSIONS, allow_pickle=False) as archive:
            confusion = {e: archive[e] for e in sources}
        index, _ = reference_from_sources(sources, confusion, grid)
        per_event = {e: fpr(confusion[e][index]) for e in sources}
        budget = float(np.mean(list(per_event.values())))
        bias = upper_tail_bias(values, budget)
        control_fpr = sum(np.count_nonzero(values[i:i+CHUNK].astype(np.float64) > -bias)
                          for i in range(0, len(values), CHUNK)) / len(values)
        if control_fpr > budget + 1e-12:
            raise ValueError(f"Control alarm rate exceeds budget: {event}")
        locked[event] = dict(source_events=sources, source_reference_bias=float(grid[index]),
                             source_fpr=per_event, fpr_budget=budget, bias=bias,
                             control_alarm_rate=control_fpr, outside_grid=not grid[0] <= bias <= grid[-1])
        print(f"Locked {event}: b={bias:+.4f}, budget={100*budget:.3f}%, "
              f"control alarm rate={100*control_fpr:.3f}%", flush=True)
    write_json(out/"bias_lock.json", dict(split="validation", construction=VERSION,
               control_manifest_sha256=control_manifest_hash, factual_manifest_sha256=factual_manifest_hash,
               rules=dict(budget="mean of five source-event nonflood FPRs at source macro-IoU reference bias",
                          candidate="exact empirical control quantile; ties go to nonflood",
                          diagnostic_tolerance=dict(absolute_fpr=ABS_TOL, fraction_of_source_budget=REL_TOL)),
               events=locked))
    return locked


def diagnose_event(event, x, labels, control, lock, biases):
    bias = lock["bias"]
    matrices = {method:np.zeros((3, 3), dtype=np.int64) for method in biases}
    matched_alarms = matched_n = 0
    for start in range(0, len(x), CHUNK):
        sl = slice(start, start+CHUNK)
        y = labels[sl]
        valid = y != IGNORE
        nonflood = y < FLOOD
        z = x[sl][valid]
        for method, value in biases.items():
            matrices[method] += predict_confusion(z, y[valid], value)
        matched_n += int(nonflood.sum())
        matched_alarms += int(np.count_nonzero(control[sl][nonflood].astype(np.float64) > -bias))
    actual_fpr = fpr(matrices[CONTROL])
    matched_fpr = matched_alarms/matched_n
    budget = lock["fpr_budget"]
    control_fpr = lock["control_alarm_rate"]
    tolerance = max(ABS_TOL, REL_TOL*budget)
    allowed = (abs(actual_fpr-budget) <= tolerance and abs(actual_fpr-control_fpr) <= tolerance)

    rng = np.random.default_rng(np.random.SeedSequence([SEED, int(event)]))
    indices = rng.choice(len(control), size=min(DIAG_CAP, len(control)), replace=False)
    paired = labels[indices] < FLOOD
    sample = dict(zip(NAMES, (control[indices], control[indices][paired],
                             flood_margin(x[indices[paired]])), strict=True))
    factual = sample["factual_non_flood"]
    if not len(factual):
        raise ValueError(f"No factual non-flood sample in {event}")
    diagnostics = dict(event_id=event, **lock, actual_non_flood_fpr=actual_fpr,
                       control_matched_non_flood_fpr=matched_fpr,
                       actual_minus_budget_pp=100*(actual_fpr-budget),
                       actual_minus_control_all_pp=100*(actual_fpr-control_fpr),
                       actual_minus_control_matched_pp=100*(actual_fpr-matched_fpr),
                       scored_non_flood_pixels=matched_n, tolerance_pp=100*tolerance,
                       budget_transfer_within_tolerance=allowed)
    for key, values in sample.items():
        q50,q90,q95,q99 = np.quantile(values, (.5,.9,.95,.99))
        diagnostics.update({f"{key}_sample_n":len(values), f"{key}_median":q50,
                            f"{key}_p90":q90, f"{key}_p95":q95, f"{key}_p99":q99})
        if key != "factual_non_flood":
            diagnostics[f"{key}_ks"] = ks_2samp(values, factual, method="asymp").statistic
            diagnostics[f"{key}_wasserstein"] = wasserstein_distance(values, factual)
    return matrices, diagnostics, sample


def plots(out, samples, mismatch):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for ax, row in zip(axes.flat, mismatch, strict=True):
        for name, style in zip(NAMES, ("-", "--", "-."), strict=True):
            scores = np.sort(samples[row["event_id"]][name])
            values = np.linspace(0, len(scores)-1, min(1800,len(scores))).astype(int)
            thresholds = scores[values]
            tail = 1-np.searchsorted(scores, thresholds, side="right")/len(scores)
            ax.step(thresholds, tail, style, label=name.replace("_"," "), where="post")
        ax.axvline(-row["bias"], color=".35", linestyle=":", label="locked threshold")
        ax.axhline(row["fpr_budget"], color=".65", linestyle=":", label="source FPR budget")
        ax.set(title=f"Event {row['event_id']}", xlabel="Flood margin threshold", ylabel="Upper-tail fraction")
        ax.set_yscale("symlog", linthresh=.001)
    axes.flat[0].legend(fontsize=7)
    fig.savefig(out/"margin_tails.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.4), layout="constrained")
    position = np.arange(len(mismatch))
    for i,(key,label) in enumerate((("fpr_budget", "Source budget"), ("control_alarm_rate", "Control, all"),
                                    ("actual_non_flood_fpr", "Factual non-flood"))):
        ax.bar(position + (i-1)*.26, [100*row[key] for row in mismatch], width=.25, label=label)
    ax.set(xticks=position, xticklabels=[r["event_id"] for r in mismatch], ylabel="False alarms (%)",
           xlabel="Held-out validation event")
    ax.legend()
    fig.savefig(out/"false_alarm_transfer.png", dpi=170)
    plt.close(fig)


def evaluate(args, manifest, grid, factual, identity):
    control = open_control(args.control_cache, identity)
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    lock_biases(out, grid, control, digest(args.control_cache/"manifest.json"),
                digest(args.cache/"manifest.json"))
    locked = load_json(out/"bias_lock.json")["events"]
    events, comparators, _ = load_artifacts(args.iw_results, args.bias_results, CONFUSIONS)
    known = {(row["event_id"], row["method"]):row for row in comparators}
    pooled = {method:np.zeros((3, 3), dtype=np.int64) for method in METHODS}
    all_results, mismatch, samples = [], [], {}
    for event in events:
        labels = cache_file(args.cache, manifest, event, labels=True)
        entry = locked[event]
        prior = known[event, REF]
        checked(entry["source_reference_bias"], prior["bias"], f"{event} Source Reference Bias")
        biases = {m:known[event, m]["bias"] for m in (GLOBAL, REF, MARGIN, SAR)}
        biases.update({CONTROL:entry["bias"], ORACLE:prior["oracle_bias"]})
        matrices, diagnostic, samples[event] = diagnose_event(event, factual[event], labels,
                                                               control[event], entry, biases)
        mismatch.append({key:value for key,value in diagnostic.items() if key not in ("source_events","source_fpr")})
        for method, cm in matrices.items():
            result = scores(cm)
            tp = int(cm[FLOOD, FLOOD])
            fp = int(cm[:FLOOD, FLOOD].sum())
            fn = int(cm[FLOOD, :FLOOD].sum())
            if (event, method) in known:
                observed = known[event, method]
                checked([tp,fp,fn], [observed[k] for k in ("tp","fp","fn")], f"{event} {method} counts")
            if method == ORACLE:
                checked(result["flood_iou"], prior["oracle_iou"], f"{event} historical oracle")
            pooled[method] += cm
            all_results.append(dict(event_id=event, method=method, bias=biases[method], tp=tp, fp=fp, fn=fn,
                 non_flood_fpr=fpr(cm), exact_oracle_iou=prior["oracle_iou"],
                 oracle_gap_pp=100*(prior["oracle_iou"]-result["flood_iou"]), **result))
    summary = []
    for method in METHODS:
        rows = [row for row in all_results if row["method"] == method]
        worst = min(rows, key=lambda row:row["flood_iou"])
        values = [r["flood_iou"] for r in rows]
        s = scores(pooled[method])
        summary.append(dict(method=method, events=len(rows), pooled_miou=s["pooled_miou"],
             **{f"pooled_{k}":v for k,v in s.items() if k != "pooled_miou"},
             event_macro_flood_iou=float(np.mean(values)), event_flood_iou_std=float(np.std(values)),
             worst_event_iou=worst["flood_iou"], worst_event_id=worst["event_id"],
             mean_oracle_gap_pp=float(np.mean([r["oracle_gap_pp"] for r in rows]))))
    save_csv(out/"summary.csv", summary)
    save_csv(out/"event_metrics.csv", all_results)
    save_csv(out/"distribution_mismatch.csv", mismatch)
    plots(out, samples, mismatch)
    failed = [row["event_id"] for row in mismatch if not row["budget_transfer_within_tolerance"]]
    status = "STOP_control_did_not_transfer" if failed else "within_tolerance_on_six_exploratory_events"
    write_json(out/"evaluation_config.json", dict(split="validation", status=status, failed_events=failed,
               bias_lock_sha256=digest(out/"bias_lock.json"), selection_sha256=identity["selection_sha256"],
               iw_results=str(args.iw_results), historical_bias_results=str(args.bias_results),
               iw_metrics_sha256=digest(args.iw_results/"event_metrics.csv"),
               historical_bias_metrics_sha256=digest(args.bias_results/"event_metrics.csv"),
               distribution_statistics="descriptive KS and Wasserstein on a deterministic uniform sample "
                                       "of at most 100000 aligned pixels per event",
               gate="actual FPR must be within max(0.5 pp, 25% of budget) of both source budget and "
                    "achieved all-control alarm rate, for every event; fixed descriptive rule",
               caveats=["a SAR acquisition is duplicated in the post slot, so temporal dependence changes",
                        "pre acquisitions are pseudo negatives, without independent no-flood verification",
                        "target negative control for bias uses all input-valid pixels",
                        "factual and control share factual input-valid support; no labels used before lock",
                        "frozen checkpoint and global bias were historically selected on these six events"],
               code_sha256=digest(Path(__file__))))
    print(f"{status}: failed events={','.join(failed) or 'none'}; results={out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "infer", "analyze"))
    for name in ("cache", "sar-cache", "control-cache"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    for name in ("iw-results", "bias-results", "out"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", default=4, type=int)
    args = parser.parse_args()
    if args.mode == "analyze" and any(getattr(args,key) is None for key in ("iw_results","bias_results","out")):
        parser.error("analyze requires --iw-results, --bias-results and --out")
    recipe, selection, grid, events = reference(load_confusions=False)
    manifest, factual = open_predictions(args.cache, recipe, selection, sorted(events, key=int))
    identity = cache_identity(args.cache, args.sar_cache, recipe, factual)
    if args.mode == "analyze":
        evaluate(args, manifest, grid, factual, identity)
    else:
        build_control(args, recipe, selection, factual, identity, smoke=args.mode=="smoke")


if __name__ == "__main__":
    main()
