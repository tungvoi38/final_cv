#!/usr/bin/env python3
"""Audit saved A3 11-channel Online/SMA trajectories without model inference."""
import argparse
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import torch

from unet_sma import selected_epoch

EPOCHS = tuple(range(1, 21))
SEEDS = (42, 43, 44)
SMA_EPOCHS = dict(zip(SEEDS, (14, 18, 20), strict=True))
METRICS = ("pooled_miou", "pooled_flood_iou", "pooled_flood_f1", "pooled_flood_precision",
           "pooled_flood_recall", "event_macro_flood_iou")
MODELS = ("online", "sma")
KEYS = {"online": ("online_model", "online_validation", "online_history.csv", "best_online.pt"),
        "sma": ("model", "validation", "history.csv", "best.pt")}
LOG_LINE = re.compile(r"^epoch=(\d+) val online mIoU=([\d.]+) SMA mIoU=([\d.]+) "
                      r"steps=(\d+) count=(\d+) best SMA epoch=(\d+|None)$", re.MULTILINE)


@lru_cache
def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number_match(actual, expected):
    return np.isclose(float(actual), float(expected), atol=1e-12, rtol=0)


def read_trajectory(run, seed):
    folder = run / f"{run.name}-seed{seed}"
    cfg = read_json(folder / "config.json")
    summary = read_json(folder / "summary.json")
    require((cfg["run_id"], cfg["seed"], cfg["epochs"], cfg["input_channels"], cfg["averaging"],
             cfg["selection"]) == (folder.name, seed, 20, 11, "sma", "pooled_miou"),
            f"Unexpected run config: {folder}")
    require((summary["completed_epochs"], summary["selection"]) == (20, "pooled_miou"), str(folder))
    log_path = run / f"{folder.name}.log"
    logs = LOG_LINE.findall(log_path.read_text(encoding="utf-8"))
    require(tuple(int(entry[0]) for entry in logs) == EPOCHS, f"Incomplete log: {log_path}")
    histories = {model: read_csv(folder / KEYS[model][2]) for model in MODELS}
    for history in histories.values():
        require(tuple(int(r["epoch"]) for r in history) == EPOCHS, f"Incomplete history: {folder}")
    rows = []
    for index, epoch in enumerate(EPOCHS):
        checkpoint_path = folder / "checkpoints" / f"epoch_{epoch:03d}.pt"
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu", mmap=True)
        require(checkpoint["config"] == cfg and checkpoint["epoch"] == epoch, str(checkpoint_path))
        _, online_log, sma_log, steps, count, best_log = logs[index]
        sma_row = histories["sma"][index]
        require((int(steps), int(count)) == (int(sma_row["sma_steps"]), int(sma_row["sma_count"]))
                == (checkpoint["sma"]["steps"], checkpoint["sma"]["count"]), str(checkpoint_path))
        require(int(count) == max(0, int(steps) - cfg["sma_start_step"] + 1), str(checkpoint_path))
        current_best = selected_epoch(histories["sma"][:index + 1], require_sma=True)
        require(best_log == ("None" if current_best is None else str(current_best["epoch"])), str(log_path))
        for model, rounded in zip(MODELS, (online_log, sma_log), strict=True):
            weight_key, validation_key, history, _ = KEYS[model]
            row = histories[model][index]
            require(weight_key in checkpoint and tuple(checkpoint[weight_key]["encoder.conv1.weight"].shape)[1] == 11,
                    f"Missing 11-channel {weight_key}: {checkpoint_path}")
            require(abs(float(rounded) - float(row["pooled_miou"])) <= 0.00005001, str(log_path))
            require(all(number_match(row[m], checkpoint[validation_key][m]) for m in METRICS),
                    f"Checkpoint validation mismatches history: {checkpoint_path}")
            rows.append(dict(seed=seed, model=model, epoch=epoch, checkpoint=str(checkpoint_path),
                             weights_key=weight_key, validation_source=str(folder / history),
                             training_log=str(log_path),
                             selection_eligible=model == "online" or int(count) > 0,
                             **{f"val_{m}": float(row[m]) for m in METRICS},
                             **{f"test_{m}": None for m in METRICS}, test_source=None))
    require((summary["sma_steps"], summary["sma_count"]) == (int(steps), int(count)), str(folder))
    return folder, summary, rows


def rank_validation(rows):
    eligible = [row for row in rows if row["selection_eligible"]]
    ordered = sorted(eligible, key=lambda row: (-row["val_pooled_miou"], row["epoch"]))
    for row in rows:
        row["validation_rank"] = None
    for rank, row in enumerate(ordered, 1):
        row["validation_rank"] = rank
    return ordered


def check_selection(folder, summary, ranked, member):
    model = ranked[0]["model"]
    weight_key, _, _, filename = KEYS[model]
    best = ranked[0]
    epoch_key = "best_online_epoch" if model == "online" else "best_epoch"
    metric_key = "best_online_validation_score" if model == "online" else "best_validation_score"
    path = folder / filename
    require((member["run_id"], member["seed"], member["epoch"], member["weights_key"], member["input_channels"])
            == (folder.name, best["seed"], best["epoch"], weight_key, 11), f"Selection mismatch: {path}")
    require(Path(member["checkpoint"]).resolve() == path.resolve(), f"Checkpoint path mismatch: {path}")
    require(summary[epoch_key] == best["epoch"] and
            number_match(summary[metric_key], best["val_pooled_miou"]) and
            number_match(member["validation_pooled_miou"], best["val_pooled_miou"]), f"Best metric mismatch: {path}")
    require(digest(path) == member["sha256"], f"Best checkpoint hash differs from selection: {path}")
    epoch_path = Path(best["checkpoint"])
    require(epoch_path.samefile(path) or digest(epoch_path) == member["sha256"], f"Best artifact differs: {path}")
    if model == "sma":
        require(best["epoch"] == SMA_EPOCHS[best["seed"]], "Unexpected SMA epoch for this frozen study")
    return dict(seed=best["seed"], model=model, selected_epoch=best["epoch"],
                validation_pooled_miou=best["val_pooled_miou"], checkpoint=str(path),
                checkpoint_sha256=member["sha256"], history_checkpoint=str(epoch_path),
                weights_key=weight_key, verified=True)


def attach_test(row, values, source):
    for m in METRICS:
        if values.get(m) in (None, ""):
            continue
        value = float(values[m])
        require(np.isfinite(value) and 0 <= value <= 1, f"Invalid test metric in {source}")
        previous = row[f"test_{m}"]
        require(previous is None or number_match(previous, value), f"Conflicting test metric in {source}")
        row[f"test_{m}"] = value
    row["test_source"] = " | ".join(filter(None, (row["test_source"], str(source))))


def load_run(run, historical):
    selection_path = run / "comparison/selection.json"
    selection = read_json(selection_path)
    require(selection["selection_metric"] == "validation_pooled_miou", str(selection_path))
    test_path = run / "comparison/test/test_metrics.json"
    test = read_json(test_path)
    require((test["split"], test["selection_sha256"]) == ("test", digest(selection_path)), str(test_path))
    val_path = run / "comparison/validation_metrics.json"
    val = read_json(val_path)
    require((val["split"], val["selection_sha256"]) == ("validation", digest(selection_path)), str(val_path))
    by_key, verified = {}, []
    for seed in SEEDS:
        folder, summary, trajectory = read_trajectory(run, seed)
        by_key.update(((row["seed"], row["model"], row["epoch"]), row) for row in trajectory)
        for model in MODELS:
            member, = [m for m in selection[model] if m["seed"] == seed]
            ranked = rank_validation([r for r in trajectory if r["model"] == model])
            verified.append(check_selection(folder, summary, ranked, member))
            best = ranked[0]
            v, = [r for r in val["rows"] if (r["case"], r["seed"], r["checkpoints"])
                  == (model, seed, member["checkpoint"])]
            require(all(number_match(v[m], best[f"val_{m}"]) for m in METRICS), str(val_path))
            observed, = [r for r in test["rows"] if (r["case"], r["seed"], r["checkpoints"])
                         == (model, seed, member["checkpoint"])]
            attach_test(best, observed, test_path)
    for path in historical:
        for saved in read_csv(path):
            row = by_key[(int(saved["seed"]), saved["model"], int(saved["epoch"]))]
            require((saved["split"], saved["dataset_sha256"], saved["weights_key"], saved["checkpoint_sha256"])
                    == ("test", selection["protocol"]["dataset_sha256"], row["weights_key"],
                        digest(Path(row["checkpoint"]))), f"Saved epoch provenance mismatch: {path}")
            attach_test(row, saved, path)
    return list(by_key.values()), verified


def paired_spearman(paired, metric):
    if len(paired) < 3:
        return None, None, "fewer_than_3_epoch_pairs"
    x, y = ([r[f"{split}_{metric}"] for r in paired] for split in ("val", "test"))
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None, None, "constant_ranks"
    correlation = spearmanr(x, y, alternative="two-sided")
    return float(correlation.statistic), float(correlation.pvalue), "descriptive_nominal_pvalue"


def dispersion(values, function):
    return 100 * float(function(values)) if len(values) else None


def summarize(rows):
    result, details = [], []
    for seed in SEEDS:
        for model in MODELS:
            group = [r for r in rows if (r["seed"], r["model"]) == (seed, model)]
            ranked = sorted((r for r in group if r["validation_rank"] is not None),
                            key=lambda row: row["validation_rank"])
            top = ranked[:5]
            details.extend(top)
            for metric in METRICS:
                all_val = [r[f"val_{metric}"] for r in group]
                top_val = [r[f"val_{metric}"] for r in top]
                paired = [r for r in group if r[f"test_{metric}"] is not None]
                n_top = sum(r[f"test_{metric}"] is not None for r in top)
                top_test = [r[f"test_{metric}"] for r in top] if n_top == len(top) == 5 else []
                rho, p, status = paired_spearman(paired, metric)
                result.append(dict(seed=seed, model=model, metric=metric, n_validation=len(group),
                                   n_val_test_pairs=len(paired), spearman_rho=rho, spearman_p_two_sided=p,
                                   spearman_status=status, best_epoch=top[0]["epoch"],
                                   val_at_selected_best=top_val[0], test_at_selected_best=top[0][f"test_{metric}"],
                                   val_all_std_pp=dispersion(all_val, np.std), val_all_range_pp=dispersion(all_val, np.ptp),
                                   top5_val_rank_epochs=" ".join(str(r["epoch"]) for r in top),
                                   top5_val_std_pp=dispersion(top_val, np.std),
                                   top5_val_range_pp=dispersion(top_val, np.ptp),
                                   top5_n_with_historical_test=n_top,
                                   top5_test_std_pp=dispersion(top_test, np.std),
                                   top5_test_range_pp=dispersion(top_test, np.ptp),
                                   top5_test_status="complete" if top_test else "incomplete_historical_coverage"))
    return result, details


def cross_seed(summary):
    result = []
    for model in MODELS:
        for metric in METRICS:
            for split in ("val", "test"):
                values = [r[f"{split}_at_selected_best"] for r in summary
                          if (r["model"], r["metric"]) == (model, metric)]
                count = sum(value is not None for value in values)
                result.append(dict(model=model, metric=metric, split=split, n_seeds=count,
                                   mean=float(np.mean(values)) if count == 3 else None,
                                   std_pp=dispersion(values, np.std) if count == 3 else None))
    return result


def plot_trajectories(out, rows, seed):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True, sharey="row")
    for j, model in enumerate(MODELS):
        group = sorted((r for r in rows if (r["seed"], r["model"]) == (seed, model)),
                       key=lambda row: row["epoch"])
        best, = [r["epoch"] for r in group if r["validation_rank"] == 1]
        for i, (metric, label) in enumerate((("pooled_miou", "Pooled mIoU"), ("pooled_flood_f1", "Flood F1"))):
            ax = axes[i, j]
            ax.plot(EPOCHS, [100*r[f"val_{metric}"] for r in group], label="Validation",
                    marker=".", color="#285285")
            paired = [r for r in group if r[f"test_{metric}"] is not None]
            ax.scatter([row["epoch"] for row in paired], [100*row[f"test_{metric}"] for row in paired],
                       label=f"Historical test ({len(paired)}/20)", color="#b95420", zorder=3)
            ax.axvline(best, color=".5", linestyle="--", label=f"Val-selected epoch {best}")
            ax.set_ylabel(f"{label} (%)")
            ax.grid(axis="y", alpha=.2)
            if i == 0:
                ax.set_title(model.upper())
                ax.legend(fontsize=8)
            else:
                ax.set_xlabel("Epoch")
                ax.set_xticks([1, 5, 10, 15, 20])
    fig.suptitle(f"A3 11ch checkpoint diagnostics — seed {seed}")
    fig.tight_layout()
    fig.savefig(out / f"seed-{seed}.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--historical-epochs", nargs="*", default=[], type=Path,
                        help="Optional existing CSVs with seed/model/epoch/split/dataset_sha256/"
                             "weights_key/checkpoint_sha256 and test metrics")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must be a new directory")
    rows, verified = load_run(args.run.resolve(), args.historical_epochs)
    summary, top = summarize(rows)
    args.out.mkdir(parents=True, exist_ok=False)
    write_csv(args.out / "epoch_metrics.csv", rows)
    write_csv(args.out / "selection_verification.csv", verified)
    write_csv(args.out / "summary.csv", summary)
    write_csv(args.out / "top5.csv", top)
    write_csv(args.out / "best_across_seeds.csv", cross_seed(summary))
    for seed in SEEDS:
        plot_trajectories(args.out, rows, seed)
    paired = sum(row["test_pooled_miou"] is not None for row in rows)
    (args.out / "README.md").write_text(
        "# Checkpoint instability audit\n\n"
        f"Run: {args.run.resolve()}\n\n"
        f"Coverage: {len(rows)} validation records; {paired} paired historical test mIoU records.\n\n"
        "Rank and top five use validation pooled mIoU with earlier-epoch tie breaks. "
        "Blank test values are NOT AVAILABLE. Test top-five ranges and population std require five "
        "historical observations. Spearman is computed separately per metric on at least three "
        "nonconstant paired epochs. The two-sided SciPy p-value is nominal and does not account "
        "for dependence along the training trajectory or selective test availability. "
        "Cross-seed std uses ddof=0. All figures plot historical test observations as points. "
        "This report does not run model inference.\n", encoding="utf-8")
    print(f"{len(rows)} validation records, {paired} historical test pairs: {args.out}", flush=True)


if __name__ == "__main__":
    main()
