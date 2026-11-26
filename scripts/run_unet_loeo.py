#!/usr/bin/env python3
"""Run the frozen A3 11-channel development-only leave-one-event-out experiment."""
import argparse
import csv
import fcntl
import os
from pathlib import Path
import subprocess
import sys
from time import time_ns

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluate_unet_eoa import digest, freeze_member, load_json, load_model, pre_variability_inputs
from evaluate_unet_eoa_diagnostics import pr_curve, rank_pooled
from evaluate_unet_vanilla import METRICS, score_group
from kuro_dataset import CONFIG, KuroDataset, RawArchive, json_hash, write_json
from loeo_splits import DRAWS, SEEDS, SETTINGS, development, footprint_neighbors, make_fold
from train_unet import FLOOD, IGNORE, N_CLASSES, flood_event_scores, move, scores, seed_everything
from unet_sma import SMA_POLICY, SMA_START_STEP, prune_epoch_checkpoints, save_csv

REPO = Path(__file__).resolve().parents[1]
VARIANTS = tuple(f"{kind}_{seed}" for kind in ("online", "sma") for seed in SEEDS) + (
    "online_ensemble", "eoa",
)


def source_hashes():
    names = ("run_unet_loeo.py", "loeo_splits.py", "train_unet.py", "unet_sma.py", "kuro_dataset.py",
             "evaluate_unet_eoa.py", "evaluate_unet_vanilla.py", "evaluate_unet_eoa_diagnostics.py")
    return {name: digest(REPO / "scripts" / name) for name in names}


def fold_directory(root, target):
    return root / f"target-{target}"


def plan(root, archive, rows, events):
    neighbors = footprint_neighbors(rows)
    folds = [make_fold(rows, events, neighbors, target)
             for target in sorted(events["train"] | events["validation"])]
    frozen = {
        "run_id": root.name, "raw": str(archive.path), "config": CONFIG, "code_sha256": source_hashes(),
        "training": SETTINGS, "seeds": list(SEEDS), "train_draws_per_epoch": DRAWS,
        "sma": {"start_step": SMA_START_STEP, "policy": SMA_POLICY},
        "inner_validation": "original validation events except target, replacement by fixed SHA256 order",
        "checkpoint_selection": "inner_validation_pooled_miou; ties use earliest epoch",
        "evaluation": {"bias": 0, "tta": False, "dg": False, "ensemble": "mean_logits_fp32",
                       "logits": "six raw float16 member arrays [B,3,H,W] per shard; ensemble reconstructible",
                       "ignore_index": IGNORE, "ap": "exact grouped flood-margin ranks within each target"},
        "folds": [{"target_event": fold["target_event"], "manifest_hash": json_hash(fold),
                   **{f"{role}_patches": len(ids) for role, ids in fold["sample_ids"].items()}}
                  for fold in folds],
    }
    for fold in folds:
        write_json(fold_directory(root, fold["target_event"]) / "split_manifest.json", fold)
    save_csv(root / "folds.csv", frozen["folds"])
    write_json(root / "frozen_config.json", frozen)
    print(f"Planned {len(folds)} folds: {root / 'folds.csv'}", flush=True)


def train_seed(archive, folder, fold, seed):
    run_id = f"{folder.parent.name}-{folder.name}-seed{seed}"
    run = folder / run_id
    if (run / "summary.json").is_file():
        print(f"Skip completed training {run_id}", flush=True)
        return run
    resume = run.exists() and any((run / "checkpoints").glob("epoch_*.pt"))
    if run.exists() and not resume:
        interrupted = run.with_name(f"{run_id}.interrupted-{time_ns()}")
        run.rename(interrupted)
    command = [sys.executable, "-u", str(REPO / "scripts/train_unet.py"), "--repo", str(REPO),
               "--raw", str(archive.path), "--out", str(folder), "--run-id", run_id,
               "--seed", str(seed), "--loeo-fold", str(folder / "split_manifest.json")]
    for key, value in SETTINGS.items():
        command.append(f"--{key.replace('_', '-')}")
        if value is not True:
            command.append(str(value))
    if resume:
        command.append("--resume")
    print(f"{'Resume' if resume else 'Train'} {run_id} ({fold['target_event']=})", flush=True)
    with (folder / f"{run_id}.log").open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"{subprocess.list2cmdline(command)}\n")
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as proc:
            for line in proc.stdout:
                log.write(line)
                print(line, end="", flush=True)
            if proc.wait():
                raise subprocess.CalledProcessError(proc.returncode, command)
    return run


def freeze_fold(folder, fold, runs):
    members = {"online": [], "sma": []}
    settings = {key: value for key, value in SETTINGS.items()
                if key not in {"sar_delta", "sar_pre_variability"}}
    for seed, run in zip(SEEDS, runs, strict=True):
        cfg = load_json(run / "config.json")
        summary = load_json(run / "summary.json")
        if (any(cfg.get(key) != value for key, value in settings.items())
                or cfg["seed"] != seed or cfg["target_event"] != fold["target_event"]
                or cfg["loeo_split_sha256"] != digest(folder / "split_manifest.json")
                or cfg["input_channels"] != 11 or cfg["sma_start_step"] != SMA_START_STEP
                or cfg["train_draws_per_epoch"] != DRAWS
                or summary["completed_epochs"] != SETTINGS["epochs"]
                or summary["selection"] != SETTINGS["selection"]
                or load_json(run / "split_manifest.json") != fold):
            raise ValueError(f"Run differs from frozen fold: {run}")
        for kind in members:
            member = freeze_member(run, cfg, summary, kind=kind)
            members[kind].append(dict(member, input_channels=11))
        # A stop immediately after the completion marker can leave redundant epoch files.
        prune_epoch_checkpoints(run)
    frozen = {"target_event": fold["target_event"], "manifest_hash": json_hash(fold),
              "selection": "inner_validation_pooled_miou", **members}
    path = folder / "selection.json"
    if path.is_file():
        if load_json(path) != frozen:
            raise ValueError("Locked selection changed")
    else:
        write_json(path, frozen)
    return frozen


def write_shard(path, arrays):
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


@torch.inference_mode()
def evaluate_target(archive, rows, folder, fold, selection):
    if (load_json(folder / "selection.json") != selection
            or selection["manifest_hash"] != json_hash(fold)
            or any([m["seed"] for m in selection[kind]] != list(SEEDS) for kind in ("online", "sma"))):
        raise ValueError("Target access requires the locked six-checkpoint selection")
    target_ids = fold["sample_ids"]["target"]
    device = torch.device("cuda")
    seed_everything(SEEDS[0])
    target = KuroDataset(archive, "validation")
    indexed = rows.set_index("tortilla:id", drop=False)
    indexed.index = indexed.index.astype(str)
    target.rows = indexed.loc[target_ids].reset_index(drop=True)
    loader = DataLoader(target, batch_size=SETTINGS["batch_size"], num_workers=SETTINGS["workers"],
                        shuffle=False, pin_memory=True)
    members = [(f"{kind}_{member['seed']}", load_model(member, device))
               for kind in ("online", "sma") for member in selection[kind]]
    staging = folder / "target.partial"
    staging.mkdir(exist_ok=True)
    confusions = {name: torch.zeros((3, 3), dtype=torch.int64, device=device) for name in VARIANTS}
    ranks = {name: [] for name in VARIANTS}
    observed = []
    for step, batch in enumerate(loader, 1):
        if set(map(int, batch["event_id"])) != {fold["target_event"]}:
            raise ValueError("Unexpected event in target loader")
        observed.extend(batch["sample_id"])
        x, y = move(batch, device)
        model_inputs = pre_variability_inputs(x)[0]
        with torch.autocast("cuda"):
            raw = {name: model(model_inputs) for name, model in members}
        if any(value.dtype != torch.float16 for value in raw.values()):
            raise ValueError("Expected raw AMP float16 logits (no storage downcast)")
        arrays = {name: value.cpu().numpy() for name, value in raw.items()}
        arrays.update(labels=y.cpu().numpy().astype(np.uint8),
                      valid_mask=(y != IGNORE).cpu().numpy(),
                      input_valid=batch["input_valid"].numpy(),
                      sample_ids=np.asarray(batch["sample_id"]), event_ids=np.asarray(batch["event_id"]))
        write_shard(staging / f"batch-{step:04d}.npz", arrays)
        logits = {name: value.float() for name, value in raw.items()}
        for kind, key in (("online", "online_ensemble"), ("sma", "eoa")):
            a, b, c = [logits[f"{kind}_{seed}"] for seed in SEEDS]
            logits[key] = (a + b + c) / 3
        keep = y != IGNORE
        for key, output in logits.items():
            group = score_group(output, y)
            if group is not None:
                ranks[key].append(group)
            code = y[keep] * N_CLASSES + output.argmax(1)[keep]
            confusions[key] += torch.bincount(code, minlength=9).reshape(3, 3)
        if step % 100 == 0:
            print(f"Target {fold['target_event']}: exported {step}/{len(loader)} batches", flush=True)
    if observed != target_ids:
        raise ValueError("Target sample order differs from the locked manifest")
    records = {}
    for key, cm in confusions.items():
        cm = cm.cpu()
        flood = flood_event_scores(cm)
        ranked = rank_pooled(ranks[key]) if ranks[key] else None
        ap = pr_curve(ranked)[-1] if ranked is not None and ranked[1].sum() else None
        records[key] = {**scores(cm), "pooled_flood_ap": ap,
                        "event_macro_flood_iou": flood["flood_iou"],
                        "event_flood_iou_std": 0. if flood["flood_iou"] is not None else None,
                        "confusion": cm.tolist(), **flood}
    write_json(staging / "metrics.json", {"target_event": fold["target_event"],
                                         "selection_sha256": digest(folder / "selection.json"),
                                         "sample_count": len(observed), "batches": len(loader),
                                         "predictions": records})
    staging.rename(folder / "target")
    print(f"Completed target {fold['target_event']} ({len(observed)} patches)", flush=True)


def report(root, expected_targets):
    records = {key: [] for key in VARIANTS}
    fold_rows = []
    for target in expected_targets:
        folder = fold_directory(root, target)
        path = folder / "target/metrics.json"
        if not path.is_file():
            continue
        packet = load_json(path)
        if (packet["target_event"] != target or packet["selection_sha256"] != digest(folder / "selection.json")
                or set(packet["predictions"]) != set(VARIANTS)
                or len(list(path.parent.glob("batch-*.npz"))) != packet["batches"]):
            raise ValueError(f"Incomplete or mismatched target artifact: {path}")
        for key in VARIANTS:
            metrics = packet["predictions"][key]
            records[key].append(metrics)
            fold_rows.append({"target_event": target, "method": key, "patches": packet["sample_count"],
                              **{metric: metrics[metric] for metric in METRICS}})
    if not fold_rows:
        print("No completed target folds", flush=True)
        return
    save_csv(root / "fold_metrics.csv", fold_rows)
    summary = []
    for key, rows in records.items():
        pooled = scores(torch.tensor([r["confusion"] for r in rows]).sum(dim=0))
        event_iou = [r["flood_iou"] for r in rows if r["flood_iou"] is not None]
        for metric in METRICS:
            values = [r[metric] for r in rows if r[metric] is not None]
            if metric == "event_flood_iou_std":
                values = []  # Across-fold dispersion is computed from individual target IoUs.
            overall = pooled.get(metric)
            if metric == "event_macro_flood_iou" and event_iou:
                overall = float(np.mean(event_iou))
            if metric == "event_flood_iou_std" and event_iou:
                overall = float(np.std(event_iou, ddof=0))
            summary.append({"method": key, "metric": metric, "folds_completed": len(rows),
                            "folds_expected": len(expected_targets), "folds_with_metric": len(event_iou)
                            if metric == "event_flood_iou_std" else len(values), "pooled_or_across_events": overall,
                            "mean_over_target_folds": float(np.mean(values)) if values else None,
                            "std_over_target_folds_ddof0": float(np.std(values, ddof=0)) if values else None})
    save_csv(root / "summary_metrics.csv", summary)
    seed_rows = []
    for kind in ("online", "sma"):
        for metric in METRICS:
            for scope in ("pooled_or_across_events", "mean_over_target_folds"):
                values = [r[scope] for r in summary if r["method"] in {f"{kind}_{s}" for s in SEEDS}
                          and r["metric"] == metric and r[scope] is not None]
                if len(values) == len(SEEDS):
                    seed_rows.append({"method": kind, "metric": metric, "scope": scope,
                                      "folds_completed": len(records[f"{kind}_{SEEDS[0]}"]),
                                      "mean_over_seeds": float(np.mean(values)),
                                      "std_over_seeds_ddof0": float(np.std(values, ddof=0))})
    save_csv(root / "summary_single_models.csv", seed_rows)
    print(f"Summary ({len(fold_rows)//len(VARIANTS)}/{len(expected_targets)} folds): {root}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run", "summarize"):
        command = sub.add_parser(name)
        command.add_argument("--out", type=Path, required=True)
        if name == "plan":
            command.add_argument("--raw", type=Path, required=True)
        elif name == "run":
            select = command.add_mutually_exclusive_group(required=True)
            select.add_argument("--target", type=int)
            select.add_argument("--all", action="store_true")
    args = parser.parse_args()
    root = args.out.resolve()
    if args.command == "plan":
        root.mkdir(parents=True, exist_ok=False)
    with (root / ".loeo.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == "plan":
            archive = RawArchive(args.raw)
            rows, events = development(archive.frame)
            plan(root, archive, rows, events)
            return
        frozen = load_json(root / "frozen_config.json")
        folds = {row["target_event"]: row["manifest_hash"] for row in frozen["folds"]}
        if args.command == "summarize":
            report(root, folds)
            return
        if frozen["code_sha256"] != source_hashes():
            raise ValueError("Code changed after planning; use the original sources or create a new study")
        selected = list(folds) if args.all else [args.target]
        if any(target not in folds for target in selected):
            parser.error("--target must be one of the development events listed in folds.csv")
        pending = [target for target in selected
                   if not (fold_directory(root, target) / "target/metrics.json").is_file()]
        if pending:
            if not torch.cuda.is_available():
                raise RuntimeError("A CUDA GPU is required")
            archive = RawArchive(frozen["raw"])
            rows, _ = development(archive.frame)
            for target in pending:
                folder = fold_directory(root, target)
                fold = load_json(folder / "split_manifest.json")
                if json_hash(fold) != folds[target]:
                    raise ValueError("Fold manifest changed after planning")
                runs = [train_seed(archive, folder, fold, seed) for seed in SEEDS]
                selection = freeze_fold(folder, fold, runs)
                evaluate_target(archive, rows, folder, fold, selection)
                report(root, folds)
        else:
            print("Requested folds are already complete", flush=True)
            report(root, folds)


if __name__ == "__main__":
    main()
