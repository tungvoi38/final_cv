#!/usr/bin/env python3
"""Audit or evaluate the three matched 11-channel Vanilla ERM members on validation."""
import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluate_unet_eoa import (
    FLIP_TTA_BASELINE, check_protocol, check_run, digest, evaluation_code_hashes, freeze_member, load_json,
    load_model, pre_variability_inputs, write_json,
)
from evaluate_unet_eoa_diagnostics import pr_curve, rank_pooled
from kuro_dataset import CONFIG, KuroDataset, RawArchive
from train_unet import FLOOD, IGNORE, N_CLASSES, flood_event_scores, move, scores, seed_everything

METRICS = (
    "pooled_miou", "pooled_flood_iou", "pooled_flood_f1", "pooled_flood_precision",
    "pooled_flood_recall", "pooled_flood_ap", "event_macro_flood_iou", "event_flood_iou_std",
)


def audit_members(fresh_root=None):
    recipe = load_json(FLIP_TTA_BASELINE)
    reference_path = Path(recipe["selection"])
    if digest(reference_path) != recipe["selection_sha256"]:
        raise ValueError("Reference selection hash mismatch")
    reference = load_json(reference_path)
    protocol = reference["protocol"]
    check_protocol(protocol)
    if reference["selection_metric"] != "validation_pooled_miou":
        raise ValueError("Expected validation pooled mIoU selection")
    expected = protocol["training"] | protocol["member_training_overrides"]
    if evaluation_code_hashes()["kuro_dataset.py"] != expected["loader_sha256"]:
        raise ValueError("Installed loader differs from the frozen data pipeline")
    fresh_protocol = protocol | {"training": expected}
    members = []
    for original in reference["online"]:
        seed = original["seed"]
        reuse = fresh_root is None
        if reuse:
            run = Path(original["checkpoint"]).parent
        else:
            run = fresh_root / f"{fresh_root.name}-seed{seed}"
        cfg, summary, manifest_hash = check_run(run, fresh_protocol, averaged=reuse)
        if cfg["seed"] != seed or manifest_hash != reference["manifest_records_hash"]:
            raise ValueError(f"{run}: seed or dataset manifest mismatch")
        member = freeze_member(run, cfg, summary, kind="online" if reuse else "reference")
        member["input_channels"] = cfg["input_channels"]
        if reuse and member != original:
            raise ValueError(f"{run}: online checkpoint differs from the frozen selection")
        history = run / ("online_history.csv" if reuse else "history.csv")
        with history.open(newline="", encoding="utf-8") as stream:
            row = next(r for r in csv.DictReader(stream) if int(r["epoch"]) == member["epoch"])
        log = run.parent / f"{cfg['run_id']}.log"
        if not log.is_file() or not log.stat().st_size:
            raise ValueError(f"Missing training log: {log}")
        members.append((member, cfg, row, log))
    if tuple(m[0]["seed"] for m in members) != (42, 43, 44):
        raise ValueError("Expected one checkpoint for each seed 42, 43, 44")
    return reference_path, expected, members


def validation_summary(by_event, score_groups):
    if not by_event:
        raise ValueError("Validation contains no scored pixels")
    events = {event: flood_event_scores(cm) for event, cm in sorted(by_event.items())}
    ious = [r["flood_iou"] for r in events.values() if r["flood_iou"] is not None]
    if not ious:
        raise ValueError("No defined flood IoU on validation events")
    pooled = sum(by_event.values())
    _, _, _, ap = pr_curve(rank_pooled(score_groups))
    return dict(**scores(pooled), pooled_flood_ap=ap, event_macro_flood_iou=float(np.mean(ious)),
                event_flood_iou_std=float(np.std(ious, ddof=0)),
                n_events=len(events), n_scored_events=len(ious)), events


def score_group(logits, target):
    # One-vs-rest flood margin, grouping exact ties for non-interpolated AP.
    margin = logits[:, FLOOD] - logits[:, :FLOOD].amax(dim=1)
    valid = target != IGNORE
    margin = margin[valid]
    if not margin.numel():
        return None
    if not torch.isfinite(margin).all():
        raise ValueError("Nonfinite validation flood margin")
    values, inverse, totals = np.unique(margin.cpu().numpy(), return_inverse=True, return_counts=True)
    positive = (target[valid] == FLOOD).cpu().numpy()
    positives = np.bincount(inverse[positive], minlength=len(values))
    return values, positives, totals


@torch.inference_mode()
def evaluate(model, loader, device):
    by_event, groups = {}, []
    model.eval()
    for batch in loader:
        x, target = move(batch, device)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(pre_variability_inputs(x)[0])
        pred = logits.argmax(dim=1)
        group = score_group(logits.float(), target)
        if group is not None:
            groups.append(group)
        for i, raw_event in enumerate(batch["event_id"]):
            event = str(raw_event.item() if torch.is_tensor(raw_event) else raw_event)
            valid = target[i] != IGNORE
            code = target[i][valid] * N_CLASSES + pred[i][valid]
            cm = torch.bincount(code, minlength=N_CLASSES**2).reshape(N_CLASSES, N_CLASSES).cpu()
            by_event[event] = by_event.get(event, torch.zeros_like(cm)) + cm
    return validation_summary(by_event, groups)


def verify_metrics(metrics, saved):
    for key in (*METRICS, "n_events", "n_scored_events"):
        if key == "pooled_flood_ap":
            continue
        if not math.isclose(metrics[key], float(saved[key]), rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"Recomputed {key}={metrics[key]} differs from checkpoint history ({saved[key]})")


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_members(out, reference_path, expected, members, reuse):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to match the validation AMP protocol")
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    archive = RawArchive(expected["raw"])
    validation = KuroDataset(archive, "validation")
    frozen = {
        "run_id": out.name, "split": "validation", "selection": "validation_pooled_miou",
        "method": "vanilla_erm", "reference_selection": str(reference_path),
        "reference_selection_sha256": digest(reference_path),
        "weights_origin": "reused_online_erm_trajectory" if reuse else "standalone_vanilla_erm_training",
        "training": expected, "preprocessing": CONFIG,
        "evaluation_code_sha256": evaluation_code_hashes(),
        "augmentation": {"d4": True, "sar_noise": False},
        "sma": False, "eoa": False, "online_ensemble": False, "tta": False,
        "flood_logit_bias": 0.0, "dg": False,
        "ap": "pooled non-interpolated AP of float32 flood-minus-max-other logit margin",
        "std_ddof": 0, "evaluation_batch_size": expected["batch_size"],
        "evaluation_amp": True, "evaluator_sha256": digest(__file__),
        "members": [member for member, *_ in members],
    }
    write_json(out / "frozen_config.json", frozen)
    rows = []
    for member, cfg, saved, log in members:
        seed, epoch = member["seed"], member["epoch"]
        run_id = f"{out.name}-seed{seed}"
        seed_everything(seed)
        loader = DataLoader(validation, batch_size=cfg["batch_size"], num_workers=cfg["workers"],
                            pin_memory=True, shuffle=False, generator=torch.Generator().manual_seed(seed))
        model = load_model(member, device)
        metrics, events = evaluate(model, loader, device)
        verify_metrics(metrics, saved)
        seed_out = out / run_id
        seed_out.mkdir()
        seed_config = {key: value for key, value in cfg.items()
                       if not key.startswith("sma_") and key != "averaging"}
        seed_config.update(run_id=run_id, averaging="none", source=member,
                           source_training_log=str(log), source_training_log_sha256=digest(log))
        checkpoint = seed_out / "best.pt"
        torch.save({"model": model.cpu().state_dict(), "epoch": epoch, "config": seed_config,
                    "validation": metrics}, checkpoint)
        del model
        write_json(seed_out / "frozen_config.json", seed_config)
        shutil.copyfile(log, seed_out / "training.log")
        row = dict(run_id=run_id, seed=seed, epoch=epoch, split="validation", checkpoint=str(checkpoint),
                   checkpoint_sha256=digest(checkpoint), **{key: metrics[key] for key in METRICS},
                   n_events=metrics["n_events"], n_scored_events=metrics["n_scored_events"])
        write_csv(seed_out / "metrics.csv", [row])
        write_csv(seed_out / "validation_events.csv",
                  [dict(event_id=event, **values) for event, values in events.items()])
        rows.append(row)
        print(f"seed={seed} epoch={epoch} Val mIoU={metrics['pooled_miou']:.6f} "
              f"Flood AP={metrics['pooled_flood_ap']:.6f}", flush=True)
    write_csv(out / "metrics_by_seed.csv", rows)
    summary = []
    for key in METRICS:
        values = [row[key] for row in rows]
        summary.append(dict(metric=key, mean=float(np.mean(values)), std=float(np.std(values, ddof=0)),
                            n_seeds=len(rows), std_ddof=0))
    write_csv(out / "mean_std.csv", summary)
    print(f"Saved validation metrics: {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "evaluate"):
        child = sub.add_parser(command)
        child.add_argument("--fresh-root", type=Path, help="Use completed standalone training runs")
        if command == "evaluate":
            child.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fresh = args.fresh_root.resolve() if args.fresh_root else None
    reference, expected, members = audit_members(fresh)
    if args.command == "audit":
        print(json.dumps([member for member, *_ in members], indent=2))
    else:
        evaluate_members(args.out.resolve(), reference, expected, members, reuse=fresh is None)


if __name__ == "__main__":
    main()
