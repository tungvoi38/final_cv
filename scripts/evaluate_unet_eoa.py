#!/usr/bin/env python3
"""Freeze validation-selected A3 members, then evaluate the fixed comparison."""

import argparse
import csv
import hashlib
import json
import statistics
from functools import partial
from pathlib import Path

import torch
from kuro_dataset import (
    CONFIG,
    RAW_SHA256,
    REVISION,
    KuroDataset,
    RawArchive,
    json_hash,
    model_inputs,
)
from torch import nn
from torch.utils.data import DataLoader
from train_unet import (
    FLOOD, N_CLASSES, SAR_PRE_VARIABILITY, append_sar_delta, build_model, seed_everything,
    validate, validate_with_events,
)
from unet_sma import SMA_POLICY, SMA_START_STEP, selected_epoch

METRICS = (
    "pooled_miou",
    "pooled_flood_iou",
    "pooled_flood_f1",
    "event_macro_flood_iou",
    "pooled_flood_precision",
    "pooled_flood_recall",
)
REPO = Path(__file__).resolve().parents[1]
FLIP_TTA_BASELINE = REPO / "configs/experiments/unet_a3_eoa_flip_tta.json"
NO_TTA_BASELINE = REPO / "configs/experiments/unet_a3_eoa_calibrated.json"
PRE_VARIABILITY_CHANNELS = ["delta_VV", "delta_VH", "pre_variability_VV", "pre_variability_VH"]
PRE_VARIABILITY_MEMBER_TRAINING = {
    "epochs": 20, "input_channels": 11,
    "sar_delta": "normalized_post_minus_mean_pre_vv_vh",
    "sar_delta_init": "zero_extra_stem_kernels",
    "sar_pre_variability": SAR_PRE_VARIABILITY,
}


class LogitEnsemble(nn.Module):
    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)
        if not self.members:
            raise ValueError("An ensemble requires at least one model")

    def forward(self, inputs):
        expected = (inputs.shape[0], N_CLASSES, *inputs.shape[-2:])
        total = None
        for model in self.members:
            logits = model(inputs).float()
            if tuple(logits.shape) != expected:
                raise ValueError(
                    f"Misaligned segmentation logits: {tuple(logits.shape)} != {expected}"
                )
            total = logits if total is None else total + logits
        return total / len(self.members)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write("\n")


def evaluation_code_hashes():
    paths = [Path(__file__), REPO / "scripts/train_unet.py", REPO / "scripts/kuro_dataset.py"]
    return {path.name: digest(path) for path in paths}


def check_protocol(protocol):
    identity = (
        protocol["dataset_revision"],
        protocol["dataset_sha256"],
        protocol["preprocess_hash"],
    )
    if identity != (REVISION, RAW_SHA256, json_hash(CONFIG)):
        raise ValueError("Protocol does not match the installed dataset loader")
    order = [f"{time}_{band}" for time in CONFIG["time_order"] for band in CONFIG["band_order"]]
    if protocol["input_order"] != [*order, "DEM"]:
        raise ValueError("Protocol has a different input channel order")
    overrides = protocol.get("member_training_overrides")
    member_order = protocol.get("member_input_order")
    if overrides is not None or member_order is not None:
        if (overrides != PRE_VARIABILITY_MEMBER_TRAINING
                or member_order != [*protocol["input_order"], *PRE_VARIABILITY_CHANNELS]
                or protocol["training"]["input_channels"] != 7):
            raise ValueError("Unexpected pre-variability EoA member protocol")
    if (
        protocol["class_order"] != ["background", "permanent_water", "flood"]
        or protocol["ignore_index"] != 255
    ):
        raise ValueError("Protocol has a different class order or ignore index")
    if protocol["averaging"] != {
        "start_step": SMA_START_STEP,
        "frequency": 1,
        "policy": SMA_POLICY,
    }:
        raise ValueError("Protocol does not match the installed SMA recurrence")


def check_run(run, protocol, *, averaged):
    cfg = load_json(run / "config.json")
    overrides = protocol.get("member_training_overrides", {}) if averaged else {}
    for key, expected in (protocol["training"] | overrides).items():
        if cfg.get(key) != expected:
            raise ValueError(f"{run} differs from protocol for {key}")
    if averaged and (cfg.get("averaging"), cfg.get("sma_policy"), cfg.get("sma_start_step")) != (
        "sma",
        SMA_POLICY,
        SMA_START_STEP,
    ):
        raise ValueError(f"{run} is not a matching SMA run")
    if not averaged and cfg.get("averaging", "none") != "none":
        raise ValueError(f"{run} is not the original ERM reference")
    augmentation = load_json(run / "augmentation.json")
    if any(augmentation.get(key) != expected for key, expected in protocol["augmentation"].items()):
        raise ValueError(f"{run} differs from protocol for augmentation")
    manifest = load_json(run / "meta/manifest.json")
    if (
        manifest["verified_release"],
        manifest["revision"],
        manifest["expected_sha256"],
        manifest["config_hash"],
    ) != (
        True,
        REVISION,
        RAW_SHA256,
        protocol["preprocess_hash"],
    ):
        raise ValueError(f"{run} differs from protocol for dataset provenance")
    summary = load_json(run / "summary.json")
    if summary["completed_epochs"] != cfg["epochs"] or summary["selection"] != "pooled_miou":
        raise ValueError(
            f"{run} must finish the frozen training schedule with pooled_miou selection"
        )
    return cfg, summary, json_hash(manifest["records"])


def freeze_member(run, cfg, summary, *, kind):
    online = kind == "online"
    history = run / ("online_history.csv" if online else "history.csv")
    with history.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if [int(row["epoch"]) for row in rows] != list(range(1, cfg["epochs"] + 1)):
        raise ValueError(f"{history} does not cover the frozen training schedule")
    best = selected_epoch(rows, require_sma=kind == "sma")
    epoch = summary["best_online_epoch" if online else "best_epoch"]
    score = summary["best_online_validation_score" if online else "best_validation_score"]
    if best is None or epoch != int(best["epoch"]) or score != float(best["pooled_miou"]):
        raise ValueError(f"{run} summary disagrees with validation selection")
    path = run / ("best_online.pt" if online else "best.pt")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    weights_key = "online_model" if online else "model"
    metrics_key = "online_validation" if online else "validation"
    if (
        checkpoint["epoch"] != epoch
        or checkpoint["config"] != cfg
        or checkpoint[metrics_key]["pooled_miou"] != score
        or weights_key not in checkpoint
    ):
        raise ValueError(f"{path} disagrees with the validation summary")
    if kind == "sma" and checkpoint["sma"]["count"] != int(best["sma_count"]):
        raise ValueError(f"{path} disagrees with the recorded SMA count")
    return {
        "run_id": cfg["run_id"],
        "seed": cfg["seed"],
        "head": cfg["head"],
        "epoch": epoch,
        "validation_pooled_miou": score,
        "checkpoint": str(path),
        "sha256": digest(path),
        "weights_key": weights_key,
    }


def freeze_selection(protocol, runs, reference):
    check_protocol(protocol)
    cfg, summary, manifest_hash = check_run(reference, protocol, averaged=False)
    if cfg["seed"] != protocol["seeds"][0]:
        raise ValueError("Reference seed does not match the frozen protocol")
    original = freeze_member(reference, cfg, summary, kind="reference")
    by_seed = {}
    source_hashes = set()
    for run in runs:
        cfg, summary, member_manifest = check_run(run, protocol, averaged=True)
        seed = cfg["seed"]
        source_hashes.add((cfg["trainer_sha256"], cfg["sma_sha256"]))
        if seed in by_seed or member_manifest != manifest_hash:
            raise ValueError("Duplicate seed or mismatched dataset manifest")
        members = [freeze_member(run, cfg, summary, kind=kind) for kind in ("online", "sma")]
        if "member_training_overrides" in protocol:
            for member in members:
                member["input_channels"] = cfg["input_channels"]
        by_seed[seed] = members
    if set(by_seed) != set(protocol["seeds"]):
        raise ValueError(f"Expected completed SMA runs for seeds {protocol['seeds']}")
    if len(source_hashes) != 1:
        raise ValueError("SMA runs must use the same trainer and averaging code")
    online, sma = zip(*(by_seed[seed] for seed in protocol["seeds"]), strict=True)
    return {
        "protocol": protocol,
        "manifest_records_hash": manifest_hash,
        "sma_training_code_sha256": list(next(iter(source_hashes))),
        "selection_metric": "validation_pooled_miou",
        "aggregation": "mean_logits_fp32_then_argmax",
        "evaluation_code_sha256": evaluation_code_hashes(),
        "original_erm": original,
        "online": list(online),
        "sma": list(sma),
    }


def verify_frozen(selection):
    check_protocol(selection["protocol"])
    if selection["evaluation_code_sha256"] != evaluation_code_hashes():
        raise ValueError("Evaluation code changed after validation")
    hashes = {}
    for member in [selection["original_erm"], *selection["online"], *selection["sma"]]:
        path = member["checkpoint"]
        if path not in hashes:
            hashes[path] = digest(path)
        if hashes[path] != member["sha256"]:
            raise ValueError(f"Frozen checkpoint changed: {path}")


def load_model(member, device):
    checkpoint = torch.load(member["checkpoint"], map_location="cpu", weights_only=True, mmap=True)
    if checkpoint["epoch"] != member["epoch"]:
        raise ValueError("Frozen checkpoint epoch changed")
    channels = member.get("input_channels", 7)
    config = checkpoint["config"]
    if (channels not in (7, 11) or config["input_channels"] != channels
            or (channels == 11 and config.get("sar_pre_variability") != SAR_PRE_VARIABILITY)):
        raise ValueError("Checkpoint does not match the frozen input layout")
    pre_variability = channels == 11
    model = build_model(member["head"], encoder_weights=None, sar_delta=pre_variability,
                        pre_variability=pre_variability)
    model.load_state_dict(checkpoint[member["weights_key"]], strict=True)
    return model.to(device).eval()


def pre_variability_inputs(batch):
    inputs, temporal = model_inputs(batch)
    return append_sar_delta(inputs, pre_variability=True), temporal


def predict_with_flood_bias(logits, bias):
    other, winner = logits[:, :FLOOD].float().max(dim=1)
    margin = logits[:, FLOOD].float() - other
    # Match the archived sweep: flood loses exact ties to class 0 or 1.
    return torch.where(margin.double() > -bias, FLOOD, winner)


def load_calibrated_selection(calibration, selection_path):
    if digest(selection_path) != calibration["selection_sha256"]:
        raise ValueError("Fixed bias belongs to a different checkpoint selection")
    selection = load_json(selection_path)
    check_protocol(selection["protocol"])
    members = selection["sma"]
    if (selection["selection_metric"] != "validation_pooled_miou"
            or selection["aggregation"] != "mean_logits_fp32_then_argmax"
            or [member["seed"] for member in members] != [42, 43, 44]
            or any((member["input_channels"], member["weights_key"], member["head"])
                   != (11, "model", "a3") for member in members)):
        raise ValueError("Fixed bias requires the three 11-channel A3 SMA checkpoints")
    for member in members:
        if digest(member["checkpoint"]) != member["sha256"]:
            raise ValueError(f"Frozen checkpoint changed: {member['checkpoint']}")
    return selection


def evaluate_calibrated(args, device):
    recipe = load_json(args.calibration)
    selection_path = args.selection or Path(recipe["selection"])
    selection = load_calibrated_selection(recipe, selection_path)
    seed_everything(42)
    settings = selection["protocol"]["training"]
    archive = RawArchive(settings["raw"])
    loader = DataLoader(
        KuroDataset(archive, args.evaluation_split), batch_size=settings["batch_size"],
        num_workers=settings["workers"], shuffle=False, pin_memory=True,
    )
    members = [load_model(member, device) for member in selection["sma"]]
    code = evaluation_code_hashes()
    if "flip_axes" in recipe:
        from evaluate_unet_eoa_flip_tta import FLIPS, FlipTTAEnsemble

        if recipe["flip_axes"] != [list(axes) for axes in FLIPS]:
            raise ValueError("Flip TTA configuration differs from the validation recipe")
        model = FlipTTAEnsemble(members)
        flip_script = Path(__file__).with_name("evaluate_unet_eoa_flip_tta.py")
        code[flip_script.name] = digest(flip_script)
    else:
        model = LogitEnsemble(members)
    bias = recipe["flood_logit_bias"]
    args.out.mkdir(parents=True, exist_ok=False)
    summary, events = validate_with_events(
        model, loader, device, pre_variability_inputs, settings["loss"],
        predict=partial(predict_with_flood_bias, bias=bias),
    )
    archive.check_unchanged()
    metrics = {key: summary[key] for key in METRICS}
    if args.evaluation_split == "validation" and any(
        abs(metrics[key] - recipe["validation_metrics"][key]) > 1e-12 for key in METRICS
    ):
        raise ValueError("Validation metrics differ from the locked calibration result")
    row = {"split": args.evaluation_split, "flip_tta": "flip_axes" in recipe,
           "flood_logit_bias": bias, **metrics}
    write_json(args.out / "metrics.json", {
        **row, "selection_sha256": recipe["selection_sha256"],
        "calibration_sha256": digest(args.calibration),
        "validation_bias_lock_sha256": recipe["validation_bias_lock_sha256"],
        "evaluation_code_sha256": code, "events": events,
    })
    with (args.out / "metrics.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(json.dumps(row, indent=2))
    print(f"Saved: {args.out}")


def evaluate_cases(selection, split, device):
    settings = selection["protocol"]["training"]
    seed_everything(selection["protocol"]["seeds"][0])
    archive = RawArchive(settings["raw"])
    loader = DataLoader(
        KuroDataset(archive, split),
        batch_size=settings["batch_size"],
        shuffle=False,
        num_workers=settings["workers"],
        pin_memory=True,
    )
    rows = []

    def evaluate(case, members, seed=None):
        widths = {member.get("input_channels", 7) for member in members}
        if len(widths) != 1:
            raise ValueError("Ensemble members must share the same input layout")
        input_adapter = pre_variability_inputs if widths == {11} else model_inputs
        models = [load_model(member, device) for member in members]
        model = models[0] if len(models) == 1 else LogitEnsemble(models)
        metrics = validate(model, loader, device, input_adapter, settings["loss"])
        row = {
            "case": case,
            "seed": seed,
            "checkpoints": " | ".join(m["checkpoint"] for m in members),
            **{key: metrics[key] for key in METRICS},
        }
        rows.append(row)
        print(
            f"{split} {case} seed={seed} mIoU={row['pooled_miou']:.5f} "
            f"Flood IoU={row['pooled_flood_iou']:.5f}",
            flush=True,
        )

    evaluate("erm_original", [selection["original_erm"]], selection["original_erm"]["seed"])
    for case in ("online", "sma"):
        for member in selection[case]:
            evaluate(case, [member], member["seed"])
    evaluate("erm_online_ensemble", selection["online"])
    evaluate("eoa", selection["sma"])
    return rows


def write_comparison(directory, split, selection_path, rows):
    table = [
        row.copy()
        for row in rows
        if row["case"]
        in (
            "erm_original",
            "erm_online_ensemble",
            "eoa",
        )
    ]
    for case in ("online", "sma"):
        group = [row for row in rows if row["case"] == case]
        table.append(
            {
                "case": f"{case}_mean_over_seeds",
                "seed": None,
                "checkpoints": " | ".join(row["checkpoints"] for row in group),
                **{key: statistics.mean(row[key] for row in group) for key in METRICS},
            }
        )
    for suffix, source in (("metrics", rows), ("comparison", table)):
        with (directory / f"{split}_{suffix}.csv").open(
            "x", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(source[0]))
            writer.writeheader()
            writer.writerows(source)
    write_json(
        directory / f"{split}_metrics.json",
        {
            "split": split,
            "selection_sha256": digest(selection_path),
            "rows": rows,
            "comparison": table,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="split", required=True)
    val = subparsers.add_parser("validation", help="Freeze members and run validation only")
    val.add_argument("--runs", nargs=3, required=True, type=Path)
    val.add_argument("--protocol", type=Path, default=REPO / "configs/experiments/unet_a3_eoa.json")
    val.add_argument("--reference-run", type=Path)
    val.add_argument("--out", type=Path, required=True)
    test = subparsers.add_parser("test", help="Evaluate the previously frozen selection once")
    test.add_argument("--out", type=Path, required=True)
    calibrated = subparsers.add_parser("calibrated", help="Evaluate the fixed 11-channel EoA")
    calibrated.add_argument("--split", dest="evaluation_split",
                            choices=("validation", "test"), default="validation")
    calibrated.add_argument("--out", type=Path, required=True)
    calibrated.add_argument("--selection", type=Path, help="Relocated copy of the original selection")
    calibrated.add_argument("--no-tta", action="store_true",
                            help="Use the historical calibrated EoA without flips")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    device = torch.device("cuda")
    if args.split == "calibrated":
        args.calibration = NO_TTA_BASELINE if args.no_tta else FLIP_TTA_BASELINE
        evaluate_calibrated(args, device)
        return
    output = args.out.resolve()
    selection_path = output / "selection.json"
    if args.split == "validation":
        protocol = load_json(args.protocol)
        reference = args.reference_run or Path(protocol["reference_run"])
        selection = freeze_selection(
            protocol, [run.resolve() for run in args.runs], reference.resolve()
        )
        output.mkdir(parents=True, exist_ok=False)
        write_json(selection_path, selection)
        target = output
    else:
        selection = load_json(selection_path)
        validation = load_json(output / "validation_metrics.json")
        if validation["split"] != "validation" or validation["selection_sha256"] != digest(
            selection_path
        ):
            raise ValueError("Validation report does not match the frozen selection")
        verify_frozen(selection)
        target = output / "test"
        target.mkdir(exist_ok=False)
    rows = evaluate_cases(selection, args.split, device)
    write_comparison(target, args.split, selection_path, rows)
    print(f"Saved: {target}")


if __name__ == "__main__":
    main()
