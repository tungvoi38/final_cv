#!/usr/bin/env python3
"""Build the frozen A3 11ch ablation from saved artifacts, optionally filling validation gaps."""
import argparse
import csv
from pathlib import Path

import numpy as np

from evaluate_unet_eoa import FLIP_TTA_BASELINE, LogitEnsemble, check_protocol, digest, load_json
from evaluate_unet_eoa_flip_tta import FlipTTAEnsemble, metrics as metrics_for_confusions
from evaluate_unet_vanilla import METRICS, write_csv

AP = "pooled_flood_ap"
STD = "event_flood_iou_std"
CORE = tuple(metric for metric in METRICS if metric not in (AP, STD))
NAMES = ("Vanilla ERM", "Single SMA", "Online Ensemble", "EoA", "EoA + Flip TTA",
         "EoA + Flip TTA + Global Bias 0.95")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def one(rows, **query):
    row, = [row for row in rows if all(row[key] == value for key, value in query.items())]
    return row


def numbers(row):
    return {key: float(row[key]) for key in METRICS if row.get(key) not in (None, "")}


def assert_match(a, b, keys):
    for key in keys:
        if not np.isclose(float(a[key]), float(b[key]), rtol=0, atol=1e-9):
            raise ValueError(f"Artifact mismatch for {key}: {a[key]} != {b[key]}")


def confusion_std(path, bias, saved):
    with np.load(path) as artifact:
        index, = np.flatnonzero(np.isclose(artifact["biases"], bias, rtol=0, atol=1e-12))
        events = {key: artifact[key][[index]] for key in artifact.files if key != "biases"}
    assert_match({k: v[0] for k, v in metrics_for_confusions(events).items()}, saved, CORE)
    cm = np.concatenate(list(events.values()))
    tp = cm[:, 2, 2]
    union = cm[:, 2, :].sum(-1) + cm[:, :, 2].sum(-1) - tp
    return float(np.std(tp[union > 0] / union[union > 0], ddof=0))


def saved_rows(path, expected_split, sha256):
    packet = load_json(path)
    if (packet["split"], packet["selection_sha256"]) != (expected_split, sha256):
        raise ValueError(f"Selection or split mismatch in {path}")
    return packet["rows"]


def record(split, method, members, data, sources, seed=None):
    return dict(split=split, method=method, seed=seed,
                bias=.95 if method == NAMES[5] else 0., flip_tta=method in NAMES[4:],
                run_ids=" | ".join(m["run_id"] for m in members),
                checkpoint_epochs=" | ".join(str(m["epoch"]) for m in members),
                checkpoints=" | ".join(m["checkpoint"] for m in members),
                checkpoint_sha256=" | ".join(m["sha256"] for m in members),
                weights_key=" | ".join(m["weights_key"] for m in members),
                metrics_sources=" | ".join(map(str, sources)), **data)


def saved_metrics(root, selection, selection_path, sha256, vanilla):
    val_path = selection_path.parent / "validation_metrics.json"
    test_path = selection_path.parent / "test/test_metrics.json"
    val = saved_rows(val_path, "validation", sha256)
    test = saved_rows(test_path, "test", sha256)
    vanilla_path = vanilla / "metrics_by_seed.csv"
    vanilla_rows = read_csv(vanilla_path)
    if load_json(vanilla / "frozen_config.json")["reference_selection_sha256"] != sha256:
        raise ValueError("Vanilla ERM artifacts reference a different selection")
    scaling = root / "scaling_sma_diagnostics/seed42-mfXZJj/results"
    scaling_source = load_json(scaling / "protocol.json")["members"]["4000"]
    rank_ap_path = scaling / "test_ap_4000.csv"
    scaling_events_path = scaling / "test_events_4000.csv"
    details = []
    for group, method in zip(("online", "sma"), NAMES, strict=False):
        for member in selection[group]:
            seed = member["seed"]
            old_val = one(val, case=group, seed=seed, checkpoints=member["checkpoint"])
            old_test = one(test, case=group, seed=seed, checkpoints=member["checkpoint"])
            if group == "online":
                raw = one(vanilla_rows, seed=str(seed), epoch=str(member["epoch"]), split="validation")
                if load_json(Path(raw["checkpoint"]).with_name("frozen_config.json"))["source"] != member:
                    raise ValueError(f"Vanilla seed {seed} has a different source checkpoint")
                val_source = vanilla_path
            else:
                val_source = Path(member["checkpoint"]).parent / "history.csv"
                raw = one(read_csv(val_source), epoch=str(member["epoch"]))
            assert_match(raw, old_val, CORE)
            details.append(record("validation", method, [member], numbers(raw), [val_source, val_path], seed))
            data, sources = numbers(old_test), [test_path]
            if group == "sma" and seed == 42 and scaling_source == member:
                pooled = one(read_csv(rank_ap_path), event_id="pooled")
                ious = [float(row["zero_pooled_flood_iou"]) for row in read_csv(scaling_events_path)]
                assert len(ious) == 10
                assert_match({"event_macro_flood_iou": np.mean(ious)}, data, ("event_macro_flood_iou",))
                data.update({AP: float(pooled["average_precision"]), STD: float(np.std(ious, ddof=0))})
                sources += [rank_ap_path, scaling_events_path]
            details.append(record("test_historical", method, [member], data, sources, seed))

    ensemble_rows = []
    for name, case, group in zip(NAMES[2:4], ("erm_online_ensemble", "eoa"), ("online", "sma"), strict=True):
        members = selection[group]
        query = {"case": case, "checkpoints": " | ".join(m["checkpoint"] for m in members)}
        ensemble_rows += [
            record("validation", name, members, numbers(one(val, **query)), [val_path]),
            record("test_historical", name, members, numbers(one(test, **query)), [test_path]),
        ]

    flip = root / "a3-eoa-flip-tta-q8H1dB/results"
    flip_protocol = flip / "protocol.json"
    diag = root / "a3-eoa-flood-diagnostics-YtaAJg/results"
    diagnostics_protocol = diag / "protocol.json"
    for path in (flip_protocol, diagnostics_protocol):
        protocol = load_json(path)
        if protocol["selection_sha256"] != sha256 or protocol["checkpoint_members"] != selection["sma"]:
            raise ValueError(f"Checkpoint mismatch in {path}")
    recipe = load_json(FLIP_TTA_BASELINE)
    if (digest(flip_protocol) != recipe["validation_protocol_sha256"] or
            digest(flip / "bias_selection.json") != recipe["validation_bias_lock_sha256"] or
            load_json(diagnostics_protocol)["recipe_sha256"] != digest(FLIP_TTA_BASELINE)):
        raise ValueError("Flip TTA protocol or bias lock differs from the frozen recipe")

    flip_val = flip / "validation_comparison.csv"
    comparisons = read_csv(flip_val)
    eoa = one(ensemble_rows, split="validation", method=NAMES[3])
    assert_match(one(comparisons, case="eoa", operating_point="zero"), eoa, CORE)
    eoa_confusion = flip / "validation_confusion_eoa.npz"
    eoa[STD] = confusion_std(eoa_confusion, 0., eoa)
    eoa["metrics_sources"] += f" | {eoa_confusion}"

    flip_confusion = flip / "validation_confusion_eoa_flip_tta.npz"
    flip_test = diag / "test_paper_metrics.csv"
    event_table = diag / "test_event_oracle_DIAGNOSTIC_ONLY.csv"
    pr = diag / "test_flood_pr_summary.csv"
    test_ap = float(one(read_csv(pr), scope="pooled")["average_precision"])
    for name, point, bias, iou_key in (
        (NAMES[4], "zero", 0., "flood_iou_zero"),
        (NAMES[5], "validation_optimal", .95, "flood_iou_validation_bias"),
    ):
        data = numbers(one(comparisons, case="eoa_flip_tta", operating_point=point))
        data[STD] = confusion_std(flip_confusion, bias, data)
        ensemble_rows.append(record("validation", name, selection["sma"], data, [flip_val, flip_confusion]))
        test_point = "b_zero" if bias == 0 else "global_bias_selected_on_validation"
        data = numbers(one(read_csv(flip_test), operating_point=test_point))
        ious = [float(row[iou_key]) for row in read_csv(event_table)]
        assert len(ious) == 10
        assert_match({"event_macro_flood_iou": np.mean(ious)}, data, ("event_macro_flood_iou",))
        data.update({AP: test_ap, STD: float(np.std(ious, ddof=0))})
        ensemble_rows.append(record("test_historical", name, selection["sma"], data, [flip_test, event_table, pr]))
    return details, ensemble_rows


def fill_validation(selection, details, rows):
    """Recompute only the missing validation scores on the frozen members."""
    import torch
    from torch.utils.data import DataLoader
    from evaluate_unet_eoa import load_model, seed_everything
    from evaluate_unet_vanilla import evaluate
    from kuro_dataset import KuroDataset, RawArchive

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for evaluation with frozen AMP settings")
    checked = {}
    for member in (*selection["online"], *selection["sma"]):
        path = member["checkpoint"]
        if path not in checked:
            checked[path] = digest(path)
        if checked[path] != member["sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {path}")
    cfg = selection["protocol"]["training"]
    validation = KuroDataset(RawArchive(cfg["raw"]), "validation")
    device = torch.device("cuda")
    loader = DataLoader(validation, batch_size=cfg["batch_size"], num_workers=cfg["workers"],
                        shuffle=False, pin_memory=True)

    def update(model, row):
        result, _ = evaluate(model, loader, device)
        assert_match(result, row, (*CORE, STD) if STD in row else CORE)
        row[AP], row[STD] = result[AP], result[STD]
        row["metrics_sources"] += " | validation inference: shared Vanilla ERM evaluator"
        print(f"{row['method']} seed={row['seed']} AP={row[AP]:.6f}", flush=True)

    seed_everything(42)
    single = [load_model(member, device) for member in selection["sma"]]
    for member, model in zip(selection["sma"], single, strict=True):
        update(model, one(details, split="validation", method=NAMES[1], seed=member["seed"]))
    update(LogitEnsemble(single), one(rows, split="validation", method=NAMES[3]))
    flip = one(rows, split="validation", method=NAMES[4])
    update(FlipTTAEnsemble(single), flip)
    del model, single
    calibrated = one(rows, split="validation", method=NAMES[5])
    calibrated[AP] = flip[AP]
    calibrated["metrics_sources"] += " | same validation AP as Flip TTA, invariant to the fixed bias"
    update(LogitEnsemble([load_model(member, device) for member in selection["online"]]),
           one(rows, split="validation", method=NAMES[2]))


def summarize(details, rows):
    result = []
    for split in ("validation", "test_historical"):
        for method in NAMES:
            if method in NAMES[:2]:
                group = [r for r in details if (r["split"], r["method"]) == (split, method)]
                assert sorted(r["seed"] for r in group) == [42, 43, 44]
            else:
                group = [one(rows, split=split, method=method)]
            output = {key: group[0][key] for key in ("split", "method", "bias", "flip_tta")}
            output["n_members"] = 3
            output["aggregation"] = "seed_mean_ddof0" if len(group) == 3 else "one_ensemble"
            for key in ("run_ids", "checkpoint_epochs", "checkpoints", "checkpoint_sha256", "weights_key",
                        "metrics_sources"):
                output[key] = " | ".join(dict.fromkeys(r[key] for r in group))
            missing = []
            for metric in METRICS:
                values = [r[metric] for r in group if metric in r]
                complete = len(values) == len(group)
                output[metric] = float(np.mean(values)) if complete else None
                output[f"{metric}_seed_std"] = float(np.std(values, ddof=0)) if complete and len(group) == 3 else None
                if not complete:
                    missing.append(metric)
            output["missing_metrics"] = " | ".join(missing)
            result.append(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path(__file__).resolve().parents[2] / "artifacts")
    parser.add_argument("--vanilla", type=Path, help="Directory containing Vanilla ERM metrics_by_seed.csv")
    parser.add_argument("--infer-validation", action="store_true", help="Fill missing AP/std on validation only")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must be a new directory")
    root = args.artifacts.resolve()
    vanilla = args.vanilla or root / "vanilla-erm-validation-vFYh6v/vanilla-erm-vanilla-erm-validation-vFYh6v"
    recipe = load_json(FLIP_TTA_BASELINE)
    selection_path = Path(recipe["selection"])
    sha256 = digest(selection_path)
    if sha256 != recipe["selection_sha256"]:
        raise ValueError("Frozen selection SHA-256 mismatch")
    selection = load_json(selection_path)
    check_protocol(selection["protocol"])
    details, ensembles = saved_metrics(root, selection, selection_path, sha256, vanilla)
    if args.infer_validation:
        fill_validation(selection, details, ensembles)
    summary = summarize(details, ensembles)
    args.out.mkdir(parents=True, exist_ok=False)
    write_csv(args.out / "ablation.csv", summary)
    fields = (*details[0].keys(), "missing_metrics")
    detail_rows = []
    for row in details:
        row["missing_metrics"] = " | ".join(key for key in METRICS if key not in row)
        detail_rows.append({key: row.get(key) for key in fields})
    write_csv(args.out / "seed_details.csv", detail_rows)
    print(args.out / "ablation.csv", flush=True)


if __name__ == "__main__":
    main()
