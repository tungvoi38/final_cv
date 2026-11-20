#!/usr/bin/env python3
"""Auditable ResNet50-U-Net baseline for Kuro Siwo.
Requires the project's scripts/kuro_dataset.py interface used by the original notebook.
Train and test evaluation are intentionally separate commands.
"""

import argparse
import csv
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

NC, FLOOD, IGNORE = 3, 2, 255


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def import_dataset(repo):
    sys.path.insert(0, str(repo / "scripts"))
    from kuro_dataset import KuroDataset, RawArchive, model_inputs

    return RawArchive, KuroDataset, model_inputs


def model_factory(model_inputs):
    import segmentation_models_pytorch as smp

    class UNetR50(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = smp.Unet(
                encoder_name="resnet50",
                encoder_weights="imagenet",
                in_channels=7,
                classes=NC,
                decoder_channels=(256, 128, 64, 32, 16),
            )
            # Preserve the original notebook's explicitly assigned head.
            self.net.segmentation_head = nn.Conv2d(16, NC, kernel_size=1)

        def forward(self, batch):
            x, _ = model_inputs(batch)
            if x.shape[1] != 7:
                raise ValueError(f"Expected 7 model input channels, got {x.shape[1]}")
            return self.net(x)

    return UNetR50()


def criterion(logits, target):
    """Match original notebook: CE + three-class soft Dice with ignore=255."""
    logits = logits.float()
    valid = target != IGNORE
    if not valid.any():
        return logits.sum() * 0.0
    ce = F.cross_entropy(logits, target, ignore_index=IGNORE)
    mask = valid[:, None]
    probs = logits.softmax(1) * mask
    one_hot = F.one_hot(target.clamp(max=NC - 1).long(), NC).permute(0, 3, 1, 2) * mask
    intersect = (probs * one_hot).sum((0, 2, 3))
    denom = probs.sum((0, 2, 3)) + one_hot.sum((0, 2, 3))
    return ce + 1 - ((2 * intersect + 1) / (denom + 1)).mean()


def move_batch(batch, device):
    inputs = {k: batch[k].to(device, non_blocking=True) for k in ("sar", "dem")}
    target = batch["mask"].masked_fill(~batch["valid_mask"].bool(), IGNORE)
    return inputs, target.to(device, non_blocking=True)


def cm_to_metrics(cm):
    cm = cm.double()
    tp = cm.diag()
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    iou = tp / (tp + fp + fn).clamp(min=1)
    precision = tp / (tp + fp).clamp(min=1)
    recall = tp / (tp + fn).clamp(min=1)
    f1 = 2 * tp / (2 * tp + fp + fn).clamp(min=1)
    return dict(
        miou=iou.mean().item(),
        iou_nowater=iou[0].item(),
        iou_permwater=iou[1].item(),
        iou_flood=iou[FLOOD].item(),
        precision_flood=precision[FLOOD].item(),
        recall_flood=recall[FLOOD].item(),
        f1_flood=f1[FLOOD].item(),
    )


def event_key(event):
    return str(event.item() if torch.is_tensor(event) else event)


@torch.inference_mode()
def evaluate(model, dataloader, device, use_amp):
    model.eval()
    by_event = {}
    running_loss = 0.0
    valid_samples = 0
    for batch in tqdm(dataloader, desc="Evaluate", leave=False):
        inputs, y = move_batch(batch, device)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(inputs)
        valid = y != IGNORE
        if valid.any():
            running_loss += criterion(logits, y).item() * y.shape[0]
            valid_samples += y.shape[0]
        pred = logits.argmax(1)
        for j, event in enumerate(batch["event_id"]):
            k = event_key(event)
            codes = y[j][valid[j]].long() * NC + pred[j][valid[j]].long()
            cm = torch.bincount(codes, minlength=NC * NC).reshape(NC, NC).cpu()
            by_event[k] = by_event.get(k, torch.zeros(NC, NC, dtype=torch.int64)) + cm
    if not by_event:
        raise RuntimeError("Evaluation split is empty")
    pooled = sum(by_event.values())
    rows = []
    for event, cm in sorted(by_event.items()):
        met = cm_to_metrics(cm)
        tp = int(cm[FLOOD, FLOOD].item())
        fp = int(cm[:, FLOOD].sum().item()) - tp
        fn = int(cm[FLOOD, :].sum().item()) - tp
        rows.append(dict(event_id=event, tp=tp, fp=fp, fn=fn, **met))
    out = cm_to_metrics(pooled)
    scored = [r for r in rows if (r["tp"] + r["fp"] + r["fn"]) > 0]
    out.update(
        loss=(running_loss / valid_samples if valid_samples else None),
        event_iou_flood_mean=(float(np.mean([r["iou_flood"] for r in scored])) if scored else None),
        event_f1_flood_mean=(float(np.mean([r["f1_flood"] for r in scored])) if scored else None),
        event_iou_flood_std=(float(np.std([r["iou_flood"] for r in scored])) if scored else None),
        event_iou_flood_worst=(float(np.min([r["iou_flood"] for r in scored])) if scored else None),
        n_events=len(rows),
        n_scored_events=len(scored),
    )
    return out, rows, pooled.numpy().tolist()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dataloader(dataset, batch_size, workers, shuffle, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        generator=g,
    )


def training(args):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU/CUDA unavailable; check nvidia-smi and PyTorch installation")
    if not (args.repo / "scripts" / "kuro_dataset.py").exists():
        raise FileNotFoundError(f"Missing {args.repo / 'scripts/kuro_dataset.py'}")
    if not args.raw.is_file():
        raise FileNotFoundError(f"Missing dataset: {args.raw}")
    seed_all(args.seed)
    RawArchive, KuroDataset, model_inputs = import_dataset(args.repo)
    device = torch.device("cuda")
    run_dir = args.output / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)  # prevent accidental overwrite
    config = dict(
        run_id=args.run_id,
        model="resnet50_unet",
        method="ERM",
        train_split="train",
        validation_split="validation",
        selection_metric="event_iou_flood_mean",
        epochs=args.epochs,
        batch_size=args.batch_size,
        encoder_lr=args.encoder_lr,
        decoder_lr=args.decoder_lr,
        weight_decay=0.01,
        seed=args.seed,
        augment=True,
        sar_noise=False,
        loss="CE + soft Dice 3 classes, smoothing 1",
        scheduler="cosine per optimizer step",
        grad_clip=1.0,
        raw_dataset=str(args.raw.resolve()),
        repo=str(args.repo.resolve()),
        gpu=torch.cuda.get_device_name(0),
        torch_version=torch.__version__,
        started_at=datetime.now(UTC).isoformat(),
    )
    write_json(run_dir / "config.json", config)
    archive = RawArchive(args.raw)
    archive.save_metadata(run_dir / "meta")
    train_ds = KuroDataset(archive, "train", augment=True, sar_noise=False)
    val_ds = KuroDataset(archive, "validation")
    write_json(run_dir / "augmentation_config.json", train_ds.augmentation_config)
    train_dl = dataloader(train_ds, args.batch_size, args.workers, True, args.seed)
    val_dl = dataloader(val_ds, args.batch_size, args.workers, False, args.seed)
    model = model_factory(model_inputs).to(device)
    opt = torch.optim.AdamW(
        [
            {"params": model.net.encoder.parameters(), "lr": args.encoder_lr},
            {"params": model.net.decoder.parameters(), "lr": args.decoder_lr},
            {"params": model.net.segmentation_head.parameters(), "lr": args.decoder_lr},
        ],
        weight_decay=0.01,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * len(train_dl))
    scaler = torch.amp.GradScaler("cuda")
    best = -1.0
    hist = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_dl, desc=f"Train {epoch}/{args.epochs}"):
            x, y = move_batch(batch, device)
            if not (y != IGNORE).any():
                continue
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                pred = model(x)
            loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            old_scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                sched.step()
            losses.append(loss.item())
        if not losses:
            raise RuntimeError("No valid training pixels this epoch")
        metrics, event_rows, pooled_cm = evaluate(model, val_dl, device, True)
        row = dict(
            epoch=epoch,
            train_loss=float(np.mean(losses)),
            **{"val_" + k: v for k, v in metrics.items()},
        )
        hist.append(row)
        write_csv(run_dir / "history.csv", hist)
        ckpt = dict(
            model=model.state_dict(), epoch=epoch, config=config, validation_metrics=metrics
        )
        checkpoints = run_dir / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        torch.save(ckpt, checkpoints / f"epoch_{epoch:03d}.pt")
        score = metrics["event_iou_flood_mean"]
        if score is not None and score > best:
            best = score
            torch.save(ckpt, run_dir / "best.pt")
            write_csv(run_dir / "best_validation_events.csv", event_rows)
            write_json(run_dir / "best_validation_confusion_matrix.json", pooled_cm)
            write_json(run_dir / "best_validation_metrics.json", metrics)
        print(
            f"Epoch {epoch}: val pooled Flood IoU={metrics['iou_flood']:.4f}; "
            f"val event-macro Flood IoU={score if score is not None else float('nan'):.4f}; "
            f"best event-macro={best:.4f}",
            flush=True,
        )
    write_json(
        run_dir / "run_summary.json",
        dict(
            best_selection_score=best,
            best_checkpoint="best.pt",
            completed_epochs=args.epochs,
            test_evaluated=False,
        ),
    )
    print(f"Run complete: {run_dir}\nTest was NOT evaluated. Use --mode test separately.")


def test_only(args):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU/CUDA unavailable")
    run_dir = args.output / args.run_id
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (run_dir / "test_metrics.json").exists():
        raise FileExistsError("Test results already exist; refusing to overwrite")
    RawArchive, KuroDataset, model_inputs = import_dataset(args.repo)
    archive = RawArchive(Path(config["raw_dataset"]))
    seed_all(int(config["seed"]))
    dl = dataloader(
        KuroDataset(archive, "test"),
        int(config["batch_size"]),
        args.workers,
        False,
        int(config["seed"]),
    )
    device = torch.device("cuda")
    model = model_factory(model_inputs).to(device)
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    metrics, event_rows, cm = evaluate(model, dl, device, True)
    write_json(run_dir / "test_metrics.json", dict(checkpoint_epoch=checkpoint["epoch"], **metrics))
    write_json(run_dir / "test_confusion_matrix.json", cm)
    write_csv(run_dir / "test_events.csv", event_rows)
    print("Test metrics (checkpoint selected on validation):", json.dumps(metrics, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "test"], default="train")
    p.add_argument("--repo", type=Path, default=Path.cwd())
    p.add_argument("--raw", type=Path, help="Path to geobench_kuro_siwo.tortilla")
    p.add_argument("--output", type=Path, default=Path("experiments/unet"))
    p.add_argument("--run-id", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--encoder-lr", type=float, default=3e-5)
    p.add_argument("--decoder-lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    args.repo = args.repo.resolve()
    args.output = args.output.resolve()
    if args.mode == "train":
        if args.raw is None:
            p.error("--raw is required for --mode train")
        training(args)
    else:
        test_only(args)


if __name__ == "__main__":
    main()
