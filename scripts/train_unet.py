#!/usr/bin/env python3
"""Tunable ERM baseline: train/validation only; never uses the test split."""
import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

N_CLASSES, FLOOD, IGNORE = 3, 2, 255


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def loss_fn(logits, y, mode="ce_dice"):
    valid = y != IGNORE
    if not valid.any():
        return logits.sum() * 0
    ce = F.cross_entropy(logits.float(), y, ignore_index=IGNORE)
    if mode == 'ce':
        return ce
    p = logits.float().softmax(dim=1) * valid[:, None]
    onehot = F.one_hot(y.clamp(max=2).long(), 3).permute(0, 3, 1, 2) * valid[:, None]
    intersection = (p * onehot).sum((0, 2, 3))
    denominator = p.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    dice_loss = 1 - ((2 * intersection + 1) / (denominator + 1)).mean()
    return ce + dice_loss


def move(batch, device):
    x = {k: batch[k].to(device, non_blocking=True) for k in ('sar', 'dem')}
    y = batch['mask'].masked_fill(~batch['valid_mask'].bool(), IGNORE)
    return x, y.to(device, non_blocking=True)


def scores(cm):
    cm = cm.double()
    tp = cm.diag()
    union = cm.sum(0) + cm.sum(1) - tp
    iou = tp / union.clamp(min=1)
    precision = tp / cm.sum(0).clamp(min=1)
    recall = tp / cm.sum(1).clamp(min=1)
    f1 = 2 * tp / (2 * tp + cm.sum(0) - tp + cm.sum(1) - tp).clamp(min=1)
    return dict(pooled_flood_iou=iou[FLOOD].item(), pooled_flood_f1=f1[FLOOD].item(),
                pooled_flood_precision=precision[FLOOD].item(),
                pooled_flood_recall=recall[FLOOD].item(), pooled_miou=iou.mean().item())


def flood_event_scores(cm):
    tp = cm[FLOOD, FLOOD].item()
    fp = cm[:, FLOOD].sum().item() - tp
    fn = cm[FLOOD].sum().item() - tp
    union = tp + fp + fn
    predicted = tp + fp
    actual = tp + fn
    return dict(flood_tp=tp, flood_fp=fp, flood_fn=fn,
                flood_iou=tp / union if union else None,
                flood_precision=tp / predicted if predicted else None,
                flood_recall=tp / actual if actual else None,
                flood_f1=2 * tp / (predicted + actual) if union else None)


@torch.inference_mode()
def validate_with_events(model, loader, device, model_inputs, loss_mode="ce_dice"):
    model.eval()
    by_event = {}
    loss_sum, n_val = 0.0, 0
    for batch in loader:
        x, y = move(batch, device)
        with torch.autocast('cuda'):
            logits = model(model_inputs(x)[0])
        if (y != IGNORE).any():
            loss_sum += loss_fn(logits, y, loss_mode).item() * len(y)
            n_val += len(y)
        pred = logits.argmax(1)
        for i, event in enumerate(batch['event_id']):
            event = str(event.item() if torch.is_tensor(event) else event)
            valid = y[i] != IGNORE
            code = y[i][valid].long() * N_CLASSES + pred[i][valid].long()
            cm = torch.bincount(code, minlength=9).reshape(3, 3).cpu()
            by_event[event] = by_event.get(event, torch.zeros(3, 3, dtype=torch.long)) + cm
    if not by_event:
        raise RuntimeError('Empty validation set')
    pooled = sum(by_event.values())
    events = {event: flood_event_scores(cm) for event, cm in by_event.items()}
    event_ious = [row['flood_iou'] for row in events.values() if row['flood_iou'] is not None]
    if not event_ious:
        raise RuntimeError('No defined flood IoU on validation events')
    summary = dict(**scores(pooled), val_loss=loss_sum / n_val if n_val else float('nan'),
                   event_macro_flood_iou=float(np.mean(event_ious)),
                   event_flood_iou_std=float(np.std(event_ious)),
                   n_events=len(by_event), n_scored_events=len(event_ious))
    return summary, events


def validate(model, loader, device, model_inputs, loss_mode="ce_dice"):
    summary, _ = validate_with_events(model, loader, device, model_inputs, loss_mode)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, default=Path.cwd())
    ap.add_argument('--raw', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path('experiments/unet'))
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--encoder-lr', type=float, default=3e-5)
    ap.add_argument('--decoder-lr', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--optimizer', choices=('adamw', 'adam'), default='adamw')
    ap.add_argument('--loss', choices=('ce_dice', 'ce'), default='ce_dice')
    ap.add_argument('--selection', choices=('pooled_flood_iou', 'event_macro_flood_iou', 'pooled_miou'),
                    default='pooled_flood_iou')
    args = ap.parse_args()
    import mlflow
    if args.encoder_lr <= 0 or args.decoder_lr <= 0 or args.weight_decay < 0:
        ap.error('learning rates must be positive and weight decay must be nonnegative')
    if args.epochs < 1 or args.batch_size < 1:
        ap.error('epochs and batch-size must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required')
    if not args.raw.is_file():
        raise FileNotFoundError(args.raw)
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / 'scripts'))
    from kuro_dataset import RawArchive, KuroDataset, model_inputs
    import segmentation_models_pytorch as smp

    seed_everything(args.seed)
    device = torch.device('cuda')
    archive = RawArchive(args.raw)
    train_ds = KuroDataset(archive, 'train', augment=True, sar_noise=False)
    val_ds = KuroDataset(archive, 'validation')
    generator = torch.Generator().manual_seed(args.seed)

    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.workers, pin_memory=True, generator=generator)

    train_dl, val_dl = loader(train_ds, True), loader(val_ds, False)
    if len(train_dl) == 0 or len(val_dl) == 0:
        raise RuntimeError('Train or validation loader is empty')

    run = args.out.resolve() / args.run_id
    run.mkdir(parents=True, exist_ok=False)
    (run / 'checkpoints').mkdir()
    git_hash = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, check=False).stdout.strip()
    config = dict(run_id=args.run_id, architecture='resnet50_unet', input_channels=7,
                  method='ERM', train_split='train', val_split='validation',
                  raw=str(args.raw.resolve()), train_samples=len(train_ds),
                  val_samples=len(val_ds), epochs=args.epochs, batch_size=args.batch_size,
                  seed=args.seed, workers=args.workers, encoder_lr=3e-5,
                  decoder_lr=1e-4, weight_decay=0.01, loss='CE+Dice',
                  scheduler='cosine_per_step', selection='event_macro_flood_iou',
                  git_commit=git_hash or None, python_version=sys.version.split()[0],
                  torch_version=str(torch.__version__), torch_cuda_version=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0))
    (run / 'augmentation.json').write_text(
        json.dumps(train_ds.augmentation_config, indent=2, default=str), encoding='utf-8')
    archive.save_metadata(run / 'meta')
    config['dataset_size_bytes'] = args.raw.stat().st_size
    config['loader_sha256'] = hashlib.sha256((repo / 'scripts/kuro_dataset.py').read_bytes()).hexdigest()
    config['trainer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (run / 'config.json').write_text(json.dumps(config, indent=2, default=str), encoding='utf-8')
    tracking = (args.out.resolve() / 'mlflow.db')
    mlflow.set_tracking_uri(f'sqlite:///{tracking}')
    exp_name = 'kuro_siwo_unet_erm'
    if mlflow.get_experiment_by_name(exp_name) is None:
        mlflow.create_experiment(exp_name, artifact_location=(args.out.resolve() / 'mlflow_artifacts').as_uri())
    mlflow.set_experiment(exp_name)

    model = smp.Unet('resnet50', encoder_weights='imagenet', in_channels=7,
                     classes=3, decoder_channels=(256, 128, 64, 32, 16)).to(device)
    opt_class = torch.optim.AdamW if args.optimizer == 'adamw' else torch.optim.Adam
    optimizer = opt_class([
        {'params': model.encoder.parameters(), 'lr': args.encoder_lr},
        {'params': model.decoder.parameters(), 'lr': args.decoder_lr},
        {'params': model.segmentation_head.parameters(), 'lr': args.decoder_lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(train_dl))
    scaler = torch.amp.GradScaler('cuda')
    best_score, best_epoch = -1.0, None
    history_path = run / 'history.csv'
    event_path = run / 'validation_events.csv'

    with mlflow.start_run(run_name=args.run_id) as active:
        (run / 'mlflow_run_id.txt').write_text(active.info.run_id)
        mlflow.log_params({k: str(v) for k,v in config.items() if k not in {'raw'}})
        mlflow.log_artifact(str(run / 'config.json'))
        mlflow.log_artifact(str(run / 'augmentation.json'))
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for batch in tqdm(train_dl, desc=f'Epoch {epoch}/{args.epochs}'):
                x, y = move(batch, device)
                if not (y != IGNORE).any():
                    continue
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda'):
                    logits = model(model_inputs(x)[0])
                loss = loss_fn(logits, y, args.loss)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
                losses.append(float(loss.item()))
            if not losses:
                raise RuntimeError('No valid labeled pixels in this training epoch')
            metrics, events = validate_with_events(model, val_dl, device, model_inputs, args.loss)
            row = dict(epoch=epoch, train_loss=float(np.mean(losses)), **metrics)
            with history_path.open('a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if epoch == 1:
                    writer.writeheader()
                writer.writerow(row)
            event_rows = [dict(epoch=epoch, event_id=event, **events[event])
                          for event in sorted(events)]
            with event_path.open('a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=event_rows[0].keys())
                if epoch == 1:
                    writer.writeheader()
                writer.writerows(event_rows)
            mlflow.log_metrics({k: float(v) for k, v in row.items() if k != 'epoch'}, step=epoch)
            checkpoint = {'model': model.state_dict(), 'epoch': epoch,
                          'validation': metrics, 'config': config}
            torch.save(checkpoint, run / 'checkpoints' / f'epoch_{epoch:03}.pt')
            if metrics[args.selection] > best_score:
                best_score = metrics[args.selection]
                best_epoch = epoch
                torch.save(checkpoint, run / 'best.pt')
            print(f'epoch={epoch} val pooled IoU={metrics["pooled_flood_iou"]:.4f} '
                  f'val event-macro IoU={metrics["event_macro_flood_iou"]:.4f} '
                  f'val loss={metrics["val_loss"]:.4f} '
                  f'{args.selection}={metrics[args.selection]:.4f} '
                  f'best epoch={best_epoch}', flush=True)
        (run / 'summary.json').write_text(json.dumps(dict(best_epoch=best_epoch,
            selection=args.selection, best_validation_score=best_score, completed_epochs=args.epochs), indent=2))
        mlflow.log_artifact(str(history_path))
        mlflow.log_artifact(str(event_path))
        mlflow.log_artifact(str(run / 'summary.json'))
        mlflow.log_artifact(str(run / 'best.pt'), artifact_path='checkpoints')
    print(f'Done: {run} (test set was NOT evaluated)')


if __name__ == '__main__':
    main()
