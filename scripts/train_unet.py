#!/usr/bin/env python3
"""Tunable ERM baseline: train/validation only; never uses the test split."""
import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

N_CLASSES, FLOOD, IGNORE = 3, 2, 255
DECODER_CHANNELS = (256, 128, 64, 32, 16)
EVENT_SAMPLING_EXPONENTS = {'event_balanced': 1.0, 'event_sqrt': 0.5}
SAR_PRE_VARIABILITY = 'absolute_normalized_pre2_minus_pre1_vv_vh'


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_sample_indices(total, count, seed):
    if not 1 <= count <= total:
        raise ValueError(f'--train-samples must be between 1 and {total}')
    indices = list(range(total))
    if count == total:
        return indices
    # All sizes with the same seed use a prefix of the same permutation.
    random.Random(seed).shuffle(indices)
    return sorted(indices[:count])


def append_sar_delta(inputs, *, pre_variability=False):
    """Append normalized post-minus-mean-pre VV and VH to the seven A3 channels."""
    pre_mean = (inputs[:, :2] + inputs[:, 2:4]) * 0.5
    delta = inputs[:, 4:6] - pre_mean
    if pre_variability:
        variability = (inputs[:, 2:4] - inputs[:, :2]).abs()
        return torch.cat((inputs, delta, variability), dim=1)
    return torch.cat((inputs, delta), dim=1)


def build_model(head, *, encoder_weights, sar_delta=False, pre_variability=False):
    import segmentation_models_pytorch as smp

    if head not in ('standard', 'a3'):
        raise ValueError(f'Unknown segmentation head: {head}')
    if sar_delta and head != 'a3':
        raise ValueError('SAR delta requires the A3 head')
    if pre_variability and not sar_delta:
        raise ValueError('SAR pre-event variability requires SAR delta')
    model = smp.Unet('resnet50', encoder_weights=encoder_weights, in_channels=7,
                     classes=N_CLASSES, decoder_channels=DECODER_CHANNELS)
    if head == 'a3':
        # Match the historical trainer, including Conv2d's default initialization.
        model.segmentation_head = torch.nn.Conv2d(DECODER_CHANNELS[-1], N_CLASSES, 1, padding=0)
    if sar_delta:
        encoder = model.encoder
        stem = encoder.conv1
        weight = stem.weight.detach()
        extra_channels = 4 if pre_variability else 2
        extra = torch.zeros_like(weight[:, :extra_channels])
        stem.weight = torch.nn.Parameter(torch.cat((weight, extra), dim=1))
        stem.in_channels = encoder._in_channels = 7 + extra_channels
        encoder._out_channels = [stem.in_channels, *encoder.out_channels[1:]]
    return model


def event_balanced_sampler(event_ids, seed, *, exponent=1.0, draws=None):
    event_ids = list(event_ids)
    counts = Counter(event_ids)
    weights = [1 / counts[event] ** exponent for event in event_ids]
    return WeightedRandomSampler(weights, num_samples=len(weights) if draws is None else draws, replacement=True,
                                 generator=torch.Generator().manual_seed(seed))


def step_optimizer(model, optimizer, scaler, scheduler):
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    stepped = scaler.get_scale() >= scale_before
    if stepped:
        scheduler.step()
    return stepped


def loss_fn(logits, y, mode="ce_dice"):
    valid = y != IGNORE
    if not valid.any():
        return logits.sum() * 0
    logits = logits.float()
    ce = F.cross_entropy(logits, y, ignore_index=IGNORE)
    if mode == 'ce':
        return ce
    p = logits.softmax(dim=1) * valid[:, None]
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
                pooled_flood_recall=recall[FLOOD].item(), pooled_miou=iou.mean().item(),
                pooled_background_f1=f1[0].item(), pooled_permanent_water_f1=f1[1].item(),
                pooled_macro_f1=f1.mean().item())


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
def validate_with_events(model, loader, device, model_inputs, loss_mode="ce_dice",
                         *, report_water_confusion=False, predict=None):
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
        pred = logits.argmax(1) if predict is None else predict(logits)
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
    if report_water_confusion:
        # Confusion-matrix rows are true labels and columns are predictions.
        summary.update(permanent_water_as_flood_pixels=int(pooled[1, FLOOD]),
                       flood_as_permanent_water_pixels=int(pooled[FLOOD, 1]))
    return summary, events


def validate(model, loader, device, model_inputs, loss_mode="ce_dice"):
    summary, _ = validate_with_events(model, loader, device, model_inputs, loss_mode)
    return summary


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, default=Path.cwd())
    ap.add_argument('--raw', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path('experiments/unet'))
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--train-samples', type=int,
                    help='Number of distinct training patches to use (default: all)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--encoder-lr', type=float, default=3e-5)
    ap.add_argument('--decoder-lr', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--optimizer', choices=('adamw', 'adam'), default='adamw')
    ap.add_argument('--loss', choices=('ce_dice', 'ce'), default='ce_dice')
    ap.add_argument('--head', choices=('standard', 'a3'), default='standard')
    ap.add_argument('--sar-delta', action='store_true',
                    help='Append normalized post-minus-mean-pre VV/VH channels (A3 only)')
    ap.add_argument('--sar-pre-variability', action='store_true',
                    help='Append absolute pre-event VV/VH differences (requires --sar-delta)')
    ap.add_argument('--sampling', choices=('shuffle', *EVENT_SAMPLING_EXPONENTS),
                    default='shuffle', help='Sampling strategy for training patches')
    ap.add_argument('--selection', choices=('pooled_flood_iou', 'event_macro_flood_iou', 'pooled_miou'),
                    default='pooled_flood_iou')
    ap.add_argument('--averaging', choices=('none', 'sma'), default='none')
    ap.add_argument('--loeo-fold', type=Path, help='Frozen development-fold manifest for LOEO')
    ap.add_argument('--crossfit-fold', type=Path, help='Train-only CCLA fold with fixed final SMA weights')
    ap.add_argument('--resume', action='store_true', help='Resume a previously saved SMA run')
    args = ap.parse_args()
    import mlflow
    if args.sar_delta and args.head != 'a3':
        ap.error('--sar-delta requires --head a3')
    if args.sar_pre_variability and not args.sar_delta:
        ap.error('--sar-pre-variability requires --head a3 --sar-delta')
    if args.averaging == 'sma' and args.selection != 'pooled_miou':
        ap.error('--averaging sma requires --selection pooled_miou')
    if args.resume and args.averaging != 'sma':
        ap.error('--resume is supported for SMA runs')
    if args.crossfit_fold and args.loeo_fold:
        ap.error('Choose one fold protocol')
    if args.encoder_lr <= 0 or args.decoder_lr <= 0 or args.weight_decay < 0:
        ap.error('learning rates must be positive and weight decay must be nonnegative')
    if args.epochs < 1 or args.batch_size < 1:
        ap.error('epochs and batch-size must be positive')
    if args.train_samples is not None and args.train_samples < 1:
        ap.error('--train-samples must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required')
    if not args.raw.is_file():
        raise FileNotFoundError(args.raw)
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / 'scripts'))
    from kuro_dataset import CONFIG, KuroDataset, RawArchive, model_inputs, write_json

    loader_sha256 = hashlib.sha256((repo / 'scripts/kuro_dataset.py').read_bytes()).hexdigest()

    def unet_inputs(batch):
        inputs, temporal = model_inputs(batch)
        if args.sar_delta:
            inputs = append_sar_delta(inputs, pre_variability=args.sar_pre_variability)
        return inputs, temporal

    seed_everything(args.seed)
    device = torch.device('cuda')
    archive = RawArchive(args.raw)
    train_source = KuroDataset(archive, 'train', augment=True, sar_noise=False)
    val_ds = None if args.crossfit_fold else KuroDataset(archive, 'validation')
    fold = None
    crossfit = None
    if args.crossfit_fold:
        from ccla_crossfit_protocol import SETTINGS, verify_training_fold

        if (any(getattr(args, key) != value for key, value in SETTINGS.items() if key != 'seed')
                or args.seed not in (42, 43, 44) or args.train_samples is not None):
            ap.error('Cross-fitting requires the locked A3 11-channel configuration and seed 42, 43 or 44')
        crossfit = json.loads(args.crossfit_fold.read_text())
        train_source.rows = verify_training_fold(archive.frame, crossfit)
    if args.loeo_fold:
        from loeo_splits import SEEDS, SETTINGS, verified_fold

        if (any(getattr(args, key) != value for key, value in SETTINGS.items())
                or args.train_samples is not None or args.seed not in SEEDS):
            ap.error('LOEO requires the frozen 20-epoch A3 11-channel protocol')
        fold = json.loads(args.loeo_fold.read_text())
        dev = verified_fold(archive.frame, fold).set_index(
            'tortilla:id', drop=False)
        dev.index = dev.index.astype(str)
        train_source.rows = dev.loc[fold['sample_ids']['source_train']].reset_index(drop=True)
        val_ds.rows = dev.loc[fold['sample_ids']['inner_validation']].reset_index(drop=True)
    train_ds = train_source
    sample_indices = None
    if args.train_samples is not None:
        try:
            sample_indices = train_sample_indices(len(train_source), args.train_samples, args.seed)
        except ValueError as exc:
            ap.error(str(exc))
        if len(sample_indices) < len(train_source):
            train_ds = Subset(train_source, sample_indices)
    generator = torch.Generator().manual_seed(args.seed)
    validation_generator = generator
    train_sampler = None
    sampling_exponent = EVENT_SAMPLING_EXPONENTS.get(args.sampling)
    if sampling_exponent is not None:
        rows = train_source.rows if sample_indices is None else train_source.rows.iloc[sample_indices]
        train_sampler = event_balanced_sampler(rows['actid'].astype(str), args.seed,
                                               exponent=sampling_exponent,
                                               draws=(fold or crossfit or {}).get('train_draws_per_epoch'))
        validation_generator = torch.Generator().manual_seed(args.seed)

    loader_options = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    train_dl = DataLoader(train_ds, shuffle=train_sampler is None, sampler=train_sampler,
                          generator=generator, **loader_options)
    val_dl = (DataLoader(val_ds, shuffle=False, generator=validation_generator, **loader_options)
              if val_ds is not None else None)
    if len(train_dl) == 0 or (val_dl is not None and len(val_dl) == 0):
        raise RuntimeError('Train or validation loader is empty')

    run = args.out.resolve() / args.run_id
    if args.resume:
        if (run / 'summary.json').is_file():
            ap.error('Cannot resume a completed run')
    else:
        run.mkdir(parents=True, exist_ok=False)
        (run / 'checkpoints').mkdir()
    git_hash = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, check=False).stdout.strip()
    input_channels = 7 + 2 * (args.sar_delta + args.sar_pre_variability)
    config = dict(run_id=args.run_id, architecture='resnet50_unet', input_channels=input_channels,
                  method='ERM', train_split='train', val_split='validation',
                  raw=str(args.raw.resolve()), train_samples=len(train_ds),
                  val_samples=len(val_ds) if val_ds is not None else 0, epochs=args.epochs, batch_size=args.batch_size,
                  seed=args.seed, workers=args.workers, encoder_lr=args.encoder_lr,
                  decoder_lr=args.decoder_lr, weight_decay=args.weight_decay,
                  optimizer=args.optimizer, loss=args.loss,
                  head=args.head, sampling=args.sampling, sampling_exponent=sampling_exponent,
                  train_draws_per_epoch=len(train_dl.sampler),
                  smp_version=version('segmentation-models-pytorch'),
                  scheduler='cosine_per_step', selection=args.selection,
                  git_commit=git_hash or None, python_version=sys.version.split()[0],
                  torch_version=str(torch.__version__), torch_cuda_version=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0))
    if args.sar_delta:
        config.update(sar_delta='normalized_post_minus_mean_pre_vv_vh',
                      sar_delta_init='zero_extra_stem_kernels')
    if args.sar_pre_variability:
        config['sar_pre_variability'] = SAR_PRE_VARIABILITY
    if not args.resume:
        (run / 'augmentation.json').write_text(
            json.dumps(train_source.augmentation_config, indent=2, default=str), encoding='utf-8')
    if sample_indices is not None:
        sample_ids = train_source.rows.iloc[sample_indices]['tortilla:id'].astype(str).tolist()
        sample_path = run / 'train_sample_ids.json'
        sample_data = (json.dumps(sample_ids, indent=2) + "\n").encode()
        if not args.resume:
            sample_path.write_bytes(sample_data)
        config['train_available_samples'] = len(train_source)
        config['train_sample_ids_sha256'] = hashlib.sha256(sample_data).hexdigest()
    if fold:
        config.update(train_split='loeo_source_train', val_split='loeo_inner_validation',
                      target_event=fold['target_event'],
                      loeo_split_sha256=hashlib.sha256(args.loeo_fold.read_bytes()).hexdigest(),
                      compact_checkpoints=True)
        if not args.resume:
            write_json(run / 'meta/preprocess_config.json', CONFIG)
            write_json(run / 'split_manifest.json', fold)
    elif crossfit:
        config.update(train_split='crossfit_source_train', val_split=None, selection='fixed_epoch',
                      crossfit_fold=crossfit['fold'], crossfit_split_sha256=hashlib.sha256(args.crossfit_fold.read_bytes()).hexdigest())
        if not args.resume:
            write_json(run / 'meta/preprocess_config.json', CONFIG)
            write_json(run / 'split_manifest.json', crossfit)
    elif not args.resume:
        archive.save_metadata(run / 'meta')
    config['dataset_size_bytes'] = args.raw.stat().st_size
    config['loader_sha256'] = loader_sha256
    config['trainer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.averaging == 'sma':
        from unet_sma import SMA_POLICY, SMA_START_STEP, train_sma

        config.update(averaging='sma', sma_start_step=SMA_START_STEP, sma_policy=SMA_POLICY,
                      sma_sha256=hashlib.sha256(Path(__file__).with_name('unet_sma.py').read_bytes()).hexdigest())
    if args.resume:
        if json.loads((run / 'config.json').read_text()) != config:
            ap.error('Resume requires the original run configuration and source hashes')
    else:
        (run / 'config.json').write_text(json.dumps(config, indent=2, default=str), encoding='utf-8')
    tracking = (args.out.resolve() / 'mlflow.db')
    mlflow.set_tracking_uri(f'sqlite:///{tracking}')
    exp_name = 'kuro_siwo_unet_sma' if args.averaging == 'sma' else 'kuro_siwo_unet_erm'
    if fold:
        exp_name = 'kuro_siwo_unet_loeo'
    if crossfit:
        exp_name = 'kuro_siwo_unet_ccla_crossfit'
    if mlflow.get_experiment_by_name(exp_name) is None:
        mlflow.create_experiment(exp_name, artifact_location=(args.out.resolve() / 'mlflow_artifacts').as_uri())
    mlflow.set_experiment(exp_name)

    model = build_model(args.head, encoder_weights=None if args.resume else 'imagenet',
                        sar_delta=args.sar_delta, pre_variability=args.sar_pre_variability).to(device)
    opt_class = torch.optim.AdamW if args.optimizer == 'adamw' else torch.optim.Adam
    optimizer = opt_class([
        {'params': model.encoder.parameters(), 'lr': args.encoder_lr},
        {'params': model.decoder.parameters(), 'lr': args.decoder_lr},
        {'params': model.segmentation_head.parameters(), 'lr': args.decoder_lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(train_dl))
    scaler = torch.amp.GradScaler('cuda')
    def validate_model(current_model):
        return validate_with_events(current_model, val_dl, device, unet_inputs, args.loss,
                                    report_water_confusion=args.sar_delta)

    def train_batches(epoch, average=None):
        model.train()
        losses = []
        for batch in tqdm(train_dl, desc=f'Epoch {epoch}/{args.epochs}'):
            x, y = move(batch, device)
            if not (y != IGNORE).any():
                continue
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                logits = model(unet_inputs(x)[0])
            loss = loss_fn(logits, y, args.loss)
            scaler.scale(loss).backward()
            if step_optimizer(model, optimizer, scaler, scheduler) and average is not None:
                average.update()
            losses.append(float(loss.item()))
        if not losses:
            raise RuntimeError('No valid labeled pixels in this training epoch')
        return float(np.mean(losses))

    if args.averaging == 'sma':
        generators = {'train_loader': generator}
        if val_dl is not None:
            generators['validation_loader'] = validation_generator
        if train_sampler is not None:
            generators['train_sampler'] = train_sampler.generator
        train_sma(run, config, model, train_batches, validate_model if val_dl is not None else None, optimizer, scheduler, scaler,
                  generators, mlflow, resume=args.resume)
        print(f'Done: {run} (fixed-epoch SMA)' if crossfit else f'Done: {run} (validation only)')
        return

    best_score, best_epoch = -1.0, None
    history_path = run / 'history.csv'
    event_path = run / 'validation_events.csv'

    with mlflow.start_run(run_name=args.run_id) as active:
        (run / 'mlflow_run_id.txt').write_text(active.info.run_id)
        mlflow.log_params({k: str(v) for k,v in config.items() if k not in {'raw'}})
        mlflow.log_artifact(str(run / 'config.json'))
        mlflow.log_artifact(str(run / 'augmentation.json'))
        if sample_indices is not None:
            mlflow.log_artifact(str(sample_path))
        for epoch in range(1, args.epochs + 1):
            train_loss = train_batches(epoch)
            metrics, events = validate_model(model)
            row = dict(epoch=epoch, train_loss=train_loss, **metrics)
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
