#!/usr/bin/env python3
"""Offline diagnostics ONLY. Never use test/oracle results to select models."""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from train_unet import FLOOD, IGNORE, move, validate, seed_everything


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=lambda x: float(x)), encoding='utf-8')


def save_plot(path, x, curves, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, y in curves.items():
        ax.plot(x, y, label=label, marker='.' if len(x) < 35 else None)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def model_load(model, path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck['model'])
    return ck


@torch.inference_mode()
def margin_histograms(model, dl, model_inputs, device, biases):
    """Streaming margin histogram; never holds all pixel logits in RAM.

    Flood prediction iff bias > (max(other logits) - flood logit).
    All non-flood decisions remain as original argmax. Candidate grid is finite.
    """
    by_event = defaultdict(lambda: [np.zeros(len(biases)+1, dtype=np.int64),
                                    np.zeros(len(biases)+1, dtype=np.int64)])
    thresh = torch.tensor(biases, device=device, dtype=torch.float32)
    model.eval()
    for batch in dl:
        x, y = move(batch, device)
        with torch.autocast('cuda'):
            z = model(model_inputs(x)[0]).float()
        delta = torch.maximum(z[:, 0], z[:, 1]) - z[:, FLOOD]
        for i, event in enumerate(batch['event_id']):
            event = str(event.item() if torch.is_tensor(event) else event)
            good = y[i] != IGNORE
            indexes = torch.bucketize(delta[i][good].contiguous(), thresh, right=True)
            positive = y[i][good] == FLOOD
            by_event[event][0][:] += torch.bincount(indexes[positive], minlength=len(biases)+1).cpu().numpy()
            by_event[event][1][:] += torch.bincount(indexes[~positive], minlength=len(biases)+1).cpu().numpy()
    return by_event


def score_curve(hists, biases):
    pos = sum((a for a, _ in hists), np.zeros(len(biases)+1, dtype=np.int64))
    neg = sum((b for _, b in hists), np.zeros(len(biases)+1, dtype=np.int64))
    # For threshold biases[j], eligible bucket indexes <=j; bucket j contains
    # margins below that threshold; exact equal-margin points fall in j+1.
    tp = np.cumsum(pos)[:-1]
    fp = np.cumsum(neg)[:-1]
    fn = pos.sum() - tp
    p = np.divide(tp, tp+fp, out=np.ones_like(tp, dtype=float), where=(tp+fp)>0)
    r = np.divide(tp, pos.sum(), out=np.zeros_like(tp, dtype=float), where=pos.sum()>0)
    iou = np.divide(tp, tp+fp+fn, out=np.full(len(tp), np.nan), where=(tp+fp+fn)>0)
    f1 = np.divide(2*tp, 2*tp+fp+fn, out=np.full(len(tp), np.nan), where=(2*tp+fp+fn)>0)
    return pd.DataFrame({'bias': biases, 'precision': p, 'recall': r, 'iou': iou, 'f1': f1,
                         'tp': tp, 'fp': fp, 'fn': fn})


def nearest(df, bias):
    return df.iloc[int(np.abs(df.bias.to_numpy() - bias).argmin())]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path.cwd())
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--raw', type=Path, default=None)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--bias-min', type=float, default=-8.)
    p.add_argument('--bias-max', type=float, default=8.)
    p.add_argument('--bias-step', type=float, default=.05)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required')
    run = args.run.resolve()
    cfg = json.loads((run / 'config.json').read_text())
    raw = args.raw or Path(cfg['raw'])
    if not raw.is_file():
        raise FileNotFoundError(raw)
    if args.bias_min >= args.bias_max or args.bias_step <= 0:
        p.error('invalid bias grid')
    biases = np.round(np.arange(args.bias_min, args.bias_max + args.bias_step/2,
                                args.bias_step), 6)
    sys.path.insert(0, str(args.repo.resolve() / 'scripts'))
    from kuro_dataset import RawArchive, KuroDataset, model_inputs
    import segmentation_models_pytorch as smp
    seed_everything(int(cfg['seed']))
    device = torch.device('cuda')
    archive = RawArchive(raw)
    bs = int(cfg['batch_size'])
    def dl(split):
        return DataLoader(KuroDataset(archive, split), batch_size=bs, shuffle=False,
                          num_workers=args.workers, pin_memory=True)
    val_dl, test_dl = dl('validation'), dl('test')
    model = smp.Unet('resnet50', encoder_weights=None, in_channels=7,
                     classes=3, decoder_channels=(256, 128, 64, 32, 16)).to(device)
    output = run / 'diagnostics'
    output.mkdir(exist_ok=True)

    history = pd.read_csv(run / 'history.csv')
    history = history.sort_values('epoch')
    test_rows = []
    # Required by supervisor: retrospective per-epoch TEST curve, diagnostic only.
    for epoch in history.epoch.astype(int):
        path = run / 'checkpoints' / f'epoch_{epoch:03}.pt'
        if not path.exists():
            raise FileNotFoundError(f'Missing checkpoint {path}')
        model_load(model, path, device)
        metrics = validate(model, test_dl, device, model_inputs, cfg.get('loss', 'ce_dice'))
        test_rows.append({'epoch': epoch, **metrics})
    test_history = pd.DataFrame(test_rows)
    test_history.to_csv(output / 'test_by_epoch_DIAGNOSTIC_ONLY.csv', index=False)
    combined = history[['epoch', 'pooled_flood_iou']].merge(
        test_history[['epoch', 'pooled_flood_iou']], on='epoch', suffixes=('_val', '_test'))
    save_plot(output / 'checkpoint_curve_DIAGNOSTIC_ONLY.png', combined.epoch,
              {'Validation pooled Flood IoU': combined.pooled_flood_iou_val,
               'Test pooled Flood IoU (diagnostic)': combined.pooled_flood_iou_test},
              'Epoch', 'Pooled Flood IoU')
    top3 = history.nlargest(3, cfg['selection']).merge(
        test_history, on='epoch', suffixes=('_val', '_test'))
    top3.to_csv(output / 'top3_val_selected_test_DIAGNOSTIC_ONLY.csv', index=False)

    ck = model_load(model, run / 'best.pt', device)
    val_hist = margin_histograms(model, val_dl, model_inputs, device, biases)
    test_hist = margin_histograms(model, test_dl, model_inputs, device, biases)
    val_curve = score_curve(list(val_hist.values()), biases)
    test_curve = score_curve(list(test_hist.values()), biases)
    val_curve.to_csv(output / 'validation_bias_sweep.csv', index=False)
    test_curve.to_csv(output / 'test_pooled_bias_sweep_DIAGNOSTIC_ONLY.csv', index=False)
    if val_curve.iou.notna().sum() == 0:
        raise RuntimeError('No valid Flood IoU on validation')
    selected = float(val_curve.loc[val_curve.iou.idxmax(), 'bias'])
    event_rows = []
    for event, hist in sorted(test_hist.items()):
        curve = score_curve([hist], biases)
        curve.to_csv(output / f'pr_event_{event.replace("/", "_")}.csv', index=False)
        good = curve.dropna(subset=['iou'])
        if good.empty:
            oracle = float('nan')
            oracle_iou = float('nan')
        else:
            oracle = float(good.loc[good.iou.idxmax(), 'bias'])
            oracle_iou = float(nearest(curve, oracle).iou)
        event_rows.append({'event_id': event,
                           'flood_prevalence': int(hist[0].sum()) / max(1, int(hist[0].sum()+hist[1].sum())),
                           'baseline_iou': float(nearest(curve, 0).iou),
                           'validation_bias_iou': float(nearest(curve, selected).iou),
                           'oracle_bias_DIAGNOSTIC_ONLY': oracle,
                           'oracle_iou_DIAGNOSTIC_ONLY': oracle_iou})
    ev = pd.DataFrame(event_rows)
    ev.to_csv(output / 'event_bias_analysis_DIAGNOSTIC_ONLY.csv', index=False)
    pooled_pr = test_curve[['recall', 'precision', 'bias']]
    pooled_pr.to_csv(output / 'pr_pooled.csv', index=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(pooled_pr.recall, pooled_pr.precision, label='Pooled', linewidth=2)
    for event, hist in sorted(test_hist.items()):
        curve = score_curve([hist], biases)
        ax.plot(curve.recall, curve.precision, alpha=.5, label=event)
    ax.set(xlabel='Recall', ylabel='Precision', title='Flood PR (bias sweep)')
    ax.grid(alpha=.25); ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(output / 'flood_pr_curves.png', dpi=160); plt.close(fig)
    save_plot(output / 'bias_vs_iou.png', biases,
              {'Validation IoU': val_curve.iou, 'Test pooled IoU (diagnostic)': test_curve.iou},
              'Flood logit bias', 'Flood IoU')
    report = {'best_checkpoint_epoch_selected_on_validation': int(ck['epoch']),
              'selection_metric': cfg['selection'], 'validation_selected_bias': selected,
              'test_pooled_iou_baseline': float(nearest(test_curve, 0).iou),
              'test_pooled_iou_with_validation_bias': float(nearest(test_curve, selected).iou),
              'test_event_iou_std_baseline': float(ev.baseline_iou.std(ddof=0)),
              'test_event_iou_std_with_validation_bias': float(ev.validation_bias_iou.std(ddof=0)),
              'test_event_iou_std_oracle_DIAGNOSTIC_ONLY': float(ev.oracle_iou_DIAGNOSTIC_ONLY.std(ddof=0)),
              'test_oracle_bias_std_DIAGNOSTIC_ONLY': float(ev.oracle_bias_DIAGNOSTIC_ONLY.std(ddof=0)),
              'bias_search_range': [float(biases[0]), float(biases[-1])],
              'caveat': 'Oracle and per-epoch test evaluations are diagnostic only. Never tune or select using test.'}
    save_json(output / 'bias_report.json', report)
    print(json.dumps(report, indent=2))
    print('Saved diagnostics:', output)


if __name__ == '__main__':
    main()
