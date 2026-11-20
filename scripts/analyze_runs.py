#!/usr/bin/env python3
"""Generate one reproducible results table and optional data-scaling curve."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs-root', type=Path, default=Path('experiments/unet'))
    ap.add_argument('--out', type=Path, default=Path('reports/unet'))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for cfg_file in sorted(args.runs_root.glob('*/config.json')):
        run = cfg_file.parent
        hp = run / 'history.csv'
        if not hp.exists():
            continue
        cfg = json.loads(cfg_file.read_text())
        hist = pd.read_csv(hp)
        if hist.empty:
            continue
        selection = cfg['selection']
        # Same criterion used in training, and no test-dependent decision.
        best = hist.sort_values('epoch').loc[hist[selection].idxmax()]
        row = {'run_id': cfg['run_id'], 'seed': cfg['seed'],
               'train_samples': cfg['train_samples'], 'completed_epochs': len(hist),
               'best_epoch': int(best.epoch), 'selection_metric': selection,
               'val_event_macro_flood_iou': best['event_macro_flood_iou'],
               'val_pooled_flood_iou': best['pooled_flood_iou'],
               'val_pooled_flood_f1': best['pooled_flood_f1'],
               'val_pooled_flood_precision': best['pooled_flood_precision'],
               'val_pooled_flood_recall': best['pooled_flood_recall'],
               'git_commit': cfg.get('git_commit')}
        # Test metrics are allowed for reporting only, never for selecting the run.
        diag = run / 'diagnostics' / 'bias_report.json'
        if diag.exists():
            d = json.loads(diag.read_text())
            row['test_pooled_flood_iou_baseline'] = d['test_pooled_iou_baseline']
            row['test_pooled_flood_iou_validation_calibrated'] = d['test_pooled_iou_with_validation_bias']
        results.append(row)
    if not results:
        raise SystemExit(f'No completed runs in {args.runs_root}')
    df = pd.DataFrame(results).sort_values('run_id')
    df.to_csv(args.out / 'all_runs_summary.csv', index=False)
    print(df.to_string(index=False))
    if df.train_samples.nunique() >= 2:
        # Multiple seeds/variants at the same size are shown individually.
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for _, row in df.iterrows():
            ax.scatter(row.train_samples, row.val_event_macro_flood_iou)
            ax.annotate(row.run_id, (row.train_samples, row.val_event_macro_flood_iou), fontsize=7)
        group = df.groupby('train_samples')['val_event_macro_flood_iou'].mean().reset_index()
        ax.plot(group.train_samples, group.val_event_macro_flood_iou, '--', alpha=.6)
        ax.set(xlabel='Training patches', ylabel='Best validation event-macro Flood IoU',
               title='Data scaling (compare only runs with matching protocol)')
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(args.out / 'data_scaling_curve.png', dpi=160)
        plt.close(fig)
    print('Reports:', args.out)


if __name__ == '__main__':
    main()
