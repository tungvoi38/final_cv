---
id: ADR-0016
title: Compare independent A3 pre-variability ablations on validation
status: proposed
level: L3
proposed_by:
  developer: unknown
  agent: OpenAI Codex (GPT-6)
  session_id: not available
related_issue: A3 11-channel ablation suite
related_pr: not available
impact:
  - scripts/train_unet.py
  - scripts/unet_ablation.py
  - scripts/fit_unet_ablation_stats.py
  - scripts/evaluate_unet_ablation.py
---

## Context

Historical proposal. The experiment-specific implementation was removed after
the calibrated 11-channel EoA baseline was selected.

Use the frozen 20-epoch, three-seed A3 11-channel SMA EoA as the paired
baseline. Class indices are background 0, permanent water 1, and flood 2.

## Decision

Each training ablation has the same archive, split, sampler, nominal 2,000
batches per epoch, seeds 42/43/44, SMA start step 100, and validation checkpoint
rule. The modes change one factor each.

- `norm_derived` standardizes the four existing derived channels once.
- `weighted_ce` applies weights 1, 1, 1.5 to CE and retains the original Dice.
- `tversky` replaces Dice with Flood Tversky, with smoothing 1. When no flood
  pixel is present, use CE alone. Ignored pixels do not enter either loss term.
- `ndsi` appends two standardized normalized differences from linear power:
  `(mean(pre1, pre2) - post) / (mean(pre1, pre2) + post + 1e-6)`, in VV/VH
  order. Power is recovered from the pinned loader's normalized dB in float64.
  Keep the eleven existing inputs and zero-initialize the extra stem kernels.
- `lr_low` and `lr_high` change only the initial encoder/decoder LR setting.
- `sar_aug` enables the existing Gamma power-domain perturbation with
  probability 0.3 and standard deviation 0.05 on train patches. Derived
  channels are computed from the perturbed normalized SAR.
- `hetero_eoa` uses the available DEM-retaining nine-channel SMA checkpoint
  for seed 42, and the frozen eleven-channel SMA checkpoints of seeds 43/44.
  Each model uses its own compatible input adapter and contributes equally.

Fit the shared feature statistics on valid input pixels of the original
4,000 train patches, once without augmentation. Use population variance.
Keep invalid features at zero and reuse the exact fitted values across seeds.
Validation computes the existing metrics and cross-run validation loss
with the baseline CE+Dice protocol.

## Rationale

The frozen per-seed validation pooled mIoU selection and common validation
labels make the paired differences attributable to the specified experimental
factor under the fixed budget. Aggregate three SMA logits in FP32, then
take the per-pixel argmax. The suite has no test evaluation path.

## Alternatives Considered

Combining loss, channel, or augmentation changes makes it difficult to
attribute the observed differences.

## Consequences

Record the fitted statistics and their hash, checkpoint digests, per-epoch
successful optimizer steps, and validation comparison for reuse. AMP can skip
different updates across ablations despite the fixed training batch budget.

## Testing / Verification

Verify transformations, ignore handling, checkpoint compatibility and
three-member validation input paths on CPU.

## Approval Status

Proposed. Research Lead review is pending.
