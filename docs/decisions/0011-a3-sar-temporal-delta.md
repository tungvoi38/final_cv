---
id: ADR-0011
title: Evaluate temporal SAR difference channels in A3
status: proposed
level: L2
proposed_by:
  developer: unknown
  agent: OpenAI Codex
  session_id: not available
related_issue: A3 temporal SAR input experiment
related_pr: not available
impact:
  - scripts/train_unet.py
  - scripts/compare_unet_sar_delta.py
---

## Context

A3 concatenates three SAR dates and DEM in seven channels. The explicit SAR
change channels test whether a temporal representation helps distinguish
permanent water from flood water.

## Decision

Add opt-in `--sar-delta` for A3. Append `post − (pre1 + pre2)/2` independently
for normalized VV and VH after the existing loader transform. The new stem
preserves the original seven kernels and zero-initializes the two added kernels.
Use the same train and validation splits, training settings and validation
pooled mIoU checkpoint rule as the paired A3 SMA run. Report the two directions
of water-class confusion as validation counts.

## Rationale

The existing normalization is shared across times per polarization. The
differences preserve spatial alignment and expose changes to the first
convolution with the same starting response up to floating-point precision.

## Alternatives Considered

A shared temporal encoder would change the training cost and skip connections.
Changing the pretrained weights together with the input channels would combine
two experimental factors.

## Consequences

Nine-channel checkpoints contain a larger encoder stem. They require the
matching input adapter. The existing seven-channel protocol stays the default.
The paired comparison runs on validation only.

## Testing / Verification

Verify ordering, neutral invalid input, zero-initialized stem kernels, seed and
RNG parity, initial logits, gradient flow and confusion orientation on CPU.

## Approval Status

Proposed. Research Lead review is pending.
