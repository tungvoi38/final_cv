---
id: ADR-0014
title: Evaluate SAR pre-event variability in A3 and EoA
status: proposed
level: L2
proposed_by:
  developer: unknown
  agent: OpenAI Codex
  session_id: not available
related_issue: A3 SAR pre-event variability and EoA experiments
related_pr: not available
impact:
  - scripts/train_unet.py
  - scripts/compare_unet_sar_delta.py
  - scripts/evaluate_unet_eoa.py
---

## Context

The paired control is the nine-channel ImageNet A3 SAR-delta SMA run. Two
pre-event observations provide a fixed-order VV/VH variability signal.

## Decision

Append the absolute difference between normalized
pre-event dates for VV and VH, preserving the nine existing stem kernels.
Initialize the two added kernels to zero.
Train seeds 42, 43 and 44 with the same 20-epoch schedule. Select each SMA
checkpoint using validation pooled mIoU. Average aligned logits across models.

## Rationale

The absolute difference exposes pre-event variability independently of the
direction of change. All three ensemble members share channel order, model shape
and class ordering.

## Alternatives Considered

Two pre-event acquisitions are insufficient for a robust per-pixel variance
estimate.

## Consequences

Training, checkpoint selection, and paired comparisons use the same validation
protocol. Evaluation reconstructs the matching 11-channel adapter from the
frozen selection. The historical seven-channel ERM reference is separate.

## Testing / Verification

Verify channel order, zero initialization and strictly load the 11-channel
checkpoint with the frozen model and input configuration on CPU.

## Approval Status

Proposed. Research Lead review is pending.
