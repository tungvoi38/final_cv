---
id: ADR-0008
title: Restore the specified A3 training configuration
status: proposed
level: L3
proposed_by:
  developer: unknown
  agent: OpenAI Codex
  session_id: not available
related_issue: A3 reproduction audit
related_pr: not available
impact:
  - scripts/train_unet.py
  - scripts/evaluate_unet.py
  - tests/test_unet_a3.py
---

## Context

The checkpoint from the reproduction identified in the audit
(`a3_original_full_15ep_20261121-181150-6da05338`) stores a 3×3 head. Git commit
`5eb1987` contains a historical server trainer that replaces the SMP head with a
single 1×1 convolution using default PyTorch initialization. That trainer uses
ordinary shuffling and selects by event-macro Flood IoU. The supplied A3
specification instead calls for inverse event-frequency sampling with replacement
and selection by validation pooled mIoU. The successful notebook and its
checkpoint and dependency lock were unavailable.

## Decision

Use `--head a3 --sampling event_balanced --selection pooled_miou` with the full
4,000-patch train split, 15 epochs, seed 42, batch size 2, AdamW, encoder LR
3e-5, decoder and head LR 1e-4, weight decay 0.01 and CE + Dice. Use one single
16-to-3 1×1 convolution. Draw one dataset-length epoch from the train split
using weights equal to the reciprocal of each event's patch count. Seed the
sampler, training loader and validation loader independently from the run seed.

## Rationale

The two changes restore explicit requirements of A3. The historical trainer
provides evidence for the head initialization. Independent generators prevent
validation iteration from advancing the sampling or training worker RNG.

## Alternatives Considered

The existing default 3×3 head and shuffled sampler belong to the previous ERM
configuration. Both defaults remain available for existing callers.

## Consequences

A3 runs record their head and sampler in their configuration and MLflow. Eval
constructs the head recorded by the run and uses the standard head for earlier
runs that lack that field. Exact historical sampling streams and dependency
versions require the successful run artifacts for comparison.

## Testing / Verification

The CPU suite covers input and output shapes, preprocessing and masks, all D4
transforms, weighted replacement, CE + Dice, AMP overflow and the cosine
schedule, and pooled and event metrics.

## Approval Status

Proposed. Research Lead review is pending.
