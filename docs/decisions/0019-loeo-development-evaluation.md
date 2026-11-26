---
id: ADR-0019
title: Evaluate the A3 baseline with development-only leave-one-event-out folds
status: proposed
level: L3
proposed_by:
  developer: unknown
  agent: OpenAI Codex
  session_id: not available
related_issue: Development-only leave-one-event-out evaluation
related_pr: not available
impact:
  - scripts/loeo_splits.py
  - scripts/run_unet_loeo.py
  - scripts/train_unet.py
  - scripts/unet_sma.py
---

## Context

The fixed release contains 27 training events, six validation events and ten
original test events. LOEO estimates transfer to development events under
event-disjoint inner selection.

## Decision

Use each development event as the target once. Retain the six original
validation events for inner selection, replacing a held-out validation event
with a source training event chosen by a deterministic hash of event IDs.
Discard source and inner patches whose footprints intersect the target. Also
discard source training patches that intersect retained inner patches.
Keep 4,000 event-sqrt draws per epoch, the frozen 20-epoch A3 hyperparameters
and seeds 42, 43 and 44. Select online and SMA checkpoints separately by
inner-validation pooled mIoU and lock the selection before target inference.

Export logits from all six members with their original AMP float16 values.
Store labels and masks separately within each shard. Reconstruct ensembles
by averaging member logits in float32. Report across-fold AP as the mean
over targets with flood labels.

## Rationale

Retaining the original validation events when possible gives a deterministic
split that uses no held-out labels. The footprint audit excludes geographic
overlap between experimental roles. Fixed draws retain the original optimizer
step budget as the number of source patches changes.

## Alternatives Considered

Holding out random patches for inner validation permits event overlap.
Keeping all epoch checkpoints would exceed available storage for 99 runs.

## Consequences

Results cover the 33 development events. The source data available to each
fold and the number of inner-validation patches vary with the target event.
The official ten test events remain outside the LOEO metrics. Raw scene IDs
are unavailable in the release, so the geographic audit is based on patch
footprints. Resume uses the most recent completed training epoch. Completed
training retains the two best weight sets.

## Testing / Verification

Check all 33 manifests against the pinned archive. Smoke-test target exclusion,
resume against an uninterrupted synthetic run, the real 11-channel source path,
and the held-out export with synthetic predictions.

## Approval Status

Proposed for Research Lead review. Implementation and smoke tests do not
constitute approval of the protocol for publication.
