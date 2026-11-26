---
id: ADR-0009
title: Evaluate square-root event sampling for A3
status: proposed
level: L3
proposed_by:
  developer: unknown
  agent: OpenAI Codex
  session_id: not available
related_issue: A3 performance investigation
related_pr: not available
impact:
  - scripts/train_unet.py
  - tests/test_unet_a3.py
---

## Context

The audited training split contains 4,000 patches from 27 events. Event sizes
range from 2 to 1,542 patches. The A3 sampler gives each event equal probability,
which assigns about 148 draws per epoch even to the smallest event. The A3
trial 005 control reached validation pooled mIoU 0.67757 with seed 42.

## Decision

Make `--sampling event_sqrt` available as an opt-in comparison against
`--sampling event_balanced`. Assign each training patch weight
`1 / sqrt(number of training patches in its event)`. Sample with replacement
using the same seeded sampler and the same number of draws per epoch.
Compare validation pooled mIoU at equal model, optimizer, and epoch settings.
Also record the effect on validation event-level Flood IoU.

## Rationale

The total sampling probability of an event is proportional to the square root
of its size. This reduces repeated draws from very small events while retaining
higher per-patch weights for them. Square-root sampling has been tested for
long-tailed classification. Applying it to events in Kuro Siwo segmentation is
a new hypothesis. [Kang et al., ICLR 2020](https://openreview.net/pdf?id=r1gRTCVFvB)

## Alternatives Considered

Keep equal event mass as the A3 control. Fully uniform patch sampling would
allocate exposure in proportion to the original event sizes. The proposed
exponent is fixed for the comparison.

## Consequences

The smallest event receives about 22 expected draws in the new experiment.
This may help patch diversity and can also reduce attention to rare events.
The A3 defaults and validation/test split behavior remain available.

## Testing / Verification

On training metadata with seed 42, the new sampler yielded 2,215 unique
patches in the first 4,000 draws, compared with 1,574 under equal event mass.
Across 15 simulated epochs the means were 2,200 and 1,540, respectively.
All 27 events appeared in each simulated epoch for both strategies. This
checks sampling behavior. Segmentation performance requires a training run.

## Approval Status

Proposed. Research Lead review is pending.
