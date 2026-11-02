---
id: ADR-0002
title: Modality preprocessing
status: proposed
level: L3
proposed_by:
  developer: unknown
  agent: OpenAI Codex
  session_id: not available
related_issue: local-task-repository-bootstrap
related_pr: not available
impact:
  - data/preprocess
  - data/augmentation
  - training/losses
---

# Modality preprocessing

## Context

SAR product, units, cloud policy and normalization provenance need revision-level verification.

## Decision

Propose 4 SAR + 12 optical + 1 DEM channels; float32 aligned inputs; training-only statistics. Chips 256, train stride 128, evaluation stride 256.

## Rationale

Make assumptions reviewable and keep independent implementations scientifically compatible.

## Alternatives Considered

GRD versus RTC; alternative cloud masking and scaling; avoid choosing before provenance audit.

## Consequences

Implementation of research semantics is blocked until approval. No existing results exist to migrate.
Future changes require experiment compatibility review and a superseding ADR when substantive.

## Testing / Verification

Inspect band order, no-data, grid alignment, train-only statistics, CE/Dice ignore behavior and all-ignore handling.

## Approval Status

Proposed only. @tungvoi38 must review; no approval or experiment validation is claimed.
