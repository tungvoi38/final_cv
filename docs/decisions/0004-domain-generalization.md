---
id: ADR-0004
title: Domain generalization comparisons
status: proposed
level: L3
proposed_by:
  developer: unknown
  agent: OpenAI Codex
  session_id: not available
related_issue: local-task-repository-bootstrap
related_pr: not available
impact:
  - dg
  - training
  - experiments
---

# Domain generalization comparisons

## Context

Unseen-event gains require source-only interventions and comparable evaluation.

## Decision

Propose ClassMix, MixStyle, Deep CORAL and class-conditional CORAL with CE + Dice. Pair source events; record skips, weights and seeds.

## Rationale

Make assumptions reviewable and keep independent implementations scientifically compatible.

## Alternatives Considered

ERM control, independent single-method runs before combinations; alternative covariance estimators require review.

## Consequences

Implementation of research semantics is blocked until approval. No existing results exist to migrate.
Future changes require experiment compatibility review and a superseding ADR when substantive.

## Testing / Verification

Approve insertion sites, pairing, covariance sample minimum, aggregation, loss weights, seed plan and validation-only selection.

## Approval Status

Proposed only. @tungvoi38 must review; no approval or experiment validation is claimed.
