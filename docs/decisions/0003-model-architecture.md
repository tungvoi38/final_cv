---
id: ADR-0003
title: Model and fusion interfaces
status: proposed
level: L2
proposed_by:
  developer: unknown
  agent: OpenAI Codex
  session_id: not available
related_issue: local-task-repository-bootstrap
related_pr: not available
impact:
  - models
  - contracts
---

# Model and fusion interfaces

## Context

ResNet and TerraMind have different backbone and modality assumptions.

## Decision

Propose ResNet50-U-Net with ImageNet initialization, early then intermediate fusion; TerraMind-B with explicit modality and alignment adapters.

## Rationale

Make assumptions reviewable and keep independent implementations scientifically compatible.

## Alternatives Considered

Single-modality baselines; separate temporal adapters; defer unsupported backbone combinations.

## Consequences

Implementation of research semantics is blocked until approval. No existing results exist to migrate.
Future changes require experiment compatibility review and a superseding ADR when substantive.

## Testing / Verification

Verify checkpoint license/source and temporal support; test logits and alignment adapters without assuming native temporal SAR.

## Approval Status

Proposed only. @tungvoi38 must review; no approval or experiment validation is claimed.
