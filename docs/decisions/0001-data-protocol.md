---
id: ADR-0001
title: Data and evaluation protocol
status: proposed
level: L3
proposed_by:
  developer: unknown
  agent: OpenAI Codex
  session_id: not available
related_issue: local-task-repository-bootstrap
related_pr: not available
impact:
  - data
  - evaluation
  - configs/data
---

# Data and evaluation protocol

## Context

Official assignments may overlap at event level. Dataset revision and domain definition are unresolved.

## Decision

Preserve official splits, audit event/AoI/parent-tile isolation, propose 0/1/2/255 labels and event-macro flood evaluation per CONTRACTS.md.

## Rationale

Make assumptions reviewable and keep independent implementations scientifically compatible.

## Alternatives Considered

Re-split events only under a separately reviewed protocol; use official AoI evaluation with explicitly narrower claims.

## Consequences

Implementation of research semantics is blocked until approval. No existing results exist to migrate.
Future changes require experiment compatibility review and a superseding ADR when substantive.

## Testing / Verification

Verify revision, license, split metadata, ignore policy, undefined metrics and selection protocol before M1.

## Approval Status

Proposed only. @tungvoi38 must review; no approval or experiment validation is claimed.
