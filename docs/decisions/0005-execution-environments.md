---
id: ADR-0005
title: Local and Colab runtime boundaries
status: proposed
level: L2
proposed_by:
  developer: unknown
  agent: OpenAI Codex
  session_id: not available
related_issue: local-task-colab-extension
related_pr: not available
impact:
  - training/runtime
  - training/checkpoints
  - configs/environments
  - notebooks
---

# Local and Colab runtime boundaries

## Context

The user requested Colab GPU support without model implementation or notebook
training logic. Colab sessions and local storage are ephemeral.

## Decision

Keep one package/CLI pipeline. Environment YAML controls paths, device, precision,
checkpoint cadence, resume and tracking. The notebook only sets up the environment,
mounts Drive, checks out an immutable commit and invokes the CLI. Checkpoints store
full state at epoch boundaries. Resuming requires matching research provenance.

## Rationale

Avoid drift between local and notebook research logic and preserve reproducibility
when sessions disconnect. Environment choice does not approve research methodology.

## Alternatives Considered

Notebook-native training duplicates logic. Mid-batch exact resume would require
worker/augmentation/prefetch state and is deferred. Containers are unnecessary here.

## Consequences

Runtime/checkpoint utilities are implemented; their integration into a real trainer
awaits model and data implementation. AMP precision changes require experiment
comparability review. Preinstalled Colab torch is outside the core lock and must
be recorded and pinned for a real study. MLflow remains a proposed adapter.

## Testing / Verification

CPU-only synthetic tests cover path resolution, CUDA rejection/fallback, AMP
selection, full checkpoint fields, provenance guards and restoration through a
fake backend. Real PyTorch/CUDA and Drive round trips require later integration
tests. No bitwise guarantee is made across devices/framework versions.

## Approval Status

Proposed. Infrastructure extension was requested; formal L2 review by @tungvoi38
is still required before integration. No research training is authorized.
