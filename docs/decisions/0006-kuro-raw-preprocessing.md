---
id: ADR-0006
title: Kuro Siwo raw-only storage and on-demand preprocessing
status: proposed
level: L3
proposed_by:
  developer: unknown
  agent: OpenAI Codex (GPT-6)
  session_id: not available
related_issue: user-task-kuro-raw-handoff
related_pr: not available
impact:
  - scripts/kuro_dataset.py
  - notebooks/kuro_download.ipynb
  - notebooks/kuro_preprocess.ipynb
---

## Context

User explicitly confirmed cleanup of both uncommitted old pipelines including
metadata audit notebook, and requested raw storage on Drive with a separate
preprocessing pipeline for model training on Colab. Old uncommitted proposals
and sources were backed up outside the repo before cleanup; accepted history,
tracked bootstrap, datasets and Git history are untouched.

## Decision

Store only pinned raw on Drive (~7.53 GB), stage to Colab local disk, preprocess
on sample access. No full persistent processed cache. Keep original splits and
activation IDs. Three VV/VH timestamps plus DEM; linear sigma0 to dB; raw labels
0/1/2 unchanged, 3→255. MNA=1 is valid. Inherited label/DEM nodata=0 does not
remove background or sea-level pixels. Honor nonzero DEM nodata, nonfinite and
float sentinel abs>=1e20. Invalid input and supervision masks are separate.

Use fixed external TerraMind constants for both adapters, no Kuro statistics
fit. D4 only train. CNN input is 7 channels; TerraMind input is separated SAR
and repeated static DEM via a temporal wrapper. Actual model integration is a
separate test. All-ignore samples retained for explicit trainer handling.

## Rationale

Avoid duplicate Drive storage while preserving one shared raw source and a
reproducible preprocessing contract. Download and preprocessing notebooks are
independent. Training teams with raw need only staging/loader/config/tests.

## Alternatives Considered

Persistent processed shards: earlier uncommitted implementation, not retained
because it can exceed the user's Drive quota. Pure local processing alone:
not chosen since model team trains on Colab. Per-patch normalization/clipping:
not chosen because it alters the physical signal. Optional local cache can be
reviewed later only if measured loader throughput warrants it.

## Consequences

CPU repeats preprocessing each epoch; benchmark must be measured on actual
Colab resources. Parents/disaster independence beyond actid remains unverified.
New dataset/modalities/normalization are not comparable with GEOID results under
its proposed contract; records must have distinct dataset/config provenance.
This is the experiment impact statement; no prior metric is migrated.

## Testing / Verification

Synthetic unit tests and small real train sample smoke runs exercise direct raw
reads, split isolation, label-zero/no-data, normalization, synchronized D4,
adapters and failure cases. Separate tests cover checksum/download/staging.
The preprocessing notebook contains the QC functions directly; actual full-release/Colab/model checks
must be reported separately. Actual models and full-release Colab execution have not been verified by the agent.

## Packaging update — 2026-11-09

User authorized committing/pushing exactly four files: standalone dataset module,
download notebook, preprocessing/QC notebook, and this proposed ADR. Dependencies
are installed in notebook cells. Download/QC helpers are inline; no other project
files are needed. Existing auxiliary local files are not included in this commit.
Scientific preprocessing and split semantics remain unchanged. Responsible human
identity and session ID: unavailable; tool/provider: OpenAI Codex.

Verification executed for the four-file packaging: 26 existing local unit tests
passed; inline notebook QC ran on five real training samples and returned
PASS_PARTIAL (fixture, not full release). Notebook schema/syntax validation and
Ruff lint/format passed. Governance reported zero hard failures; human review
remains required. No real backbone forward/backward or full Colab run was executed.

## Approval Status

Proposed only. L2/L3 Lead review by @tungvoi38 required before integration/merge.
User authorized the standalone deliverable and cleanup, not agent self-approval
of research protocols. No accepted ADRs or shared contracts were changed.
