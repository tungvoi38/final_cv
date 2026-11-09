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

## Optional SAR noise proposal — 2026-11-09

Task: user requested implementation, tests and Git push of optional train SAR noise.
Tool/provider: OpenAI Codex; model/session ID: not available; human identity: unknown.
This extends the proposed protocol; it does not accept the ADR or change shared contracts.

`KuroDataset(..., sar_noise=False, noise_p=0.3, noise_std=0.05)` keeps noise off by
default. When enabled, each sample is selected with probability 0.3; valid linear
SAR power is multiplied before dB by independent Gamma factors for every
pixel/band/time, shape=rate=1/std² (400 at std=0.05). Mean is 1, std is 0.05;
this is not a hard ±5% bound. The Gamma form is motivated by the intensity speckle
model (https://www.mdpi.com/2072-4292/11/13/1532); the selected strength/probability
are heuristic trial values, not calibrated Kuro sensor parameters or optimal values.
This adds perturbations to already noisy data, not despeckling. Independent noise
may perturb temporal/polarization relationships; no acquisition realism is claimed.

Only input-valid pixels change. Labels, DEM, validity and raw arrays are unchanged.
Val/test constructors reject noise even at p=0. Noise is independent of the D4 flag.
Finite p in [0,1] and std in [0.001,1] are required; std bounds are implementation
limits, not recommended search bounds. Noisy nonpositive/nonfinite powers fail
instead of silently changing validity. Torch CPU RNG follows worker seeding.
Disabled noise consumes no RNG; physical(i) remains unaugmented by default.
Physical(i, apply_noise=True) and ds[i] apply the configured probability.
Save dataset.augmentation_config with every training run; global CONFIG remains
baseline preprocessing. QC reports record selected and forced-test configurations.

Experiment impact: compare baseline versus noise with the same splits, seed and
other settings; select on validation only. Extra noise RNG draws mean later D4
sequences need not match. Trainer should separately seed sampler generators when
paired ordering matters. No accuracy improvement is established; real model runs
remain the model team's responsibility. Alternatives: no noise (default baseline),
or calibrated/correlated noise (deferred pending evidence and review).

Verification executed for noise: 26 existing local regression tests passed.
New executable notebook assertions passed on a small real-training fixture and
synthetic tensors: on/off/p=0, probability branches, seeded repeatability, invalid
parameters, val/test rejection, Gamma moments, independence, all-invalid input,
raw nonmutation, masks/DEM unchanged and normalization/D4 preserved. Exact outputs
and RNG state matched the committed baseline across five real train samples with
D4 both on and off. Both adapters and dummy-head backward with noisy input passed;
comparison PNG generated. These are code checks, not measured model improvements.

## Approval Status

Proposed only. L2/L3 Lead review by @tungvoi38 required before integration/merge.
User authorized the standalone deliverable and cleanup, not agent self-approval
of research protocols. No accepted ADRs or shared contracts were changed.
