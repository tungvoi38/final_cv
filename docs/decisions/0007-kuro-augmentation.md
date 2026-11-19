---
id: ADR-0007
title: Kuro supervised ClassMix and event-aware MixStyle
status: proposed
level: L3
proposed_by:
  developer: unknown
  agent: OpenAI Codex
  session_id: not available
related_issue: user-task-kuro-augmentation
related_pr: not available
impact:
  - scripts/kuro_augmentation.py
  - src/geoid/dg/mixstyle.py
  - notebooks/kuro_augmentation_qc.ipynb
---

## Context

User authorized implementation of standalone data-team modules and Colab QC,
keeping current preprocessing/normalization/D4 unchanged. Model integration is
owned by the training team. Research Lead approval is not inferred from this task.
Sources: [ClassMix reference](https://github.com/WilhelmT/ClassMix/blob/master/trainSSL.py),
[MixStyle reference](https://github.com/KaiyangZhou/mixstyle-release), and
[Kuro GRD graph](https://github.com/Orion-AI-Lab/KuroSiwo/blob/4347ed173c4e48f5a9d578bb5fe8453706b08e5e/configs/grd_preprocessing.xml).

## Decision

Propose supervised ClassMix with p=0.5 per recipient, random other-event donors with at least two valid classes
within the original training batch (reuse allowed), floor(K/2) randomly selected
present classes excluding 255. Skip fewer than two classes, degenerate masks or
missing other-event donors. Copy all times/bands, DEM, labels and validity with
one mask; retain recipient IDs plus explicit donor provenance. No in-place edits.

Provide MixStyle for floating [B,C,H,W], p=0.5 per forward, alpha=0.1, eps=1e-6,
sample variance and detached spatial statistics. Explicit other-event donor
selection supports imbalanced batches without assuming two ordered domains.
Skip single-domain batches and spatial grids smaller than two pixels. FP16/BF16
statistics are calculated in FP32. Disabled/eval paths consume no RNG. All draws
use torch device RNG; workers/device RNG must be seeded and saved by training.

Keep original preprocessing, D4 and optional SAR noise defaults unchanged. No
extra speckle filter: upstream GRD graph already contains Lee Sigma. No new
clipping, S2, modality dropout or fitted statistics. ERM/DG1/DG2 stay separate.

## Rationale

Shared ClassMix before adapters keeps temporal and modality alignment. MixStyle
can be tested without an encoder while leaving model-specific insertion explicit.
Reference sources are linked above; adaptations are specified here.
Defaults are initial experimental settings, not proven optimal on Kuro.

## Alternatives Considered

Flood-only paste changes class selection bias; deferred. Ceil(K/2) was discussed
but floor follows inspected reference code. Enforced donor permutation can fail
on imbalanced batches; sampling with replacement is explicit. New sampler,
additional speckle filtering, masked feature statistics and token-native
MixStyle require separate review; they are not silently introduced.

## Consequences

Mixed samples are not pure-event observations: event_id remains recipient identity
only. Use classmix_provenance for attribution; do not feed recipient IDs as pure
pixel provenance to event-alignment losses. MixStyle has no split metadata;
trainer must supply training-only sources and call model.eval() at evaluation.
TerraMind token/time/modality conversion and insertion sites remain unintegrated.

Experiment impact: compare each method against its own backbone ERM using the
same splits/sampler/budget and seed plan, selecting only on validation. Log actual
mix/skip rates and source provenance. No prior metrics are migrated. Full-data
and real-model validation remain outstanding.

## Testing / Verification

Synthetic tests cover exact pixel provenance, no mutation, ignore/validity,
train-only ClassMix, seeded repeatability, skip behavior, MixStyle formula and
gradients, imbalanced events, constant features, low precision, and eval identity.
Notebook runs full or capped train-data QC from a checksum-verified archive, plots all
SAR times/bands and DEM, checks adapters and reports missing integration explicitly.
Executed verification before packaging: 17 augmentation tests passed; notebook
scan/plot cells passed synthetic full, capped and no-mix cases. User-provided
Colab output scanned 4,000 train samples with the prior donor rule (21.55% mixed).
The new eligibility rule has not yet been measured on real data. No real-model
forward/backward or accuracy validation is claimed.

## Approval Status

Proposed only. @tungvoi38 L2/L3 review is required before integration/merge.
User authorized commit/push of the augmentation handoff; merge is not authorized.
Model/session identifiers:
not available; responsible human: unknown. Tool/provider: OpenAI Codex.

## Donor eligibility update — user-authorized

Select uniformly only among other-event donors with at least two classes excluding
255, before sampling the donor. Keep p=0.5 and class selection unchanged. Empty
eligible pool reports no_eligible_donor; absent other events reports no_other_event.
This changes donor distribution toward multi-class chips; compare class counts,
areas and validation performance, not just application rate. Earlier 21.55% QC
belongs to the prior algorithm. Approximately 47.5% is an estimate from prior
95% eligibility, not a measured new result or a literature optimum. Proposed only;
Lead review remains required. No sampler, encoder or preprocessing change.
