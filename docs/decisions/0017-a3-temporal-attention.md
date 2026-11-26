---
id: ADR-0017
title: Evaluate temporal differential attention at one A3 decoder stage
status: proposed
level: L2
proposed_by:
  developer: unknown
  agent: OpenAI Codex (GPT-6)
  session_id: not available
related_issue: Seed-42 temporal attention ablation
related_pr: not available
impact:
  - scripts/train_unet.py
  - scripts/unet_temporal_attention.py
  - scripts/evaluate_unet_temporal_attention.py
---

## Context

Historical proposal. The experiment was discontinued after validation and
its implementation was removed from the trainer and evaluator.

The 11-channel A3 model receives normalized VV/VH from pre1, pre2 and post,
along with DEM and four derived channels. This ablation evaluates feature
differences from the original SAR pairs.

## Decision

Share two stride-2 convolutions across the three acquisitions. Combine the
absolute temporal difference and a sigmoid gate conditioned on post-event
features. Project to 64 channels and add once after the third SMP decoder
block, at quarter resolution. Initialize the projection to zero. Use the
decoder learning rate for the added parameters and include them in SMA.

## Rationale

The design borrows shared temporal features and differential attention from
[DAM-Net](https://github.com/Tamer-Saleh/S1GFlood-Detection) and
[Siamese Differential Attention U-Net](https://doi.org/10.3390/earth7030087).
The single residual isolates one architectural change.

## Alternatives Considered

A second ResNet and fusion at several decoder scales would increase compute
and change more than one decoder stage.

## Consequences

The opt-in module adds 30,432 parameters. The paired seed-42 comparison uses
the selected SMA checkpoints, no Flip TTA, and reports b=0 and the bias
selected separately for each model by validation pooled Flood IoU.

## Testing / Verification

The synthetic GPU smoke check covers zero-init logit equality, shape, first
and subsequent gradient flow, optimizer membership, SMA state and peak CUDA
allocation. The completed experiment's artifacts are retained separately.

## Approval Status

Proposed. Research Lead review is pending.
