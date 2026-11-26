---
id: ADR-0010
title: Evaluate Ensemble of Averages with the frozen A3 ERM configuration
status: proposed
level: L2
proposed_by:
  developer: unknown
  agent: OpenAI Codex (GPT-6)
  session_id: not available
related_issue: Kuro Siwo EoA implementation
related_pr: not available
impact:
  - scripts/train_unet.py
  - scripts/unet_sma.py
  - scripts/evaluate_unet_eoa.py
  - configs/experiments/unet_a3_eoa.json
---

## Context

The saved A3 event-sqrt seed 42 run used 15 epochs and batch size 2. The
separate batch-size-16 experiment has a different optimizer-step budget. The
comparison uses the original A3 event-sqrt run as its reference.

## Decision

Keep the existing training recipe and add SMA as an explicit option.
For seeds 42, 43 and 44, keep one online training trajectory per seed and
select its proposed SMA and matched online checkpoints independently by
validation pooled mIoU. Aggregate the two families of members separately by
mean per-pixel logits. Freeze the selected checkpoints and their digests before
test evaluation. Run the same segmentation evaluator on both ensembles and
the corresponding individual members.

## Rationale

The Salesforce implementation copies the latest weights and floating buffers
through step 99. At step 100, the average includes the copies from steps 99
and 100 and the averaging counter is 1. The recurrence uses a changing
coefficient and averages BatchNorm running statistics without averaging the
integer batch counter. [Official implementation](https://github.com/salesforce/ensemble-of-averages/blob/main/domainbed/algorithms.py#L109-L160)

## Alternatives Considered

An EMA has exponential weighting. Applying averaging to saved epoch
checkpoints would use a much sparser sampling of the optimizer trajectory.
Neither realizes the specified step-level recurrence.

## Consequences

The segmentation adaptation counts successful AMP optimizer updates and keeps
online BatchNorm in its existing training mode. The SMA network uses averaged
running statistics in eval mode. Salesforce freezes ResNet BatchNorm in its
classification pipeline. The SMAs use the same training splits and label metrics
as ERM. Extra validation preserves the online data generator state.
[Salesforce ResNet implementation](https://github.com/salesforce/ensemble-of-averages/blob/main/domainbed/networks.py#L101-L118)

## Testing / Verification

The tests cover the threshold, BatchNorm, AMP skips, epoch-boundary replay,
logit alignment and the frozen evaluation gate. The GPU smoke uses an isolated
training subset and the regular validation split.

## Approval Status

Proposed. Research Lead review is pending.
