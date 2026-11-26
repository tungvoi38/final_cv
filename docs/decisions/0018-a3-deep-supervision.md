---
id: ADR-0018
title: Evaluate training-only supervision at one A3 decoder stage
status: proposed
level: L2
proposed_by:
  developer: unknown
  agent: OpenAI Codex (GPT-6)
  session_id: not available
related_issue: Seed-42 deep supervision ablation
related_pr: not available
impact:
  - scripts/train_unet.py
  - scripts/unet_deep_supervision.py
  - scripts/evaluate_unet_eoa.py
---

## Context

Historical proposal. The experiment was discontinued and its implementation
was removed from the trainer and evaluator.

The A3 model supervises the final segmentation head at full resolution.

## Decision

Add a training-only Conv1×1 auxiliary head after the third decoder block.
Combine the main CE+Dice loss and 0.2 times the auxiliary CE+Dice loss.
Resize the remapped mask with nearest-neighbor interpolation and preserve
ignore label 255. Use the existing head learning rate for the new weights.
Save and average the auxiliary weights with the rest of the model.

## Rationale

The auxiliary loss provides a direct training signal to the quarter-resolution
decoder while evaluation continues through the full-resolution head.

## Alternatives Considered

Supervising multiple stages would add more heads and expose several loss
weights to selection.

## Consequences

The experiment adds 195 parameters. Compare the validation-selected seed-42
SMA checkpoint with the seed-42 reference at zero bias, without Flip TTA.
Stop the experiment if paired validation pooled mIoU fails to improve.

## Testing / Verification

The synthetic smoke command checks tensor shapes, ignore-mask behavior,
gradient flow, optimizer membership, SMA update and strict checkpoint reload.

## Approval Status

Proposed. Research Lead review is pending.
