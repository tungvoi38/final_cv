"""Proposed supervised cross-event ClassMix for collated Kuro training batches.

Uses torch's device RNG. Call after collation and before model_inputs. Donors
always come from the original batch. No dataset/model dependency.
"""

import math

import torch


class ClassMix:
    def __init__(self, p=0.5):
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("p must be finite in [0,1]")
        self.p = float(p)

    def __call__(self, batch, *, training):
        if not training or self.p == 0:
            return batch, {"applied": 0, "disabled": True, "records": []}
        labels = batch["mask"]
        if labels.ndim != 3 or labels.dtype != torch.int64:
            raise ValueError("mask must be int64 [B,H,W]")
        n, h, w = labels.shape
        if min(n, h, w) < 1:
            raise ValueError("Empty batch or spatial grid")
        events = batch["event_id"]
        if len(events) != n or any(not isinstance(e, str) or not e for e in events):
            raise ValueError("Require nonempty event_id per sample")
        if len(batch["split"]) != n or any(s != "train" for s in batch["split"]):
            raise ValueError("ClassMix accepts only training sources")
        keys = ("sar", "dem", "mask", "input_valid", "valid_mask")
        for key in keys:
            value = batch[key]
            if value.shape[0] != n or value.shape[-2:] != (h, w):
                raise ValueError(f"Unaligned {key}")
            if value.device != labels.device:
                raise ValueError("All tensors must share a device")
        if batch["sar"].ndim != 5 or batch["sar"].shape[1:3] != (3, 2):
            raise ValueError("SAR must be [B,3,2,H,W]")
        if batch["dem"].shape != (n, 1, h, w):
            raise ValueError("DEM must be [B,1,H,W]")
        for key in ("input_valid", "valid_mask"):
            if batch[key].shape != labels.shape or batch[key].dtype != torch.bool:
                raise ValueError("Validity masks must be bool [B,H,W]")
        if not torch.isin(labels, labels.new_tensor([0, 1, 2, 255])).all():
            raise ValueError("Unknown labels")
        if not torch.equal(batch["valid_mask"], labels != 255):
            raise ValueError("Label validity mismatch")
        if (batch["valid_mask"] & ~batch["input_valid"]).any():
            raise ValueError("Supervision outside valid input")
        for key in ("sar", "dem"):
            if not batch[key].is_floating_point() or not torch.isfinite(batch[key]).all():
                raise ValueError("Inputs must be finite floating tensors")
        result = {**batch, **{key: batch[key].clone() for key in keys}}
        masks = torch.zeros_like(labels, dtype=torch.bool)
        records = []
        present_classes = [row.unique() for row in labels]
        present_classes = [classes[classes != 255] for classes in present_classes]
        for i in range(n):
            record = {"recipient_index": i, "recipient_event": events[i], "reason": "probability"}
            records.append(record)
            if self.p < 1 and torch.rand((), device=labels.device).item() >= self.p:
                continue
            candidates = [j for j in range(n) if events[j] != events[i]]
            if not candidates:
                record["reason"] = "no_other_event"
                continue
            candidates = [j for j in candidates if len(present_classes[j]) >= 2]
            if not candidates:
                record["reason"] = "no_eligible_donor"
                continue
            j = candidates[torch.randint(len(candidates), (), device=labels.device).item()]
            classes = present_classes[j]
            chosen = classes[
                torch.randperm(len(classes), device=labels.device)[: len(classes) // 2]
            ]
            mask = torch.isin(labels[j], chosen)
            if not mask.any() or mask.all():
                record["reason"] = "degenerate_mask"
                continue
            masks[i] = mask
            for key in keys:
                result[key][i] = torch.where(mask, batch[key][j], batch[key][i])
            record.update(
                reason="applied",
                donor_index=j,
                donor_event=events[j],
                classes=chosen.tolist(),
                area_fraction=mask.float().mean().item(),
                recipient_id=batch.get("sample_id", [None] * n)[i],
                donor_id=batch.get("sample_id", [None] * n)[j],
            )
        info = {
            "applied": sum(r["reason"] == "applied" for r in records),
            "disabled": False,
            "records": records,
        }
        # event_id remains the recipient ID for indexing, NOT a pure-event claim.
        result["classmix_provenance"] = records
        info["mix_mask"] = masks
        return result, info
