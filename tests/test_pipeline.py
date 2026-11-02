"""Synthetic boundary integration only; no fake model inference or reported scores."""

import numpy as np

from geoid.contracts import validate_dg_inputs, validate_model_output, validate_sample
from geoid.data.manifest import canonical_hash, validate_manifest


def test_synthetic_boundary_pipeline(sample, manifest):
    validate_manifest(manifest)
    validate_sample(sample)
    batch = {k: np.stack([sample[k], sample[k]]) for k in ("sar", "optical", "dem", "mask")}
    assert batch["sar"].shape == (2, 4, 4, 4)
    boundary_fixture = {
        "logits": np.ones((2, 3, 4, 4), dtype=np.float32),
        "features": {"alignment": np.ones((2, 8, 4, 4), dtype=np.float32)},
    }
    validate_model_output(boundary_fixture, batch_size=2, height=4, width=4)
    validate_dg_inputs(
        boundary_fixture["features"]["alignment"],
        batch["mask"],
        ["event-0", "source-event-2"],
        split="train",
    )
    assert canonical_hash(manifest) == canonical_hash(dict(reversed(list(manifest.items()))))
