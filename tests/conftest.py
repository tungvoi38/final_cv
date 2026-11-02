"""Synthetic fixtures only: no real data, model output or experiment results."""

import hashlib

import numpy as np
import pytest


@pytest.fixture
def sample():
    return {
        "sar": np.ones((4, 4, 4), dtype=np.float32),
        "optical": np.ones((12, 4, 4), dtype=np.float32),
        "dem": np.ones((1, 4, 4), dtype=np.float32),
        "mask": np.array([[0, 1, 2, 255]] * 4, dtype=np.int64),
        "event_id": "synthetic-event-a",
        "aoi_id": "synthetic-aoi-a",
        "tile_id": "tile-a",
    }


@pytest.fixture
def manifest():
    return {
        "dataset_name": "synthetic",
        "source": "synthetic fixture",
        "revision": "fixture-v1",
        "downloaded_at": "2026-11-02T00:00:00+00:00",
        "records": [
            {
                "event_id": f"event-{i}",
                "aoi_id": f"aoi-{i}",
                "tile_id": f"tile-{i}",
                "parent_tile_id": f"parent-{i}",
                "split": split,
                "files": {
                    m: {
                        "path": f"event-{i}/{m}.bin",
                        "size_bytes": 3,
                        "sha256": hashlib.sha256(b"abc").hexdigest(),
                    }
                    for m in ("sar_pre", "sar_post", "optical", "dem", "mask")
                },
            }
            for i, split in enumerate(("train", "val", "test"))
        ],
    }


@pytest.fixture
def acquisition():
    revision = "a" * 40
    return {
        "revision": revision,
        "acquisition": {
            "revision": revision,
            "sample_max_bytes": 100,
            "files": [
                {
                    "path": "sample.bin",
                    "size_bytes": 3,
                    "sha256": hashlib.sha256(b"abc").hexdigest(),
                    "sample": True,
                    "url": f"https://huggingface.co/datasets/links-ads/geoid-flood/resolve/{revision}/sample.bin",
                }
            ],
        },
    }
