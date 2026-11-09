"""Standalone Kuro Siwo preprocessing, Dataset, staging and model adapters.

Only this Python file is required by the model team (plus installed numpy,
pandas, pyarrow, rasterio and torch). No sibling modules or Hub access required.
No persistent processed arrays. See RawArchive, KuroDataset and model_inputs.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import rasterio

REPO = "aialliance/kuro_siwo"
REVISION = "25997dc683df721de44fe6f68a4ce1b36f4562eb"
RAW_NAME = "geobench_kuro_siwo.tortilla"
RAW_BYTES = 7529588233
RAW_SHA256 = "4830fe6f23bf9750dee0c765850724b55026bf5d47cb67162d3ef7dcb04c3bbd"
SOURCE_REVISION = "d7ae6aafc18e1da1bd6efb130f0a306125ee0fef"
ORIGINAL_REVISION = "4347ed173c4e48f5a9d578bb5fe8453706b08e5e"
TIMES = ("pre_event_1", "pre_event_2", "post_event")
MODALITIES = (*TIMES, "dem", "mask", "invalid_data")
EXPECTED_COUNTS = {"train": 4000, "validation": 1000, "test": 2000}
CONFIG = {
    "version": "kuro-raw-v1",
    "dataset": REPO,
    "revision": REVISION,
    "raw_sha256": RAW_SHA256,
    "source_units": "linear_sigma0_power",
    "sar_transform": "10*log10(x), x>0; no epsilon or clipping",
    "time_order": list(TIMES),
    "band_order": ["VV", "VH"],
    "shape": [224, 224],
    "label_mapping": {"0": 0, "1": 1, "2": 2, "3": 255},
    "validity_value": 1,
    "dem_units": "metres",
    "dem_invalid": "nonfinite, abs>=1e20 or nonzero declared nodata",
    "invalid_physical_fill": 0,
    "invalid_normalized_fill": 0,
    "normalization": {
        "name": "terramind_v1_fixed_external",
        "sar_mean": [-12.599, -20.293],
        "sar_std": [5.195, 5.890],
        "dem_mean": 670.665,
        "dem_std": 951.272,
        "fitted_on_kuro": False,
        "source": "https://github.com/IBM/terramind/blob/53d61da09213c2c3629931b6322f94cce617886a/"
        "configs/terramind_v1_base_sen1floods11.yaml",
    },
    "source_evidence": {
        "grd": f"https://github.com/Orion-AI-Lab/KuroSiwo/blob/{ORIGINAL_REVISION}/"
        "configs/grd_preprocessing.xml",
        "validity": f"https://github.com/Orion-AI-Lab/KuroSiwo/blob/{ORIGINAL_REVISION}/"
        "dataset/Dataset.py",
        "labels": f"https://github.com/The-AI-Alliance/GEO-Bench-2/blob/{SOURCE_REVISION}/"
        "geobench_v2/datasets/kuro_siwo.py",
        "converter": f"https://github.com/The-AI-Alliance/GEO-Bench-2/blob/{SOURCE_REVISION}/"
        "geobench_v2/generate_benchmark/kuro_siwo.py",
    },
}
ARRAY_KEYS = ("sar_db", "dem_m", "mask", "input_valid", "valid_mask")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_copy(source, destination, expected_hash=None):
    """Never trust a partial copy or overwrite an unrelated existing artifact."""
    source, destination = Path(source), Path(destination)
    expected_hash = expected_hash or sha256(source)
    if destination.exists():
        if sha256(destination) != expected_hash:
            raise ValueError(f"Checksum mismatch: {destination}; move it aside before retry")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".partial")
    shutil.copyfile(source, tmp)
    if sha256(tmp) != expected_hash:
        raise ValueError(f"Copy verification failed: {tmp}")
    os.replace(tmp, destination)
    return destination


def verify_raw(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != RAW_BYTES:
        raise ValueError(f"Missing/truncated raw archive: {path}")
    digest = sha256(path)
    if digest != RAW_SHA256:
        raise ValueError(f"Raw SHA256 mismatch: {path}; keep aside and retry download")
    return digest


def stage_raw(source, destination):
    """Model-team staging: missing source raises; no automatic Hub fallback."""
    source, destination = Path(source), Path(destination)
    if not source.is_file() or source.stat().st_size != RAW_BYTES:
        raise ValueError(f"Missing/truncated shared raw archive: {source}")
    if source.resolve() == destination.resolve():
        verify_raw(source)
        return source
    return verified_copy(source, destination, RAW_SHA256)


def now():
    return datetime.now(UTC).isoformat()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)


def read_metadata(path, base=0, length=None):
    """Read bounded embedded Parquet in the pinned TACO v1 format."""
    path = Path(path)
    length = path.stat().st_size - base if length is None else int(length)
    if base < 0 or length < 18 or base + length > path.stat().st_size:
        raise ValueError("Invalid container bounds")
    with path.open("rb") as stream:
        stream.seek(base)
        header = stream.read(18)
        if header[:2] not in {b"#y", b"WX"}:
            raise ValueError("Unsupported TACO header")
        offset = int.from_bytes(header[2:10], "little")
        size = int.from_bytes(header[10:18], "little")
        if offset < 18 or not 0 < size <= 16 * 1024**2 or offset + size > length:
            raise ValueError("Invalid metadata bounds")
        stream.seek(base + offset)
        return pq.read_table(io.BytesIO(stream.read(size))).to_pandas()


def audit_metadata(frame, full=True):
    required = {"tortilla:id", "actid", "aoiid", "tortilla:data_split"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("Missing metadata columns or empty manifest")
    for key in required:
        if frame[key].isna().any() or frame[key].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing identifier: {key}")
    if frame["tortilla:id"].astype(str).duplicated().any():
        raise ValueError("Duplicate sample IDs")
    if not frame["tortilla:id"].astype(str).str.fullmatch(r"\d+").all():
        raise ValueError("Unsupported sample ID")
    if set(frame["tortilla:data_split"]) - set(EXPECTED_COUNTS):
        raise ValueError("Unknown official split")
    overlap = frame.groupby("actid")["tortilla:data_split"].nunique()
    if (overlap > 1).any():
        raise ValueError("Event overlap across splits; do not re-split automatically")
    if (
        "add_test_split" in frame
        and not frame["add_test_split"].map(lambda x: str(x).lower() in {"0", "0.0", "false"}).all()
    ):
        raise ValueError("Additional test flags require explicit protocol review")
    counts = {str(k): int(v) for k, v in frame["tortilla:data_split"].value_counts().items()}
    if full and counts != EXPECTED_COUNTS:
        raise ValueError(f"Release sample counts changed: {counts}")
    events = {
        str(k): int(v) for k, v in frame.groupby("tortilla:data_split")["actid"].nunique().items()
    }
    return {
        "samples": counts,
        "events": events,
        "event_overlap": 0,
        "parent_scene_independence": "unverified; source paths not published",
    }


def read_sample(path, row):
    """Read six TIFF payloads without extracting files or retaining GDAL handles."""
    base, length = int(row["tortilla:offset"]), int(row["tortilla:length"])
    nested = read_metadata(path, base, length)
    if nested["tortilla:id"].duplicated().any() or set(nested["tortilla:id"]) != set(MODALITIES):
        raise ValueError("Missing/duplicate/unexpected modality")
    for key in ("actid", "aoiid", "tortilla:data_split"):
        if key not in nested or not nested[key].astype(str).eq(str(row[key])).all():
            raise ValueError(f"Nested metadata mismatch: {key}")
    arrays, flags, profiles = {}, {}, {}
    with Path(path).open("rb") as stream:
        for _, entry in nested.iterrows():
            name = entry["tortilla:id"]
            offset, size = int(entry["tortilla:offset"]), int(entry["tortilla:length"])
            if offset < 18 or not 0 < size < 32 * 1024**2 or offset + size > length:
                raise ValueError("Invalid TIFF bounds")
            stream.seek(base + offset)
            with rasterio.io.MemoryFile(stream.read(size)) as mem, mem.open() as src:
                arrays[name] = src.read()
                flags[name] = src.read_masks() > 0
                profiles[name] = {
                    "crs": str(src.crs),
                    "transform": list(src.transform.to_gdal()),
                    "shape": [src.height, src.width],
                    "nodata": src.nodata,
                }
    reference = profiles[TIMES[0]]
    if reference["crs"] == "None":
        raise ValueError("Missing CRS")
    for name in MODALITIES:
        p = profiles[name]
        if (
            p["shape"] != [224, 224]
            or p["crs"] != reference["crs"]
            or not np.allclose(p["transform"], reference["transform"], rtol=0, atol=1e-6)
        ):
            raise ValueError(f"Misregistered raster: {name}")
        channels = 2 if name in TIMES else 1
        if arrays[name].shape != (channels, 224, 224):
            raise ValueError(f"Unexpected raster shape: {name}")
    if reference["crs"] != row["stac:crs"] or not np.allclose(
        reference["transform"], row["stac:geotransform"], rtol=0, atol=1e-6
    ):
        raise ValueError("Outer metadata / raster spatial mismatch")
    return arrays, flags, profiles


def multiplicative_sar_noise(sar, input_valid, std):
    """Independent Gamma factors per time/band/pixel, on valid linear power only.

    Unit mean, std=std; this is an augmentation, not an estimated sensor model.
    Uses the CPU torch RNG (including DataLoader worker seeds). Never mutates raw.
    """
    import torch

    if not np.isfinite(std) or not 0.001 <= std <= 1:
        raise ValueError("noise_std must be finite and in [0.001, 1]")
    result = sar.copy()
    selected = sar[..., input_valid]
    if selected.size == 0:
        return result
    concentration = torch.tensor(1.0 / std**2, dtype=torch.float64, device="cpu")
    factors = torch.distributions.Gamma(concentration, concentration).sample(selected.shape)
    perturbed = (selected.astype(np.float64) * factors.numpy()).astype(np.float32)
    if not (np.isfinite(perturbed) & (perturbed > 0)).all():
        raise ValueError("SAR noise produced nonfinite/nonpositive power; reduce noise_std")
    result[..., input_valid] = perturbed
    return result


def preprocess(arrays, flags, profiles, *, noise_std=None):
    """Source masks control input validity; labels never alter input pixels."""
    sar = np.stack([arrays[t] for t in TIMES]).astype(np.float32)
    dem = arrays["dem"].astype(np.float32)
    label = arrays["mask"][0]
    mna = arrays["invalid_data"][0]
    if not np.isin(label, [0, 1, 2, 3]).all():
        raise ValueError("Unknown raw label; expected 0/1/2/3 before upstream remapping")
    if not np.isin(mna, [0, 1]).all():
        raise ValueError("Unknown validity-mask encoding")
    sar_ok = np.isfinite(sar) & (sar > 0)
    for i, name in enumerate(TIMES):
        sar_ok[i] &= flags[name]
    # Converter inherits the first SAR profile for all modalities. Its nodata=0
    # cannot erase class 0 or valid sea-level DEM. Honor nonzero DEM nodata only.
    dem_ok = np.isfinite(dem) & (np.abs(dem) < 1e20)
    nd = profiles["dem"]["nodata"]
    if nd is not None and np.isfinite(nd) and nd != 0:
        dem_ok &= dem != nd
    input_valid = (mna == 1) & sar_ok.all(axis=(0, 1)) & dem_ok.all(axis=0)
    valid = input_valid & (label != 3)
    if noise_std is not None:
        sar = multiplicative_sar_noise(sar, input_valid, noise_std)
    sar_db = np.zeros_like(sar)
    np.log10(sar, out=sar_db, where=sar_ok)
    sar_db *= 10
    sar_db[:, :, ~input_valid] = 0
    dem = np.where(input_valid[None], dem, 0).astype(np.float32)
    mask = np.where(valid, label, 255).astype(np.uint8)
    return {
        "sar_db": sar_db,
        "dem_m": dem,
        "mask": mask,
        "input_valid": input_valid,
        "valid_mask": valid,
    }


def validate_arrays(sample):
    shapes = {
        "sar_db": (3, 2, 224, 224),
        "dem_m": (1, 224, 224),
        "mask": (224, 224),
        "input_valid": (224, 224),
        "valid_mask": (224, 224),
    }
    for key, shape in shapes.items():
        if sample[key].shape != shape:
            raise ValueError(f"Bad {key} shape: {sample[key].shape}")
    if sample["sar_db"].dtype != np.float32 or sample["dem_m"].dtype != np.float32:
        raise ValueError("Expected float32 physical arrays")
    if sample["mask"].dtype != np.uint8:
        raise ValueError("Expected uint8 stored labels")
    if any(sample[k].dtype != bool for k in ("input_valid", "valid_mask")):
        raise ValueError("Expected boolean masks")
    if not all(np.isfinite(sample[k]).all() for k in ("sar_db", "dem_m")):
        raise ValueError("Nonfinite physical arrays")
    if not np.isin(sample["mask"], [0, 1, 2, 255]).all():
        raise ValueError("Bad label mapping")
    if not np.array_equal(sample["valid_mask"], sample["mask"] != 255):
        raise ValueError("Label validity mismatch")
    if (sample["valid_mask"] & ~sample["input_valid"]).any():
        raise ValueError("Supervision outside valid input")
    for key in ("sar_db", "dem_m"):
        if np.any(sample[key][..., ~sample["input_valid"]] != 0):
            raise ValueError("Invalid storage fill")


def transform_sample(sample, d4=0):
    """Apply one shared dihedral transform, then fixed external normalization."""
    if d4 not in range(8):
        raise ValueError("d4 must be 0..7")
    result = dict(sample)
    for key in ARRAY_KEYS:
        value = np.rot90(sample[key], d4 % 4, axes=(-2, -1))
        if d4 >= 4:
            value = np.flip(value, axis=-1)
        result[key] = np.ascontiguousarray(value)
    norm = CONFIG["normalization"]
    sar = result["sar_db"] - np.array(norm["sar_mean"], np.float32)[None, :, None, None]
    sar /= np.array(norm["sar_std"], np.float32)[None, :, None, None]
    dem = (result["dem_m"] - norm["dem_mean"]) / norm["dem_std"]
    sar[:, :, ~result["input_valid"]] = 0
    dem[:, ~result["input_valid"]] = 0
    result["sar"] = sar.astype(np.float32)
    result["dem"] = dem.astype(np.float32)
    return result


def row_record(row):
    return {
        "sample_id": str(row["tortilla:id"]),
        "event_id": str(row["actid"]),
        "aoi_id": f"{row['actid']}:{row['aoiid']}",
        "split": row["tortilla:data_split"],
        "crs": row["stac:crs"],
        "geotransform": list(row["stac:geotransform"]),
        "flood_date": str(row["flood_date"]),
        "parent_scene_id": None,
        "source_offset": int(row["tortilla:offset"]),
        "source_length": int(row["tortilla:length"]),
    }


def model_inputs(batch):
    """Adapters only; no backbone/checkpoint construction or normalization here."""
    import torch

    sar, dem = batch["sar"], batch["dem"]
    if sar.ndim != 5 or tuple(sar.shape[1:]) != (3, 2, 224, 224):
        raise ValueError("Expected SAR [B,3,2,224,224]")
    if tuple(dem.shape) != (sar.shape[0], 1, 224, 224):
        raise ValueError("Expected DEM [B,1,224,224]")
    cnn = torch.cat([sar.flatten(1, 2), dem], dim=1)
    temporal = {
        "S1GRD": sar.permute(0, 2, 1, 3, 4).contiguous(),
        "DEM": dem.unsqueeze(2).expand(-1, -1, 3, -1, -1).contiguous(),
    }
    return cnn, temporal


class RawArchive:
    """Verify once, share audited metadata across splits; no open file handles.

    allow_fixture=True exists for tests only and marks all evidence unverified.
    Production notebooks never enable it.
    """

    def __init__(self, path, *, allow_fixture=False):
        self.path = Path(path).resolve()
        self.verified_release = not allow_fixture
        if self.verified_release:
            verify_raw(self.path)
        self.frame = read_metadata(self.path)
        self.audit = audit_metadata(self.frame, full=self.verified_release)
        stat = self.path.stat()
        self.signature = (stat.st_size, stat.st_mtime_ns)

    def check_unchanged(self):
        stat = self.path.stat()
        if (stat.st_size, stat.st_mtime_ns) != self.signature:
            raise ValueError("Raw archive changed after verification; re-open and verify it")

    def save_metadata(self, directory):
        """Small handoff artifacts only, no raster cache."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        records = [row_record(row) for _, row in self.frame.iterrows()]
        write_json(
            directory / "manifest.json",
            {
                "dataset": REPO,
                "revision": REVISION,
                "expected_sha256": RAW_SHA256,
                "verified_release": self.verified_release,
                "created_at": now(),
                "audit": self.audit,
                "config_hash": json_hash(CONFIG),
                "records": records,
            },
        )
        write_json(directory / "preprocess_config.json", CONFIG)


class KuroDataset:
    """Map-style dataset, safely usable with DataLoader workers on Colab.

    Pass one RawArchive instance to train/validation/test to hash raw only once.
    Augmentation uses torch RNG; trainer controls seeds and worker seeding.
    sar_noise defaults off; when on, noise_p is per sample and noise_std is the
    Gamma multiplier std (not a +/- bound). D4 is controlled independently by
    augment. Save augmentation_config with runs. Noise never runs on val/test.
    """

    def __init__(self, raw, split, augment=False, *, sar_noise=False, noise_p=0.3, noise_std=0.05):
        if split not in EXPECTED_COUNTS:
            raise ValueError("split must be train, validation or test")
        if (augment or sar_noise) and split != "train":
            raise ValueError("Augmentation is allowed only for train")
        if not np.isfinite(noise_p) or not 0 <= noise_p <= 1:
            raise ValueError("noise_p must be finite and in [0, 1]")
        if not np.isfinite(noise_std) or not 0.001 <= noise_std <= 1:
            raise ValueError("noise_std must be finite and in [0.001, 1]")
        self.sar_noise = bool(sar_noise)
        self.noise_p = float(noise_p)
        self.noise_std = float(noise_std)
        self.archive = raw if isinstance(raw, RawArchive) else RawArchive(raw)
        self.rows = self.archive.frame[
            self.archive.frame["tortilla:data_split"] == split
        ].reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    @property
    def augmentation_config(self):
        """Save with each training run; global CONFIG describes unaugmented preprocessing."""
        return {
            "version": "kuro-augmentation-v1",
            "d4": bool(self.augment),
            "sar_noise": self.sar_noise,
            "noise_p": self.noise_p,
            "noise_std": self.noise_std,
            "distribution": "Gamma(shape=1/std**2, rate=1/std**2)",
            "coupling": "independent per time, band and valid pixel",
            "scope": "train only; linear power before dB; per-sample application probability",
        }

    def physical(self, index, *, apply_noise=False):
        """Unaugmented physical sample by default; opt in to configured train noise."""
        self.archive.check_unchanged()
        row = self.rows.iloc[index]
        arrays, flags, profiles = read_sample(self.archive.path, row)
        noise_std = None
        if apply_noise and self.sar_noise and self.noise_p > 0:
            import torch

            if self.noise_p == 1 or torch.rand(()).item() < self.noise_p:
                noise_std = self.noise_std
        sample = preprocess(arrays, flags, profiles, noise_std=noise_std)
        validate_arrays(sample)
        record = row_record(row)
        return {
            **sample,
            **{key: record[key] for key in ["sample_id", "event_id", "aoi_id", "split"]},
        }

    def __getitem__(self, index):
        import torch

        # Draw geometry before noise so paired first samples use the same D4 for a seed.
        d4 = int(torch.randint(8, ()).item()) if self.augment else 0
        sample = self.physical(index, apply_noise=True)
        transformed = transform_sample(sample, d4)
        return {
            "sar": torch.from_numpy(transformed["sar"]),
            "dem": torch.from_numpy(transformed["dem"]),
            "mask": torch.from_numpy(transformed["mask"].astype(np.int64)),
            "input_valid": torch.from_numpy(transformed["input_valid"]),
            "valid_mask": torch.from_numpy(transformed["valid_mask"]),
            **{key: sample[key] for key in ["sample_id", "event_id", "aoi_id", "split"]},
        }
