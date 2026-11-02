"""Raw provenance schema and proposed split-isolation validation."""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath

from geoid.config import require


def safe_relative(value):
    require(isinstance(value, str) and bool(value), "path must be nonempty")
    posix, windows = PurePosixPath(value), PureWindowsPath(value)
    require(
        not posix.is_absolute()
        and not windows.drive
        and "\\" not in value
        and not any(p in {"..", "."} for p in value.split("/"))
        and all(p and ":" not in p for p in value.split("/")),
        "unsafe relative path",
    )
    return posix


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def validate_manifest(manifest):
    require(isinstance(manifest, dict), "manifest must be a mapping")
    for key in ("dataset_name", "source", "revision", "downloaded_at"):
        require(
            isinstance(manifest.get(key), str) and bool(manifest[key].strip()),
            f"manifest missing {key}",
        )
    stamp = datetime.fromisoformat(manifest["downloaded_at"].replace("Z", "+00:00"))
    require(stamp.tzinfo is not None, "download timestamp needs timezone")
    records = manifest.get("records")
    require(isinstance(records, list) and bool(records), "manifest records must be nonempty")
    groups = {key: {} for key in ("event_id", "aoi_id", "parent_tile_id")}
    identities, file_splits = set(), {}
    for record in records:
        require(isinstance(record, dict), "record must be mapping")
        for key in ("event_id", "aoi_id", "tile_id", "parent_tile_id"):
            require(
                isinstance(record.get(key), str) and bool(record[key].strip()), f"missing {key}"
            )
        split = record.get("split")
        require(split in {"train", "val", "test"}, "invalid split")
        for key, assignments in groups.items():
            identity = record[key]
            require(assignments.get(identity, split) == split, f"{key}: crosses splits")
            assignments[identity] = split
        identity = (record["event_id"], record["aoi_id"], record["tile_id"])
        require(identity not in identities, "duplicate tile record")
        identities.add(identity)
        files = record.get("files")
        require(
            isinstance(files, dict)
            and {"sar_pre", "sar_post", "optical", "dem", "mask"} <= files.keys(),
            "record missing required modalities",
        )
        for file in files.values():
            require(isinstance(file, dict), "file must be a mapping")
            safe_relative(file.get("path"))
            require(
                type(file.get("size_bytes")) is int and file["size_bytes"] > 0, "invalid file size"
            )
            require(
                "sha256" in file
                and (file["sha256"] is None or re.fullmatch(r"[0-9a-f]{64}", str(file["sha256"]))),
                "invalid checksum",
            )
            require(file_splits.get(file["path"], split) == split, "raw file crosses splits")
            file_splits[file["path"]] = split


def verify_manifest_files(manifest, raw_root):
    validate_manifest(manifest)
    root = Path(raw_root).resolve()
    unverified = []
    for record in manifest["records"]:
        for item in record["files"].values():
            path = (root / item["path"]).resolve()
            require(path.is_relative_to(root), "file escapes raw root")
            require(
                path.is_file() and path.stat().st_size == item["size_bytes"],
                f"missing/wrong-size file: {item['path']}",
            )
            if item["sha256"] is None:
                unverified.append(item["path"])
            else:
                require(sha256_file(path) == item["sha256"], f"checksum mismatch: {item['path']}")
    return unverified
