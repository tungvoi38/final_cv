"""Explicit, bounded acquisition; no dataset selection or extraction is implicit."""

import re
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

from geoid.config import require
from geoid.data.manifest import safe_relative, sha256_file

SOURCE_PREFIX = "/datasets/links-ads/geoid-flood/resolve/"


def plan_acquisition(config, *, mode="sample"):
    require(mode in {"sample", "full"}, "invalid acquisition mode")
    acquisition = config.get("acquisition", {})
    revision = acquisition.get("revision")
    require(
        isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision),
        "Acquisition blocked: select an immutable official dataset commit revision",
    )
    require(config.get("revision") == revision, "dataset and acquisition revisions disagree")
    files = acquisition.get("files")
    require(
        isinstance(files, list) and bool(files), "Acquisition blocked: no verified file selection"
    )
    selected, paths = [], set()
    for item in files:
        require(isinstance(item, dict), "acquisition file must be a mapping")
        safe_relative(item.get("path"))
        require(item["path"] not in paths, "duplicate acquisition path")
        paths.add(item["path"])
        require(type(item.get("size_bytes")) is int and item["size_bytes"] > 0, "invalid size")
        require(
            bool(re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", "")))),
            "each acquired file requires a verified SHA256",
        )
        require(type(item.get("sample")) is bool, "file must explicitly declare sample membership")
        url = urlsplit(item.get("url", ""))
        require(
            url.scheme == "https"
            and url.netloc == "huggingface.co"
            and url.path.startswith(SOURCE_PREFIX + revision + "/")
            and not url.query
            and not url.fragment,
            "URL must pin the official dataset revision",
        )
        safe_relative(url.path.removeprefix(SOURCE_PREFIX + revision + "/"))
        if mode == "full" or item["sample"]:
            selected.append(item)
    require(bool(selected), "no files selected for this mode")
    if mode == "sample":
        cap = acquisition.get("sample_max_bytes")
        require(type(cap) is int and cap > 0, "invalid sample size cap")
        require(sum(x["size_bytes"] for x in selected) <= cap, "sample exceeds configured byte cap")
    return selected


def transfer_file(item, root, *, overwrite=False):
    """Stream to a temporary file, verify, then publish; no archive extraction."""
    safe_relative(item["path"])
    root = Path(root).resolve()
    target = (root / item["path"]).resolve()
    require(target.is_relative_to(root), "output path escapes destination")
    require(overwrite or not target.exists(), f"refusing overwrite: {item['path']}")
    target.parent.mkdir(parents=True, exist_ok=True)
    require(
        shutil.disk_usage(target.parent).free >= item["size_bytes"] + 16 * 1024 * 1024,
        "insufficient disk space (including safety margin)",
    )
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".part", delete=False) as out:
            temporary = Path(out.name)
            with urlopen(item["url"], timeout=60) as response:
                require(urlsplit(response.url).scheme == "https", "insecure provider redirect")
                count = 0
                while block := response.read(1024 * 1024):
                    count += len(block)
                    require(count <= item["size_bytes"], "download exceeds declared size")
                    out.write(block)
        require(count == item["size_bytes"], "download size mismatch")
        require(sha256_file(temporary) == item["sha256"], "download checksum mismatch")
        if overwrite:
            temporary.replace(target)
        else:
            # Exclusive creation prevents a concurrent download from being overwritten.
            try:
                with target.open("xb") as out, temporary.open("rb") as src:
                    shutil.copyfileobj(src, out)
            except FileExistsError:
                raise ValueError(f"refusing overwrite: {item['path']}") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def acquire(
    config,
    *,
    output,
    mode="sample",
    execute=False,
    verify_only=False,
    overwrite=False,
    confirm_full=False,
):
    files = plan_acquisition(config, mode=mode)
    root = Path(output).resolve()
    if verify_only:
        for item in files:
            target = (root / item["path"]).resolve()
            require(target.is_relative_to(root), "verification path escapes root")
            require(
                target.is_file() and target.stat().st_size == item["size_bytes"],
                f"missing/wrong-size file: {item['path']}",
            )
            require(sha256_file(target) == item["sha256"], f"checksum mismatch: {item['path']}")
        return {"status": "verified", "files": len(files)}
    plan = {
        "status": "dry-run",
        "files": len(files),
        "bytes": sum(x["size_bytes"] for x in files),
        "output": str(root),
    }
    if not execute:
        return plan
    require(mode != "full" or confirm_full, "full download requires --confirm-full")
    root.mkdir(parents=True, exist_ok=True)
    # Temporary file and exclusive publication may temporarily duplicate one file.
    require(
        shutil.disk_usage(root).free
        >= plan["bytes"] + max(x["size_bytes"] for x in files) + 16 * 1024 * 1024,
        "insufficient disk space for selected acquisition",
    )
    for item in files:
        transfer_file(item, root, overwrite=overwrite)
    return {**plan, "status": "downloaded-and-verified"}
