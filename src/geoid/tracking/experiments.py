"""Run-record integrity boundary; MLflow logging remains unimplemented."""

import json
import math
import os
import re
import tempfile
from pathlib import Path

from geoid.config import require

COMPARISON_KEYS = (
    "architecture_version",
    "manifest_hash",
    "preprocessing_version",
    "evaluation_protocol",
)


def validate_run(run):
    require(isinstance(run, dict), "run must be a mapping")
    for key in (
        "experiment_id",
        "run_id",
        "responsible_developer",
        "agent",
        "architecture_version",
        "preprocessing_version",
        "model",
        "fusion",
        "dg_method",
        "evaluation_protocol",
    ):
        require(isinstance(run.get(key), str) and bool(run[key].strip()), f"run missing {key}")
    for key, length in (("git_sha", 40), ("config_hash", 64), ("manifest_hash", 64)):
        require(
            bool(re.fullmatch(rf"[0-9a-f]{{{length}}}", str(run.get(key, "")))), f"invalid {key}"
        )
    require(type(run.get("seed")) is int and run["seed"] >= 0, "invalid seed")
    require(isinstance(run.get("hyperparameters"), dict), "missing hyperparameters")
    require(
        run.get("status") in {"planned", "running", "completed", "failed", "cancelled"},
        "invalid run status",
    )
    metrics = run.get("metrics")
    require(
        isinstance(metrics, dict)
        and all(
            isinstance(k, str) and (v is None or (type(v) in (float, int) and math.isfinite(v)))
            for k, v in metrics.items()
        ),
        "invalid metrics",
    )
    if run["status"] != "completed":
        require(not run.get("verification"), "only completed runs may be verified")
    if run.get("verification"):
        record = run["verification"]
        require(
            isinstance(record, dict)
            and all(
                isinstance(record.get(k), str)
                and record[k].strip()
                and record[k] not in {"unknown", "not available"}
                for k in ("reviewer", "evidence")
            ),
            "incomplete verification evidence",
        )


def comparable(first, second):
    validate_run(first)
    validate_run(second)
    return all(first[key] == second[key] for key in COMPARISON_KEYS)


def log_run(record, *, tracking_uri):
    """Future MLflow adapter; never silently pretend a run was logged."""
    validate_run(record)
    raise NotImplementedError("MLflow adapter is not implemented or deployed")


def write_run(record, directory):
    """Persist a validated real run record; caller owns lifecycle and evidence."""
    validate_run(record)
    require(bool(re.fullmatch(r"[A-Za-z0-9_-]+", record["run_id"])), "unsafe run ID")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{record['run_id']}.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        immutable = ("run_id", "experiment_id", "git_sha", "config_hash", "manifest_hash", "seed")
        require(
            all(previous[k] == record[k] for k in immutable), "run identity cannot be overwritten"
        )
    fd, name = tempfile.mkstemp(dir=root, suffix=".tmp")
    os.close(fd)
    temporary = Path(name)
    try:
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
