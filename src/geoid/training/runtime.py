"""Environment resolution and optional CUDA/AMP context, independent of models."""

import importlib
import os
import re
from contextlib import nullcontext
from pathlib import Path

from geoid.config import load_yaml, require
from geoid.data.manifest import canonical_hash


def torch_backend():
    try:
        return importlib.import_module("torch")
    except ImportError:
        return None


def expand_path(value, root):
    require(isinstance(value, str) and bool(value), "runtime path must be nonempty")

    def substitute(match):
        name, fallback = match.group(1), match.group(2)
        result = os.environ.get(name, fallback)
        require(bool(result), f"set environment variable {name}")
        return result

    value = re.sub(r"\$\{([A-Za-z_][A-Za-z_0-9]*)(?::-([^}]*))?\}", substitute, value)
    require("${" not in value, "unresolved path variable")
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else Path(root) / path).resolve())


def validate_environment(config):
    require(config.get("environment") in {"local", "colab"}, "invalid environment")
    require(config.get("device") in {"auto", "cpu", "cuda"}, "invalid device")
    require(config.get("precision") in {"auto", "fp32", "fp16", "bf16"}, "invalid precision")
    paths = config.get("paths")
    require(
        isinstance(paths, dict)
        and all(
            isinstance(paths.get(k), str) and paths[k] for k in ("dataset", "checkpoints", "runs")
        ),
        "missing runtime paths",
    )
    require(
        type(config.get("checkpoint_every_epochs")) is int
        and config["checkpoint_every_epochs"] > 0,
        "checkpoint cadence must be positive",
    )
    require(
        config.get("resume") is None or isinstance(config["resume"], str), "invalid resume path"
    )
    tracking = config.get("tracking")
    require(
        isinstance(tracking, dict) and tracking.get("backend") in {"json", "mlflow"},
        "invalid tracking backend",
    )


def resolve_environment(config, *, root, backend=None):
    validate_environment(config)
    available = bool(backend is not None and backend.cuda.is_available())
    device = config["device"]
    if device == "auto":
        device = "cuda" if available else "cpu"
    require(device != "cuda" or available, "CUDA requested but unavailable; select a GPU runtime")
    precision = config["precision"]
    if precision == "auto":
        precision = "fp16" if device == "cuda" else "fp32"
    require(
        device == "cuda" or precision == "fp32", "mixed precision requires CUDA in this adapter"
    )
    if precision == "bf16":
        require(backend.cuda.is_bf16_supported(), "this GPU does not support bf16")
    return {
        **config,
        "paths": {k: expand_path(v, root) for k, v in config["paths"].items()},
        "resume": expand_path(config["resume"], root) if config.get("resume") else None,
        "device": device,
        "precision": precision,
        "torch_version": str(backend.__version__) if backend is not None else "unavailable",
        "cuda_version": str(backend.version.cuda) if backend is not None else "unavailable",
    }


def precision_context(runtime, backend):
    """Return autocast context and scaler for a future approved trainer."""
    require(backend is not None, "PyTorch is required for training")
    precision = runtime["precision"]
    if precision == "fp32":
        return nullcontext(), None
    dtype = backend.float16 if precision == "fp16" else backend.bfloat16
    context = backend.autocast(device_type="cuda", dtype=dtype)
    scaler = backend.amp.GradScaler("cuda", enabled=True) if precision == "fp16" else None
    return context, scaler


def runtime_plan(experiment_path, environment_path, *, root, backend=None, resume=None):
    root = Path(root).resolve()
    experiment = load_yaml(root / experiment_path)
    environment = load_yaml(root / environment_path)
    research = {"experiment": experiment}
    for key in ("data_config", "model_config"):
        relative = experiment.get(key)
        require(isinstance(relative, str), f"missing {key}")
        referenced = (root / relative).resolve()
        require(referenced.is_relative_to(root / "configs"), "config reference escapes configs")
        research[key] = load_yaml(referenced)
    if resume:
        environment["resume"] = str(resume)
    runtime = resolve_environment(environment, root=root, backend=backend)
    return {
        "status": "dry-run; research trainer is not implemented",
        "experiment_config_hash": canonical_hash(research),
        "runtime_config_hash": canonical_hash(runtime),
        "runtime": runtime,
        "research_approval": "required; all bootstrap ADRs remain proposed",
    }
