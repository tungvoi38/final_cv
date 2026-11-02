"""Full-state epoch-boundary checkpoint utilities; no training loop or model code."""

import os
import random
import tempfile
from pathlib import Path

import numpy as np

from geoid.config import require

IDENTITY_KEYS = (
    "git_sha",
    "architecture_version",
    "config_hash",
    "manifest_hash",
    "preprocessing_version",
    "evaluation_protocol",
    "precision",
    "torch_version",
)


def checkpoint_due(completed_epochs, every_epochs):
    require(type(every_epochs) is int and every_epochs > 0, "invalid checkpoint cadence")
    require(type(completed_epochs) is int and completed_epochs >= 0, "invalid epoch count")
    return completed_epochs > 0 and completed_epochs % every_epochs == 0


def capture_state(
    *, backend, model, optimizer, scheduler, scaler, epoch, global_step, sampler_state, identity
):
    """Call after a completed epoch, with next epoch index and complete sampler state."""
    require(
        all(isinstance(identity.get(k), str) and identity[k] for k in IDENTITY_KEYS),
        "checkpoint identity incomplete",
    )
    require(
        type(epoch) is int and epoch >= 0 and type(global_step) is int and global_step >= 0,
        "invalid training cursor",
    )
    require(isinstance(sampler_state, dict), "sampler state must be explicit")
    numpy_state = np.random.get_state()
    return {
        "schema_version": 1,
        "boundary": "epoch",
        "identity": dict(identity),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "sampler": sampler_state,
        "rng": {
            "python": random.getstate(),
            "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "torch": backend.get_rng_state(),
            "cuda": backend.cuda.get_rng_state_all() if backend.cuda.is_available() else [],
        },
    }


def save_checkpoint(state, path, *, backend):
    """Use a same-directory temporary file and atomic replacement when FS supports it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(descriptor)
    temporary = Path(name)
    try:
        backend.save(state, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def restore_checkpoint(path, *, backend, model, optimizer, scheduler, scaler, sampler, identity):
    """Restore trusted full state; mismatched provenance fails before model mutation."""
    state = backend.load(Path(path), map_location="cpu", weights_only=True)
    required = {
        "schema_version",
        "boundary",
        "identity",
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "epoch",
        "global_step",
        "sampler",
        "rng",
    }
    require(isinstance(state, dict) and required <= state.keys(), "incomplete checkpoint")
    require(state["schema_version"] == 1 and state["boundary"] == "epoch", "unsupported checkpoint")
    require(
        all(identity.get(k) and identity[k] == state["identity"].get(k) for k in IDENTITY_KEYS),
        "resume provenance mismatch; start a new experiment instead",
    )
    require((scheduler is None) == (state["scheduler"] is None), "scheduler mismatch")
    require((scaler is None) == (state["scaler"] is None), "AMP scaler mismatch")
    require(
        isinstance(state["rng"], dict)
        and {"python", "numpy", "torch", "cuda"} <= state["rng"].keys(),
        "incomplete RNG state",
    )
    require(
        type(state["epoch"]) is int
        and state["epoch"] >= 0
        and type(state["global_step"]) is int
        and state["global_step"] >= 0,
        "invalid resume cursor",
    )
    cuda_states = state["rng"]["cuda"]
    require(
        len(cuda_states) == (backend.cuda.device_count() if backend.cuda.is_available() else 0),
        "CUDA topology differs; exact resume cannot be claimed",
    )
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None:
        scaler.load_state_dict(state["scaler"])
    sampler.load_state_dict(state["sampler"])
    rng = state["rng"]
    random.setstate(rng["python"])
    n = rng["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    backend.set_rng_state(rng["torch"])
    if cuda_states:
        backend.cuda.set_rng_state_all(cuda_states)
    return {"epoch": state["epoch"], "global_step": state["global_step"]}
