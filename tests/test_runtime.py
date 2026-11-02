import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from geoid.config import load_yaml
from geoid.training.checkpoints import (
    IDENTITY_KEYS,
    capture_state,
    checkpoint_due,
    restore_checkpoint,
    save_checkpoint,
)
from geoid.training.runtime import precision_context, resolve_environment, runtime_plan

ROOT = Path(__file__).resolve().parents[1]


class StateComponent:
    def __init__(self, state):
        self.state = state

    def state_dict(self):
        return copy.deepcopy(self.state)

    def load_state_dict(self, state):
        self.state = copy.deepcopy(state)


class Backend:
    """Test double for infrastructure control flow; not a model or CUDA implementation."""

    __version__ = "synthetic-test-backend"
    version = SimpleNamespace(cuda="synthetic")
    float16, bfloat16 = "float16", "bfloat16"

    def __init__(self, cuda=False, bf16=False):
        self.cuda = SimpleNamespace(
            is_available=lambda: cuda,
            is_bf16_supported=lambda: bf16,
            get_rng_state_all=lambda: [7] if cuda else [],
            set_rng_state_all=lambda state: None,
            device_count=lambda: int(cuda),
        )
        self.amp = SimpleNamespace(GradScaler=lambda *args, **kwargs: (args, kwargs))
        self.rng = 9

    def get_rng_state(self):
        return self.rng

    def set_rng_state(self, value):
        self.rng = value

    def autocast(self, **kwargs):
        return kwargs

    def save(self, state, path):
        self.saved = copy.deepcopy(state)
        path.write_bytes(b"synthetic checkpoint marker")

    def load(self, path, **kwargs):
        assert kwargs == {"map_location": "cpu", "weights_only": True}
        assert path.is_file()
        return copy.deepcopy(self.saved)


def test_local_runtime_paths_and_cpu(monkeypatch, tmp_path):
    monkeypatch.setenv("GEOID_DATA_ROOT", str(tmp_path / "unicode-data"))
    config = load_yaml(ROOT / "configs/environments/local.yaml")
    result = resolve_environment(config, root=ROOT, backend=None)
    assert result["device"] == "cpu" and result["precision"] == "fp32"
    assert result["paths"]["dataset"] == str(tmp_path / "unicode-data")
    assert result["torch_version"] == "unavailable"


def test_colab_cuda_and_precision():
    config = load_yaml(ROOT / "configs/environments/colab.yaml")
    with pytest.raises(ValueError, match="CUDA requested"):
        resolve_environment(config, root=ROOT)
    backend = Backend(cuda=True)
    result = resolve_environment(config, root=ROOT, backend=backend)
    context, scaler = precision_context(result, backend)
    assert context == {"device_type": "cuda", "dtype": "float16"}
    assert scaler is not None
    config["precision"] = "bf16"
    with pytest.raises(ValueError, match="bf16"):
        resolve_environment(config, root=ROOT, backend=backend)
    result = resolve_environment(config, root=ROOT, backend=Backend(cuda=True, bf16=True))
    assert precision_context(result, Backend(cuda=True, bf16=True))[1] is None


def test_preflight_no_files(tmp_path):
    import shutil

    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    plan = runtime_plan(
        ROOT / "configs/experiments/baseline.yaml",
        ROOT / "configs/environments/local.yaml",
        root=tmp_path,
    )
    assert plan["runtime"]["device"] == "cpu"
    assert len(plan["runtime_config_hash"]) == 64
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before


def test_epoch_cadence():
    assert not checkpoint_due(0, 2)
    assert not checkpoint_due(1, 2)
    assert checkpoint_due(2, 2)
    with pytest.raises(ValueError):
        checkpoint_due(1, 0)


def test_research_hash_includes_referenced_model(tmp_path):
    import shutil

    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    kwargs = dict(
        experiment_path="configs/experiments/baseline.yaml",
        environment_path="configs/environments/local.yaml",
        root=tmp_path,
    )
    first = runtime_plan(**kwargs)
    model_path = tmp_path / "configs/models/resnet_unet.yaml"
    content = model_path.read_text(encoding="utf-8").replace(
        "fusion: early", "fusion: intermediate"
    )
    model_path.write_text(content, encoding="utf-8")
    second = runtime_plan(**kwargs)
    assert first["experiment_config_hash"] != second["experiment_config_hash"]


def test_complete_checkpoint_restore_and_provenance(tmp_path):
    backend = Backend()
    model, optimizer, scheduler, scaler, sampler = [StateComponent({"value": i}) for i in range(5)]
    identity = dict.fromkeys(IDENTITY_KEYS, "synthetic-fixture")
    random.seed(17)
    np.random.seed(17)
    state = capture_state(
        backend=backend,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=3,
        global_step=12,
        sampler_state=sampler.state_dict(),
        identity=identity,
    )
    next_python, next_numpy = random.random(), np.random.random()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(state, path, backend=backend)
    assert not list(tmp_path.glob("*.tmp"))
    model.state = {"value": 99}
    optimizer.state = {"value": 99}
    backend.rng = 0
    kwargs = dict(
        backend=backend,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        sampler=sampler,
        identity=identity,
    )
    cursor = restore_checkpoint(path, **kwargs)
    assert cursor == {"epoch": 3, "global_step": 12}
    assert model.state == {"value": 0} and optimizer.state == {"value": 1}
    assert backend.rng == 9
    assert random.random() == next_python and np.random.random() == next_numpy
    backend.saved["identity"]["manifest_hash"] = "changed"
    model.state = {"value": 100}
    with pytest.raises(ValueError, match="provenance mismatch"):
        restore_checkpoint(path, **kwargs)
    assert model.state == {"value": 100}
    del backend.saved["optimizer"]
    with pytest.raises(ValueError, match="incomplete"):
        restore_checkpoint(path, **kwargs)


def test_notebook_is_unexecuted_setup_only():
    notebook = json.loads((ROOT / "notebooks/colab_train.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    setup_source = "".join(notebook["cells"][3]["source"])
    invocation_source = "".join(notebook["cells"][4]["source"])
    assert '"--python", sys.executable' in setup_source
    assert '"--python",\n    sys.executable' in invocation_source
    assert "--no-checkout" not in "".join(notebook["cells"][2]["source"])
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            source = "".join(cell["source"])
            compile(source, "colab-cell", "exec")
            assert ".backward(" not in source and "torch.save(" not in source
