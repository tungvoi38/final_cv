"""Validate bootstrap configuration without starting research computations."""

from pathlib import Path

import yaml

EXPERIMENTS = {
    "baseline": "none",
    "classmix": "classmix",
    "mixstyle": "mixstyle",
    "coral": "coral",
    "class_coral": "class_coral",
}


def load_yaml(path):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected mapping")
    return value


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_configs(root):
    from geoid.training.runtime import validate_environment

    root = Path(root)
    expected = ["data/geoid.yaml", "models/resnet_unet.yaml", "models/terramind.yaml"]
    expected += [f"experiments/{name}.yaml" for name in EXPERIMENTS]
    expected += ["environments/local.yaml", "environments/colab.yaml"]
    for name in expected:
        require((root / "configs" / name).is_file(), f"missing configs/{name}")
    for path in (root / "configs").rglob("*.yaml"):
        cfg = load_yaml(path)
        require(cfg.get("schema_version") == 1, f"{path}: schema_version must be 1")
        require(cfg.get("status") == "proposed", f"{path}: bootstrap config must be proposed")
        require(cfg.get("contract_version") == "0.1-proposed", f"{path}: contract mismatch")
        if path.parent.name == "environments":
            validate_environment(cfg)
    data = load_yaml(root / "configs/data/geoid.yaml")
    require(data.get("modalities") == {"sar": 4, "optical": 12, "dem": 1}, "channel mismatch")
    require(
        data.get("class_mapping") == {0: "background", 1: "permanent_water", 2: "flooded_water"},
        "class mapping mismatch",
    )
    require(data.get("num_classes") == 3 and data.get("ignore_index") == 255, "label mismatch")
    for key, value in {"chip_size": 256, "train_stride": 128, "eval_stride": 256}.items():
        require(type(data.get(key)) is int and data[key] == value, f"{key}: proposal mismatch")
    require(
        isinstance(data.get("normalization"), dict)
        and data["normalization"].get("fit_split") == "train",
        "statistics must be train-only",
    )
    require(data.get("split_protocol") == "official_pending_event_audit", "split protocol mismatch")
    for key in ("source", "dataset_name", "raw_root", "processed_root", "missing_data_policy"):
        require(isinstance(data.get(key), str) and bool(data[key]), f"missing {key}")
    acquisition = data.get("acquisition")
    require(
        isinstance(acquisition, dict) and isinstance(acquisition.get("files"), list),
        "acquisition requires an explicit files list",
    )
    require(
        type(acquisition.get("sample_max_bytes")) is int and acquisition["sample_max_bytes"] > 0,
        "invalid sample cap",
    )
    for name in ("resnet_unet", "terramind"):
        model = load_yaml(root / f"configs/models/{name}.yaml")
        require(model.get("model") == name and model.get("num_classes") == 3, "model mismatch")
        require(model.get("input_modalities") == ["sar", "optical", "dem"], "modality mismatch")
        for key in ("backbone", "pretraining_source", "fusion", "adapters"):
            require(bool(model.get(key)), f"{name}: missing {key}")
        require("checkpoint" in model, f"{name}: checkpoint must be explicit, possibly null")
    for name, method in EXPERIMENTS.items():
        cfg = load_yaml(root / f"configs/experiments/{name}.yaml")
        require(
            cfg.get("experiment") == name and cfg.get("dg_method") == method,
            f"{name}: experiment/method mismatch",
        )
        for key, directory in (("data_config", "data"), ("model_config", "models")):
            relative = cfg.get(key)
            require(isinstance(relative, str), f"{name}: missing {key}")
            target = (root / relative).resolve()
            require(
                target.is_relative_to((root / "configs" / directory).resolve())
                and target.is_file(),
                f"{name}: invalid config reference {key}",
            )
        require(type(cfg.get("seed")) is int and cfg["seed"] >= 0, f"{name}: invalid seed")
        require(
            isinstance(cfg.get("loss"), dict) and cfg["loss"].get("segmentation") == "ce_plus_dice",
            f"{name}: invalid loss",
        )
        require(isinstance(cfg.get("hyperparameters"), dict), f"{name}: missing hyperparameters")
        require(
            cfg.get("evaluation_protocol") == "event_macro_v0_proposed"
            and cfg.get("selection_metric") == "val_event_macro_flood_iou",
            f"{name}: evaluation protocol mismatch",
        )
