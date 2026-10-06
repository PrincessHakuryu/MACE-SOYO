"""Validate dataset settings and map training heads to checkpoint heads."""

from copy import deepcopy
import os
from pathlib import Path

import yaml


def read_config(config_path=None):
    if config_path is None:
        config_path = os.environ.get("MACE_SOYO_CONFIG") or "config/config.yml"
    with open(config_path, "r", encoding="utf-8") as file:
        return normalize_training_config(yaml.safe_load(file))


def parse_pbc(value):
    """True -> TTT, False -> FFF; preserve three-axis values used internally."""
    if isinstance(value, bool):
        return (value,) * 3
    if isinstance(value, str):
        text = value.strip().upper().replace(" ", "")
        if text in ("TRUE", "FALSE"):
            return (text == "TRUE",) * 3
        if len(text) == 3 and set(text) <= {"T", "F"}:
            return tuple(c == "T" for c in text)
    elif isinstance(value, (list, tuple)):
        if len(value) == 3 and all(isinstance(v, bool) for v in value):
            return tuple(value)
    raise ValueError(f"pbc must be True or False (three-axis PBC is also supported), got {value!r}")


def normalize_training_config(config):
    if not isinstance(config, dict):
        raise ValueError("Training config must be a YAML mapping.")
    config = deepcopy(config)

    model_dtype = config.get("model_dtype", "float32")
    if model_dtype not in ("float32", "float64"):
        raise ValueError(f"model_dtype must be float32 or float64, got {model_dtype!r}")
    config["model_dtype"] = model_dtype

    for key, default in (("num_heads", 1), ("readout_hidden", 64),
                         ("max_correlations", 3)):
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer, got {value!r}")
        config[key] = value

    datasets = config.get("datasets")
    if config.get("train_path") or config.get("valid_path"):
        raise ValueError("Put train_path/valid_path inside datasets entries.")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("datasets must be a non-empty list, with one entry per training head.")
    if not config.get("resume") and len(datasets) != config["num_heads"]:
        raise ValueError(
            f"num_heads={config['num_heads']} does not match the number of datasets "
            f"(training folders)={len(datasets)}. One dataset entry must map to one head."
        )

    names, train_paths = set(), set()
    normalized = []
    for head_id, entry in enumerate(datasets):
        if not isinstance(entry, dict):
            raise ValueError(f"datasets[{head_id}] must be a mapping.")
        entry = deepcopy(entry)
        name = entry["name"]
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError(f"Dataset names must be non-empty and unique, got {name!r}.")
        names.add(name)
        if "finetune_head" in entry:
            choice = entry["finetune_head"]
            if not isinstance(choice, str) or not choice.strip():
                raise ValueError(f"Dataset {name}: finetune_head must be a checkpoint head name.")
        train = entry.get("train_path")
        valid = entry.get("valid_path") or ""
        if not isinstance(train, str) or not train.strip():
            raise ValueError(f"Dataset {name}: train_path must be one .aselmdb folder/file.")
        if not isinstance(valid, str):
            raise ValueError(f"Dataset {name}: valid_path must be one folder/file or empty.")
        if train in train_paths:
            raise ValueError(f"Training folder {train!r} is assigned to more than one head.")
        train_paths.add(train)
        entry.update(name=name, head_id=head_id, train_path=train, valid_path=valid)
        entry["pbc"] = parse_pbc(entry["pbc"])
        if not isinstance(entry.get("e0_yaml_path"), str) or not entry["e0_yaml_path"].strip():
            raise ValueError(f"Dataset {name}: e0_yaml_path is required; prepare E0 before training.")
        normalized.append(entry)

    config["datasets"] = normalized
    config.pop("train_path", None)
    config.pop("valid_path", None)
    config["head_names"] = [d["name"] for d in normalized]
    return config


def align_checkpoint_heads(config, source_kwargs):
    """Resolve finetune sources per target head; resume keeps saved head IDs."""
    config = deepcopy(config)
    source_names = list(source_kwargs["head_names"])
    old_heads = source_kwargs["num_heads"]
    if len(source_names) != old_heads or len(set(source_names)) != old_heads:
        raise ValueError("Checkpoint head_names must contain num_heads unique names.")

    active_names = [entry["name"] for entry in config["datasets"]]
    if config.get("finetune", False):
        # The target model contains exactly the heads declared by datasets.
        config["num_heads"] = len(active_names)
        config["head_names"] = active_names
        for head_id, entry in enumerate(config["datasets"]):
            choice = entry.get("finetune_head")
            if choice is None:
                if entry["name"] in source_names:
                    choice = entry["name"]
                elif old_heads == 1:
                    choice = source_names[0]
                else:
                    raise ValueError(
                        f"New head {entry['name']!r} must set finetune_head to a checkpoint "
                        f"head name; available: {source_names}."
                    )
            if choice not in source_names:
                raise ValueError(f"Unknown finetune head {choice!r}; available: {source_names}.")
            entry["head_id"] = head_id
            entry["finetune_head"] = choice
            entry["finetune_source_head_id"] = source_names.index(choice)
    else:
        # resume
        missing = set(active_names) - set(source_names)
        if missing:
            raise ValueError(f"Unknown training heads {sorted(missing)}; available: {source_names}.")
        config["num_heads"] = old_heads
        config["head_names"] = source_names
        for entry in config["datasets"]:
            entry["head_id"] = source_names.index(entry["name"])
    return config


def resolve_data_path(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return str(path.resolve())
