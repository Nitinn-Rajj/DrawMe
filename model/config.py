"""Configuration loader for DrawMe training profiles."""

import copy
import json
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(MODEL_DIR, "configs")

DEFAULT_CONFIG = {
    "paths": {
        "data_dir": os.path.join(BASE_DIR, "data"),
        "save_dir": os.path.join(MODEL_DIR, "saved"),
    },
    "runtime": {
        "seed": 42,
        "deterministic": False,
        "mixed_precision": False,
        "gpu": {
            "enable": True,
            "memory_growth": True,
        },
    },
    "data": {
        "img_size": 28,
        "samples_per_class": 12000,
        "categories": [
            "cloud", "sun", "tree", "car", "fish",
            "cat", "dog", "house", "star", "flower",
            "bird", "bicycle", "guitar", "moon", "hat",
        ],
    },
    "augmentation": {
        "enabled": True,
        "rotation": 0.08,
        "translation_height": 0.1,
        "translation_width": 0.1,
        "zoom": 0.1,
    },
    "model": {
        "name": "DrawMe_CNN_v3",
        "conv_filters": [32, 64, 128],
        "kernel_size": [3, 3],
        "dense_units": [256],
        "dropout": 0.35,
        "learning_rate": 0.001,
    },
    "training": {
        "epochs": 30,
        "batch_size": 128,
        "test_split": 0.2,
        "validation_split": 0.1,
        "lr_factor": 0.5,
        "lr_patience": 3,
        "min_lr": 1e-6,
        "early_stopping_patience": 7,
    },
}


def _deep_merge(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_training_config(profile="small_machine", override_path=None):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    sources = ["DEFAULT_CONFIG"]

    profile_path = os.path.join(CONFIG_DIR, f"{profile}.json")
    if os.path.exists(profile_path):
        profile_cfg = _read_json(profile_path)
        _deep_merge(cfg, profile_cfg)
        sources.append(profile_path)

    if override_path:
        abs_override = os.path.abspath(override_path)
        if not os.path.exists(abs_override):
            raise FileNotFoundError(f"Config override file not found: {abs_override}")
        override_cfg = _read_json(abs_override)
        _deep_merge(cfg, override_cfg)
        sources.append(abs_override)

    # Normalize paths to absolute paths for consistent behavior.
    cfg["paths"]["data_dir"] = os.path.abspath(cfg["paths"]["data_dir"])
    cfg["paths"]["save_dir"] = os.path.abspath(cfg["paths"]["save_dir"])

    return cfg, sources
