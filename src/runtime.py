import argparse
import logging
import os
import pickle
import random
import tempfile
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(paths, overrides):
    options = {}
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        if not isinstance(values, dict):
            raise ValueError(f"Expected a YAML mapping in {path}")
        options.update(values)
    options.update(
        {key: value for key, value in overrides.items() if value is not None}
    )
    return argparse.Namespace(**options)


def load_pickle(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def seed_randomness(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def create_run(config, stage, output_dir=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = Path(output_dir) if output_dir else Path("runs", stage, timestamp)
    path.mkdir(parents=True, exist_ok=False)
    with (path / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(vars(config), handle, sort_keys=False)
    return path


def setup_logger(path, filename):
    logger = logging.getLogger(f"target_to_route.{path.resolve()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    for handler in (logging.FileHandler(path / filename), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def read_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Expected a PyTorch state dictionary in {path}")
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, Mapping) or "fp_pos_weight" not in state:
        raise ValueError(
            f"{path} is missing the original TRACE state dictionary (fp_pos_weight)"
        )
    return checkpoint, state


def model_from_state(config, state, path):
    from .model import Target2RouteGen

    model = Target2RouteGen(config, fp_pos_weight=state["fp_pos_weight"])
    if "is_bimol" in state and not torch.equal(model.is_bimol, state["is_bimol"]):
        raise ValueError(
            f"Template arities/order in {config.template_csv_path} differ from {path}"
        )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            f"Checkpoint {path} does not match the model configuration.\n{exc}"
        ) from exc
    return model.to(config.device)


def model_from_checkpoint(config, path):
    checkpoint, state = read_checkpoint(path)
    model = model_from_state(config, state, path)
    metadata = {
        key: checkpoint[key]
        for key in ("epoch", "episode", "val_acc")
        if key in checkpoint
    }
    return model, metadata


def save_torch(data, path):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(data, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
