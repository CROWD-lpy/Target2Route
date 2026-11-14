import argparse
import hashlib
import importlib.util
import sys
import tempfile
from pathlib import Path

import torch

from .data import RouteBatch
from .runtime import model_from_checkpoint


def _compare_original(model, config, original):
    module_name = "_target_to_route_original_model"
    package = Path(original) / "model"
    init = package / "__init__.py"
    if not init.is_file():
        raise FileNotFoundError(f"Expected the original TRACE source at {init}")
    spec = importlib.util.spec_from_file_location(
        module_name, init, submodule_search_locations=[str(package)]
    )
    legacy = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = legacy
    spec.loader.exec_module(legacy)
    legacy_config = argparse.Namespace(**vars(config))
    with tempfile.TemporaryDirectory() as tmp:
        weights = model.fp_pos_weight.detach().cpu()
        if weights.shape == (1,):
            legacy_config.fp_pos_weights = weights.item()
        else:
            legacy_config.fp_pos_weights = str(Path(tmp) / "fp_pos_weights.pt")
            torch.save(
                {f"{config.fp_type.lower()}_{config.fp_dim}": weights},
                legacy_config.fp_pos_weights,
            )
        previous = legacy.Target2RouteGen(legacy_config).to(config.device)
        previous.load_state_dict(model.state_dict(), strict=True)
    previous.eval()
    model.eval()

    generator = torch.Generator().manual_seed(2026)
    batch = RouteBatch(
        target_features=torch.randn(
            2, config.esm2_dim + config.pocket_dim, generator=generator
        ),
        start_features=torch.randint(
            0, 2, (2, config.fp_dim), generator=generator
        ).float(),
        seq_types=torch.tensor([[1, 0, 2], [0, 2, -100]]),
        seq_templates=torch.zeros(2, 3, dtype=torch.long),
        seq_fps=torch.randint(0, 2, (2, 3, config.fp_dim), generator=generator).float(),
    )
    with torch.no_grad():
        for expected, actual in zip(previous(batch), model(batch)):
            if expected is None or actual is None:
                if expected is not actual:
                    raise AssertionError(
                        "The original and refactored models returned different fingerprint outputs"
                    )
            else:
                torch.testing.assert_close(expected, actual, rtol=1e-6, atol=1e-7)
        torch.manual_seed(2026)
        expected = previous.compute_loss(batch, mode="test")
        torch.manual_seed(2026)
        actual = model.compute_loss(batch, mode="test")
        torch.testing.assert_close(expected, actual, rtol=1e-6, atol=1e-7)


def check_checkpoint(config, path, original=None):
    model, metadata = model_from_checkpoint(config, path)
    template_file = Path(config.template_csv_path)
    result = {
        "checkpoint": str(path),
        "strict_load": True,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "template_csv": str(template_file),
        "template_sha256": hashlib.sha256(template_file.read_bytes()).hexdigest(),
        **metadata,
    }
    if original is not None:
        _compare_original(model, config, original)
        result["legacy_forward_and_loss_match"] = True
    return result
