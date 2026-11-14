from contextlib import nullcontext

import torch

from .chemistry.building_blocks import BuildingBlockIndex
from .chemistry.docking import docking_pool
from .chemistry.environment import BatchChemEnv
from .chemistry.routes import (
    MIN_REACTIONS,
    SUCCESSFUL_ENDS,
    route_record,
    write_routes,
)
from .rollout import rollout
from .runtime import model_from_checkpoint, setup_logger


def generate(
    config, checkpoint, run_dir, target, features, *, num_mols, start_mol, docking
):
    logger = setup_logger(run_dir, "generate.log")
    model, _ = model_from_checkpoint(config, checkpoint)
    model.eval()
    building_blocks = BuildingBlockIndex(config)
    env = BatchChemEnv(config)
    logger.info("Preparing protein features for target %s", target.id)
    vector = features.get(target.sequence)
    batch_features = vector.to(config.device).unsqueeze(0).expand(config.batch_size, -1)
    targets = [target] * config.batch_size
    starts = [start_mol] * config.batch_size
    mode = "scaffold" if start_mol else "denovo"

    context = nullcontext()
    if docking:
        from .training.rewards import build_catalog, score_routes

        catalog = build_catalog()
        context = docking_pool(config)

    success_path = run_dir / "success_routes.csv"
    rewarded_path = run_dir / "rewarded_routes.csv"
    write_routes(success_path)
    if docking:
        write_routes(rewarded_path)
    seen_success, seen_rewarded = set(), set()
    required_set = seen_rewarded if docking else seen_success
    batch_index = 0

    with torch.no_grad(), context as dock:
        while len(required_set) < num_mols:
            if config.max_batches is not None and batch_index >= config.max_batches:
                logger.error(
                    "Generated %d/%d unique molecules after %d batches. Partial results are in %s",
                    len(required_set),
                    num_mols,
                    batch_index,
                    run_dir,
                )
                return False
            batch_index += 1
            _, terminations = rollout(
                model, env, building_blocks, config, batch_features, starts
            )
            scores = (
                score_routes(
                    config,
                    env.routes,
                    terminations,
                    targets,
                    dock,
                    env.fingerprints,
                    catalog,
                )
                if docking
                else None
            )
            fates = scores.fates if docking else terminations
            success, rewarded = [], []
            for i, route in enumerate(env.routes):
                if len(route.steps) < MIN_REACTIONS or fates[
                    i
                ] not in SUCCESSFUL_ENDS | {"threshold_fail"}:
                    continue
                record = route_record(
                    batch_index, mode, target, route, terminations[i], scores, i
                )
                smiles = route.product
                if smiles not in seen_success and (
                    docking or len(seen_success) < num_mols
                ):
                    seen_success.add(smiles)
                    success.append(record)
                if (
                    docking
                    and fates[i] in SUCCESSFUL_ENDS
                    and scores.rewards[i] > 0
                    and smiles not in seen_rewarded
                    and len(seen_rewarded) < num_mols
                ):
                    seen_rewarded.add(smiles)
                    rewarded.append(record)
            if success:
                write_routes(success_path, success)
            if rewarded:
                write_routes(rewarded_path, rewarded)
            logger.info(
                "Batch %d | valid %d | requested count %d/%d",
                batch_index,
                len(seen_success),
                len(required_set),
                num_mols,
            )

    logger.info("Routes written to %s", run_dir)
    return True
