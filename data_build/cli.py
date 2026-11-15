import argparse
import json
import logging
import math
from pathlib import Path


def positive(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("expected an integer >= 1")
    return value


def seed(text):
    value = int(text)
    if not 0 <= value < 2**32:
        raise argparse.ArgumentTypeError("expected a seed in [0, 2**32)")
    return value


def parser():
    cli = argparse.ArgumentParser(description="Prepare and verify TRACE datasets")
    stages = cli.add_subparsers(dest="stage", required=True)

    binding = stages.add_parser("filter-bindingdb", help="Export active, single-chain ligand-target pairs")
    binding.add_argument("--bindingdb", type=Path, required=True)
    binding.add_argument("--output", type=Path, required=True)
    binding.add_argument("--threshold-nm", type=float, default=1000.0)
    binding.add_argument("--exclude-fasta", type=Path, action="append", default=[])
    binding.add_argument("--chunk-size", type=positive, default=25000)

    stock = stages.add_parser("build-stock", help="Build the AiZynthFinder stock from ZINC20 SMILES")
    stock.add_argument("--smiles", type=Path, nargs="+", required=True)
    stock.add_argument("--output", type=Path, required=True)

    faiss = stages.add_parser("build-faiss", help="Build the matching FAISS index and SMILES map")
    faiss.add_argument("--smiles", type=Path, nargs="+", required=True)
    faiss.add_argument("--output", type=Path, required=True)
    faiss.add_argument("--flat", action="store_true", help="Exact L2 index for small test libraries")
    faiss.add_argument("--nlist", type=positive, default=8192)
    faiss.add_argument("--pq-m", type=positive, default=128)
    faiss.add_argument("--pq-bits", type=positive, default=8)
    faiss.add_argument("--train-size", type=positive, default=500000)
    faiss.add_argument("--batch-size", type=positive, default=20000)
    faiss.add_argument("--threads", type=positive, default=1)
    faiss.add_argument("--seed", type=seed, default=2026)

    retro = stages.add_parser("retrosynthesis", help="Search AiZynthFinder and write one JSONL route shard")
    retro.add_argument("--pairs", type=Path, required=True)
    retro.add_argument("--aizynth-config", type=Path, required=True)
    retro.add_argument("--output", type=Path, required=True)
    retro.add_argument("--policy", default="uspto")
    retro.add_argument("--stock", default="ZINC20")
    retro.add_argument("--filter")
    retro.add_argument("--max-depth", type=positive, default=6)
    retro.add_argument("--max-routes", type=positive, default=10)
    retro.add_argument("--iterations", type=positive, default=100)
    retro.add_argument("--time-limit", type=positive, default=120)
    retro.add_argument("--workers", type=positive, default=1)
    retro.add_argument("--parts", type=positive, default=1)
    retro.add_argument("--part", type=positive, default=1)
    retro.add_argument("--limit", type=positive, help="Process only the first N pairs")
    retro.add_argument("--seed", type=seed, default=2026)

    build = stages.add_parser("build-supervised", help="Build model-ready routes, templates, splits and embeddings")
    build.add_argument("--routes", type=Path, nargs="+", required=True)
    build.add_argument("--blocks-dir", type=Path)
    templates = build.add_mutually_exclusive_group(required=True)
    templates.add_argument("--uspto-templates", type=Path)
    templates.add_argument("--frozen-templates", type=Path, help="Copy an existing TRACE table in original row order")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--exclude-fasta", type=Path, action="append", default=[])
    build.add_argument("--registry-fasta", type=Path, action="append", default=[],
                       help="Precompute additional target features without adding supervised examples")
    build.add_argument("--min-steps", type=positive, default=3)
    build.add_argument("--max-steps", type=positive, default=6)
    build.add_argument("--start-max-mw", type=float, default=350.0)
    build.add_argument("--template-coverage", type=float, default=0.90)
    build.add_argument("--similarity", type=float, default=0.75)
    build.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1),
                       metavar=("TRAIN", "VALID", "TEST"))
    build.add_argument("--seed", type=seed, default=2026)
    build.add_argument("--esm-batch-size", type=positive, default=8)
    build.add_argument("--esm2-device", default="cpu")
    build.add_argument("--raw-embeddings", type=Path, help="Optional dictionary of raw ESM-2 CLS tensors")
    build.add_argument("--reuse-embeddings", action="store_true",
                       help="Keep known vectors and center new ones using the existing calibration")

    target = stages.add_parser("register-target", help="Register an individual target or a manifest of targets")
    source = target.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--fasta", type=Path)
    source.add_argument("--sequence")
    target.add_argument("--target-id", type=int)
    target.add_argument("--pdb-id", help="Optional structure provenance to store with the target")
    target.add_argument("--chain", help="Optional receptor chain to store with the target")
    target.add_argument("--receptor", type=Path)
    target.add_argument("--box", type=float, nargs=3, metavar=("X", "Y", "Z"))
    target.add_argument("--start-mol", action="append", default=[])
    target.add_argument("--global-rl", action="store_true")
    target.add_argument("--esm2-device", default="cpu")

    verify = stages.add_parser("verify", help="Validate files with the same schemas used by the model")
    verify.add_argument("--manifest", type=Path, help="Check recorded SHA-256 digests")
    verify.add_argument("--allow-extended-bank", action="store_true",
                        help="Allow the feature bank to have changed after target registration")
    verify.add_argument("--require-global-rl", action="store_true")
    verify.add_argument("--hash-files", action="store_true", help="Record hashes of the actual model data files")

    for stage in (faiss, build, target, verify):
        stage.add_argument("--model-config", help="Original model YAML")
        stage.add_argument("--config", action="append", default=[], help="Repeatable model data-path YAML overlay")
    return cli


def _run(options):
    if options.stage == "filter-bindingdb":
        from .inputs import filter_bindingdb

        if not math.isfinite(options.threshold_nm) or options.threshold_nm <= 0:
            raise ValueError("--threshold-nm must be positive and finite")
        return filter_bindingdb(
            options.bindingdb, options.output, threshold=options.threshold_nm,
            exclude_fasta=options.exclude_fasta, chunk_size=options.chunk_size,
        )
    if options.stage == "build-stock":
        from .building_blocks import build_stock

        return build_stock(options.smiles, options.output)
    if options.stage == "retrosynthesis":
        from .retrosynthesis import retrosynthesize

        return retrosynthesize(options)

    from target_to_route.cli import GLOBAL_RL, SUPERVISED
    from target_to_route.runtime import load_config

    paths = [SUPERVISED]
    if options.model_config:
        paths.append(options.model_config)
    paths.extend([GLOBAL_RL, *options.config])
    config = load_config(paths, {"esm2_device": getattr(options, "esm2_device", None)})
    if options.stage == "build-faiss":
        from .building_blocks import build_faiss

        return build_faiss(config, options)
    if options.stage == "build-supervised":
        from .supervised import build_supervised

        return build_supervised(config, options)
    if options.stage == "register-target":
        from .targets import register_targets

        return register_targets(config, options)

    from .verify import verify_dataset

    return verify_dataset(config, options)


def main():
    cli = parser()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        print(json.dumps(_run(cli.parse_args()), indent=2, ensure_ascii=False))
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, ImportError) as exc:
        cli.exit(2, f"Error: {exc}\n")
