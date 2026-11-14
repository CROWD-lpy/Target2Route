import argparse
import json
from pathlib import Path

SUPERVISED = "configs/supervised.yaml"
GLOBAL_RL = "configs/rl_global.yaml"
TARGET_RL = "configs/rl_target.yaml"


def positive(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("expected an integer >= 1")
    return value


def parser():
    command = argparse.ArgumentParser(description="TRACE protein-conditioned synthesis")
    subcommands = command.add_subparsers(dest="command", required=True)

    train = subcommands.add_parser(
        "train-sl", help="Train or resume the supervised model"
    )
    train.add_argument(
        "--epochs", type=positive, dest="num_epoch", help="Total epochs to complete"
    )
    train.add_argument(
        "--resume", metavar="CHECKPOINT", help="Continue after a saved supervised epoch"
    )
    evaluate = subcommands.add_parser(
        "evaluate-sl", help="Evaluate teacher-forced loss and FAISS retrieval"
    )
    evaluate.add_argument("--split", choices=("train", "valid", "test"), default="test")
    global_rl = subcommands.add_parser(
        "train-global", help="Train global RL starting from a supervised checkpoint"
    )
    target_rl = subcommands.add_parser(
        "train-target", help="Train single-target RL starting from a global checkpoint"
    )
    generation = subcommands.add_parser(
        "generate", help="Generate molecules using the truncated synthesis policy"
    )
    check = subcommands.add_parser(
        "check-checkpoint",
        help="Strictly load a checkpoint and check the template bank",
    )
    check.add_argument(
        "--original",
        help="Unpacked original TRACE directory for a forward and loss comparison",
    )

    for entry in (train, evaluate, global_rl, target_rl, generation, check):
        entry.add_argument(
            "--model-config",
            help="Original model YAML to apply over configs/supervised.yaml",
        )
        entry.add_argument(
            "--config",
            action="append",
            help="YAML overlay. Repeat in order to apply several.",
        )
        entry.add_argument("--device", default="cpu" if entry is check else None)
        entry.add_argument("--seed", type=int)
        if entry is not train:
            entry.add_argument(
                "--checkpoint",
                required=True,
                help="Path to the original or newly trained .pt checkpoint",
            )
        if entry is not check:
            entry.add_argument("--batch-size", type=positive)
        if entry in (train, evaluate, generation):
            entry.add_argument(
                "--max-batches",
                type=positive,
                help="Stop each dataset pass or generation after this limit",
            )
        if entry in (train, global_rl, target_rl, generation):
            entry.add_argument(
                "--output-dir",
                help="New output directory. Defaults to a timestamped folder in runs/.",
            )
        if entry in (global_rl, target_rl):
            entry.add_argument("--episodes", type=positive, dest="num_episodes")
        if entry in (global_rl, target_rl, generation):
            entry.add_argument("--docking-workers", type=positive, dest="max_workers")
            entry.add_argument(
                "--esm2-device", help="Device for computing missing ESM-2 embeddings"
            )

    for entry in (target_rl, generation):
        entry.add_argument(
            "--target-id", type=int, help="Numeric target ID in the target index"
        )
        source = entry.add_mutually_exclusive_group()
        source.add_argument(
            "--sequence", help="Protein sequence using one-letter amino acid codes"
        )
        source.add_argument("--fasta", type=Path, help="Single-record FASTA file")
        entry.add_argument("--receptor", help="Docking receptor in PDBQT format")
        entry.add_argument(
            "--box",
            type=float,
            nargs=3,
            metavar=("X", "Y", "Z"),
            help="Docking center in Å",
        )
        entry.add_argument(
            "--vina-base", type=float, help="Override the Vina reward baseline"
        )
    generation.add_argument(
        "--num-mols",
        type=positive,
        help="Number of unique molecules requested (default: 5000)",
    )
    generation.add_argument(
        "--no-docking",
        action="store_true",
        default=None,
        help="Write routes using the protein sequence alone",
    )
    generation.add_argument("--start-mol", help="Optional starting molecule SMILES")
    return command


def _resolve_target(args, config, targets, *, docking):
    sequence = args.fasta.read_text(encoding="utf-8") if args.fasta else args.sequence
    has_new_target = sequence is not None or args.target_id is not None
    if not has_new_target:
        sequence = getattr(config, "target_sequence", None)
    target = targets.resolve(
        target_id=args.target_id
        if has_new_target
        else getattr(config, "target_id", None),
        sequence=sequence,
        receptor=args.receptor
        if has_new_target
        else args.receptor or getattr(config, "target_receptor", None),
        box=args.box
        if has_new_target
        else args.box or getattr(config, "target_box", None),
        docking=docking,
    )
    config.target_id = target.id if isinstance(target.id, int) else None
    config.target_sequence = target.sequence
    config.target_receptor = (
        str(target.receptor) if target.receptor is not None else None
    )
    config.target_box = list(target.box) if target.box is not None else None
    return target


def _run(args):
    import torch

    from .runtime import create_run, load_config, seed_randomness

    command = args.command
    defaults = {
        "train-global": [GLOBAL_RL],
        "train-target": [GLOBAL_RL, TARGET_RL],
        "generate": [GLOBAL_RL],
    }
    option_names = (
        "device",
        "seed",
        "batch_size",
        "num_epoch",
        "num_episodes",
        "max_workers",
        "esm2_device",
        "max_batches",
        "vina_base",
        "checkpoint",
    )
    overrides = {name: getattr(args, name, None) for name in option_names}
    paths = [SUPERVISED]
    if args.model_config:
        paths.append(args.model_config)
    paths.extend(defaults.get(command, []))
    paths.extend(args.config or [])
    config = load_config(paths, overrides)
    config.max_batches = getattr(config, "max_batches", None)

    if torch.device(config.device).type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA is unavailable. Select --device cpu or install PyTorch for your CUDA environment."
        )
    if config.seed is not None and not 0 <= config.seed < 2**32:
        raise ValueError("Use a seed in [0, 2**32)")
    if config.max_batches is not None and config.max_batches < 1:
        raise ValueError("max_batches must be at least 1")
    if command.startswith("train-") and config.batch_size < 2:
        raise ValueError("Training requires a batch size of at least 2")
    if (
        command == "train-sl"
        and min(config.num_epoch, config.save_top_k, config.log_interval) < 1
    ):
        raise ValueError("num_epoch, save_top_k, and log_interval must be at least 1")
    if (
        command in {"train-global", "train-target"}
        and min(config.num_episodes, config.save_freq) < 1
    ):
        raise ValueError("num_episodes and save_freq must be at least 1")
    seed_randomness(config.seed)

    if command == "check-checkpoint":
        from .checkpoint_check import check_checkpoint

        print(
            json.dumps(
                check_checkpoint(config, args.checkpoint, args.original), indent=2
            )
        )
    elif command == "evaluate-sl":
        from .training.supervised import evaluate_supervised

        print(
            json.dumps(
                evaluate_supervised(config, args.checkpoint, args.split), indent=2
            )
        )
    elif command == "train-sl":
        from .training.supervised import train_supervised

        train_supervised(config, args.output_dir, resume=args.resume)
    else:
        from .targets import TargetData

        targets = TargetData(config)
        if command == "generate":
            from .generation import generate

            config.no_docking = (
                args.no_docking
                if args.no_docking is not None
                else getattr(config, "no_docking", False)
            )
            config.num_mols = (
                args.num_mols
                if args.num_mols is not None
                else getattr(config, "num_mols", 5000)
            )
            config.start_mol = (
                args.start_mol
                if args.start_mol is not None
                else getattr(config, "start_mol", "")
            )
            if config.num_mols < 1:
                raise ValueError("num_mols must be at least 1")
            target = _resolve_target(
                args, config, targets, docking=not config.no_docking
            )
            run = create_run(config, "generation", args.output_dir)
            if not generate(
                config,
                args.checkpoint,
                run,
                target,
                targets.features,
                num_mols=config.num_mols,
                start_mol=config.start_mol,
                docking=not config.no_docking,
            ):
                raise SystemExit(1)
        else:
            from .training.reinforcement import RLTrainer

            if command == "train-target":
                pool = [_resolve_target(args, config, targets, docking=True)]
            else:
                pool = targets.global_pool()
            run = create_run(
                config,
                "rl_target" if command == "train-target" else "rl_global",
                args.output_dir,
            )
            RLTrainer(config, args.checkpoint, run, targets, pool).train()


def main():
    command = parser()
    try:
        _run(command.parse_args())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        command.exit(2, f"Error: {exc}\n")
