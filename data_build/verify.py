import json
from numbers import Integral
from pathlib import Path

import torch
from rdkit import rdBase
from rdkit.Chem import rdChemReactions

from src.chemistry.building_blocks import BuildingBlockIndex
from src.chemistry.templates import load_templates
from src.data import load_embedding_bank
from src.esm2 import TargetFeatures, normalize_sequence
from src.runtime import load_pickle
from src.targets import TargetData, read_json

from .common import describe_file
from .inputs import canonical_smiles
from .supervised import SPLITS, TEMPLATE_NAME, embedding_digest, scaffold


def _verify_actions(row, arities, max_len):
    actions = row["route"]
    if not isinstance(actions, (tuple, list)) or not 1 <= len(actions) <= max_len:
        raise ValueError(f"expected 1 to {max_len} route actions before EOS")
    start = row["start_mol"]
    if not isinstance(start, str) or (start and not canonical_smiles(start)):
        raise ValueError("invalid start_mol")
    expected = "T" if start else "R"
    reactions = 0
    for action, value, *_ in actions:
        if action != expected:
            raise ValueError(f"expected action {expected}, found {action}")
        if action == "T":
            if (
                isinstance(value, bool) or not isinstance(value, Integral)
                or not 0 <= value < len(arities)
            ):
                raise ValueError(f"template ID out of range: {value}")
            expected = "R" if arities[value] == 2 else "T"
            reactions += 1
        else:
            if not isinstance(value, str) or not canonical_smiles(value):
                raise ValueError("invalid reactant SMILES")
            expected = "T"
    if expected == "R" or not reactions:
        raise ValueError("route ends with an unfinished reaction or has no reaction")


def _verify_supervised(config, arities, bank, directory=None):
    counts = {}
    dim = config.esm2_dim + config.pocket_dim
    sequences = set()
    seen_templates = {}
    observed_groups = set()
    clusters = assignments = None
    if directory is not None:
        clusters = read_json(directory / "sequence_clusters.json", None)
        assignment_rows = read_json(directory / "split_groups.json", None)
        if not clusters or not assignment_rows:
            raise ValueError("Missing sequence clusters or split group assignments")
        assignments = {(cluster, core): split for cluster, core, split in assignment_rows}
    for split in SPLITS:
        path = getattr(config, f"{split}_dataset_path")
        if not path and split == "test":
            continue
        if not path:
            raise ValueError(f"Set {split}_dataset_path")
        rows = load_pickle(path)
        if not isinstance(rows, list) or len(rows) < (2 if split == "train" else 1):
            raise ValueError(f"Not enough records or incorrect pickle format in {path}")
        counts[split] = len(rows)
        used = set()
        for index, row in enumerate(rows):
            try:
                seq = row["target_seq"]
                if seq not in sequences:
                    normalize_sequence(seq)
                    if seq not in bank:
                        raise ValueError("target sequence is absent from the ESM-2 bank")
                    TargetFeatures._vector(bank[seq], dim, "Target embedding")
                    sequences.add(seq)
                if not isinstance(row["target_smi"], str) or not canonical_smiles(row["target_smi"]):
                    raise ValueError("invalid target_smi")
                _verify_actions(row, arities, config.max_seq_len)
                used.update(value for action, value, *_ in row["route"] if action == "T")
                if clusters is not None:
                    group = (clusters[seq], scaffold(row["target_smi"]))
                    if assignments.get(group) != split:
                        raise ValueError("composite group is in the wrong split")
                    observed_groups.add(group)
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}, record {index}: {exc}") from exc
        seen_templates[split] = used
        del rows
    if assignments is not None and observed_groups != assignments.keys():
        raise ValueError("At least one recorded split group has no input routes")
    unseen = set.union(set(), *[used for split, used in seen_templates.items() if split != "train"])
    return counts, sequences, sorted(unseen - seen_templates["train"])


def _verify_targets(config, require_global):
    stored = read_json(config.target_index, None)
    if not stored:
        if require_global:
            raise ValueError("No registered targets. Add targets with register-target.")
        return 0, 0
    targets = TargetData(config)
    allowlist = config.target_choose_list
    if isinstance(allowlist, str):
        allowlist = read_json(allowlist, [])
    ids = set(targets.records) if allowlist is None else {int(value) for value in allowlist}
    missing = []
    ready = set()
    dim = config.esm2_dim + config.pocket_dim
    vina = None
    if require_global:
        from vina import Vina

        vina = Vina(cpu=1, verbosity=0)
    for key in sorted(ids):
        try:
            target = targets.resolve(target_id=key)
            TargetFeatures._vector(targets.features.bank[target.sequence], dim, "Target embedding")
            if vina is not None:
                vina.set_receptor(str(target.receptor))
        except (ValueError, OSError, RuntimeError, KeyError) as exc:
            missing.append((key, str(exc)))
        else:
            ready.add(key)
    if require_global and (missing or not ready):
        raise ValueError(
            "Global RL needs enabled targets with sequence features, receptor PDBQT and box. "
            f"Incomplete target IDs: {missing[:20]}"
        )
    return len(targets.records), len(ready)


def verify_dataset(config, options):
    if options.allow_extended_bank and options.manifest is None:
        raise ValueError("--allow-extended-bank requires --manifest")
    templates = load_templates(config)
    arities = templates["num_reactants"].tolist()
    with rdBase.BlockLogs():
        for index, (smarts, arity) in enumerate(zip(templates["forward_template"], arities)):
            reaction = rdChemReactions.ReactionFromSmarts(smarts)
            if arity not in (1, 2) or reaction is None or reaction.GetNumReactantTemplates() != arity:
                raise ValueError(f"Unusable forward template or wrong arity at row {index}")
    blocks = BuildingBlockIndex(config)
    if not len(blocks.mapping):
        raise ValueError("The FAISS index has no entries")
    if not isinstance(blocks.mapping, (list, tuple, dict)) or (
        isinstance(blocks.mapping, dict) and any(i not in blocks.mapping for i in range(len(blocks.mapping)))
    ):
        raise ValueError("The SMILES map must be a list or have contiguous integer IDs")
    index_count = len(blocks.mapping)
    del blocks

    bank = load_embedding_bank(config.target_embedding_path, required=True)
    directory = options.manifest.parent if options.manifest else None
    counts, sequences, unseen = _verify_supervised(config, arities, bank, directory)
    if options.manifest:
        with options.manifest.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if options.allow_extended_bank:
            digest = embedding_digest(bank, sequences, config.esm2_dim + config.pocket_dim)
            if digest != manifest["supervised_embedding_sha256"]:
                raise ValueError("An original supervised ESM-2 vector has changed")
    del bank
    if config.esm2_mean_path:
        mean = torch.load(config.esm2_mean_path, map_location="cpu", weights_only=True)
        TargetFeatures._vector(mean, config.esm2_dim, "ESM-2 mean")
    if isinstance(config.fp_pos_weights, str):
        if not Path(config.fp_pos_weights).is_file():
            if options.manifest:
                raise FileNotFoundError(config.fp_pos_weights)
        else:
            weights = torch.load(config.fp_pos_weights, map_location="cpu", weights_only=True)
            fp_key = f"{config.fp_type.lower()}_{config.fp_dim}"
            weight = weights.get(fp_key) if isinstance(weights, dict) else None
            if (not torch.is_tensor(weight) or weight.shape not in ((1,), (config.fp_dim,))
                    or not torch.isfinite(weight).all() or (weight <= 0).any()):
                raise ValueError(f"Missing or invalid weight {fp_key} in {config.fp_pos_weights}")

    registered, ready = _verify_targets(config, options.require_global_rl)
    fields = {
        f"templates/{TEMPLATE_NAME}": "template_csv_path",
        "supervised/seqs_embeddings.pt": "target_embedding_path",
        "supervised/esm2_mean.pt": "esm2_mean_path",
        "supervised/fp_pos_weights.pt": "fp_pos_weights",
        **{f"supervised/{split}.pkl": f"{split}_dataset_path" for split in SPLITS},
    }
    if options.manifest:
        for name, original in manifest["outputs"].items():
            # Registering a new target legally extends the central feature bank.
            if options.allow_extended_bank and name == "supervised/seqs_embeddings.pt":
                continue
            path = getattr(config, fields[name]) if name in fields else options.manifest.parent / name
            if not isinstance(path, (str, Path)):
                raise ValueError(f"The manifest expects a file for {name}")
            if describe_file(path)["sha256"] != original["sha256"]:
                raise ValueError(f"Checksum mismatch: {path}")
        for field in ("fp_index_path", "smi_list_path"):
            original = manifest["input_files"]["building_blocks"][field]
            if describe_file(getattr(config, field))["sha256"] != original["sha256"]:
                raise ValueError(f"Checksum mismatch: {field}")
    report = {
        "splits": counts, "supervised_sequences": len(sequences), "templates": len(arities),
        "template_sha256": describe_file(config.template_csv_path)["sha256"],
        "building_blocks": index_count, "registered_targets": registered,
        "global_rl_targets_ready": ready, "evaluation_templates_absent_from_train": unseen,
    }
    if options.hash_files:
        for field in ("fp_index_path", "smi_list_path"):
            fields[field] = field
        report["file_digests"] = {
            name: describe_file(path) for name, field in fields.items()
            if isinstance(path := getattr(config, field, None), (str, Path))
            and Path(path).is_file()
        }
    return report
