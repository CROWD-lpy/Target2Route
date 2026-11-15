import hashlib
import math
import pickle
import random
import shutil
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch
import yaml
from rdkit.Chem.Scaffolds import MurckoScaffold

from src.chemistry.building_blocks import BuildingBlockIndex
from src.chemistry.fingerprints import Fingerprints
from src.chemistry.routes import MAX_REACTIONS
from src.esm2 import TargetFeatures

from .building_blocks import block_names
from .common import describe_file, new_directory, write_json
from .inputs import fasta_sequence, molecule
from .routes import build_routes, read_templates, template_catalog


SPLITS = ("train", "valid", "test")
TEMPLATE_NAME = "trace_core_templates_top90_update.csv"


def embedding_digest(bank, sequences, dim):
    digest = hashlib.sha256()
    for seq in sorted(sequences):
        vector = TargetFeatures._vector(bank[seq], dim, "Target embedding")
        digest.update(seq.encode("ascii") + b"\0")
        digest.update(vector.numpy().astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


def _embedding_bank(sequences, extra_sequences, config, options):
    if config.pocket_dim:
        raise ValueError(
            "This data builder requires the sequence-only TRACE configuration (pocket_dim=0)"
        )
    if not options.reuse_embeddings:
        # A fresh rebuild does not read or depend on a historical feature bank.
        config = type(config)(**{**vars(config), "target_embedding_path": None})
    features = TargetFeatures(config)
    supplied = (
        torch.load(options.raw_embeddings, map_location="cpu", weights_only=True)
        if options.raw_embeddings else None
    )
    if supplied is not None and not isinstance(supplied, dict):
        raise ValueError("--raw-embeddings must contain a sequence-to-raw-CLS dictionary")
    registry_sequences = sorted(extra_sequences.difference(sequences))
    selected = [*sequences, *registry_sequences]
    bank, mean = {}, None
    if options.reuse_embeddings:
        if not features.bank and not config.esm2_mean_path and config.esm2_centering != "none":
            raise ValueError("Provide the existing ESM-2 bank or its mean with --config")
        bank = {
            seq: features._vector(features.bank[seq], config.esm2_dim, "Stored ESM-2 vector")
            for seq in selected if seq in features.bank
        }
        mean = features._mean
    missing = [seq for seq in selected if seq not in bank]
    for i in range(0, len(missing), options.esm_batch_size):
        chunk = missing[i:i + options.esm_batch_size]
        if supplied is None:
            raw = features._raw_features(chunk)
        else:
            absent = next((seq for seq in chunk if seq not in supplied), None)
            if absent is not None:
                raise ValueError(f"--raw-embeddings lacks a sequence beginning {absent[:20]}")
            raw = [supplied[seq] for seq in chunk]
        for seq, vector in zip(chunk, raw):
            bank[seq] = features._vector(vector, config.esm2_dim, "Raw ESM-2 vector")
            if mean is not None:
                bank[seq] = bank[seq] - mean
    if mean is None:
        if config.esm2_centering == "none":
            mean = torch.zeros(config.esm2_dim)
        elif config.esm2_centering == "auto":
            mean = torch.stack([bank[seq] for seq in sequences]).mean(dim=0)
            bank = {seq: vector - mean for seq, vector in bank.items()}
        else:
            raise ValueError("esm2_centering must be auto or none")
    return {seq: bank[seq] for seq in selected}, mean


def _cluster_sequences(bank, cutoff):
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_distances

    sequences = list(bank)
    if len(sequences) == 1:
        return {sequences[0]: 0}
    distances = cosine_distances(torch.stack(list(bank.values())).numpy())
    labels = AgglomerativeClustering(
        n_clusters=None, distance_threshold=1.0 - cutoff, metric="precomputed",
        linkage="average",
    ).fit_predict(distances)
    return dict(zip(sequences, map(int, labels)))


@lru_cache(maxsize=50000)
def scaffold(smiles):
    return MurckoScaffold.MurckoScaffoldSmiles(mol=molecule(smiles)) or smiles


def _split_examples(records, clusters, ratios, seed):
    groups = defaultdict(list)
    template_groups = defaultdict(set)
    group_templates = defaultdict(set)
    for row in records:
        group = (clusters[row["target_seq"]], scaffold(row["target_smi"]))
        groups[group].append(row)
        for action, value, *_ in row["route"]:
            if action == "T":
                template_groups[value].add(group)
                group_templates[group].add(value)

    train_groups, covered = set(), set()
    for template, members in sorted(template_groups.items()):
        if template not in covered:
            selected = min(members, key=lambda group: (
                len(groups[group]) < 2, len(groups[group]), group
            ))
            train_groups.add(selected)
            covered.update(group_templates[selected])
    remaining = sorted(groups.keys() - train_groups)
    if len(remaining) < 2:
        raise ValueError("The template-covering train groups leave fewer than two held-out groups")
    random.Random(seed).shuffle(remaining)
    assignments = {group: "train" for group in train_groups}
    totals = Counter(train=sum(len(groups[group]) for group in train_groups))
    for i, group in enumerate(remaining):
        left = len(remaining) - i
        if left > 2 and totals["train"] < len(records) * ratios[0]:
            split = "train"
        elif left > 1 and totals["valid"] < len(records) * ratios[1]:
            split = "valid"
        else:
            split = "test"
        assignments[group] = split
        totals[split] += len(groups[group])
    splits = {split: [] for split in SPLITS}
    for group in sorted(groups):
        splits[assignments[group]].extend(groups[group])
    return splits, assignments


def _pos_weights(records, config):
    reactants = Counter(
        value for row in records for action, value, *_ in row["route"] if action == "R"
    )
    if not reactants:
        raise ValueError("The train split has no reactant prediction actions")
    bits = np.zeros(config.fp_dim, dtype=np.int64)
    fingerprints = Fingerprints(config)
    for smiles, count in reactants.items():
        positions = list(fingerprints.bit_vector(molecule(smiles)).GetOnBits())
        bits[positions] += count
    weights = np.ones(config.fp_dim, dtype=np.float32)
    present = bits > 0
    weights[present] = np.maximum(1, (reactants.total() - bits[present]) / bits[present])
    return {f"{config.fp_type.lower()}_{config.fp_dim}": torch.from_numpy(weights)}


def _versions():
    versions = {}
    for package in ("torch", "rdkit", "numpy", "pandas", "faiss-cpu", "scikit-learn", "transformers"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            pass
    return versions


def _model_paths(config, output, template_rows):
    targets = output / "targets"
    return {
        "num_templates": template_rows, "max_seq_len": config.max_seq_len,
        "fp_dim": config.fp_dim, "fp_type": config.fp_type,
        "esm2_model": config.esm2_model, "esm2_revision": config.esm2_revision,
        "esm2_dim": config.esm2_dim, "pocket_dim": config.pocket_dim,
        "esm2_max_length": config.esm2_max_length,
        "esm2_centering": config.esm2_centering,
        "esm2_cache_dir": str(config.esm2_cache_dir),
        "esm2_mean_path": str(output / "supervised/esm2_mean.pt"),
        "fp_pos_weights": str(output / "supervised/fp_pos_weights.pt"),
        "fp_index_path": str(config.fp_index_path), "smi_list_path": str(config.smi_list_path),
        "template_csv_path": str(output / "templates" / TEMPLATE_NAME),
        "target_embedding_path": str(output / "supervised/seqs_embeddings.pt"),
        "train_data_path": str(output / "supervised/train.pkl"),
        "target_index": str(targets / "target_index_cleaned.json"),
        "target_choose_list": str(targets / "train_target_index.json"),
        "target_box_center_path": str(targets / "docking_centers.json"),
        "target_base_dir": str(targets / "receptors"),
        **{f"{split}_dataset_path": str(output / "supervised" / f"{split}.pkl") for split in SPLITS},
    }


def build_supervised(config, options):
    if not 1 <= options.min_steps <= options.max_steps <= MAX_REACTIONS:
        raise ValueError(f"Require 1 <= min-steps <= max-steps <= {MAX_REACTIONS}")
    if not math.isfinite(options.start_max_mw) or options.start_max_mw <= 0:
        raise ValueError("--start-max-mw must be positive and finite")
    if not 0 < options.template_coverage <= 1 or not 0 < options.similarity <= 1:
        raise ValueError("--template-coverage and --similarity must be in (0, 1]")
    if not all(ratio > 0 for ratio in options.ratios) or not math.isclose(sum(options.ratios), 1):
        raise ValueError("Provide positive train/valid/test ratios summing to 1")
    if options.output.exists():
        raise FileExistsError(f"Output already exists: {options.output}")

    if options.blocks_dir is not None:
        index_name, mapping_name = block_names(config)
        config.fp_index_path = str(options.blocks_dir / index_name)
        config.smi_list_path = str(options.blocks_dir / mapping_name)
    if not config.fp_index_path or not config.smi_list_path:
        raise ValueError("Provide the index and SMILES map with --blocks-dir or --config")
    blocks = BuildingBlockIndex(config)
    index_count = len(blocks.mapping)
    if not index_count:
        raise ValueError("The FAISS index has no building blocks")
    del blocks

    held_out = {fasta_sequence(path) for path in options.exclude_fasta}
    fixed = bool(options.frozen_templates)
    source = options.frozen_templates or options.uspto_templates
    frame = read_templates(source, fixed=fixed)
    catalog = template_catalog(frame, fixed=fixed)
    records, templates, stats = build_routes(
        options.routes, options, config, held_out, frame, catalog
    )
    sequences = sorted({row["target_seq"] for row in records})
    registry_sequences = {fasta_sequence(path) for path in options.registry_fasta}
    bank, mean = _embedding_bank(sequences, registry_sequences, config, options)
    clusters = _cluster_sequences({seq: bank[seq] for seq in sequences}, options.similarity)
    splits, assignments = _split_examples(records, clusters, options.ratios, options.seed)
    if len(splits["train"]) < 2 or not all(splits.values()):
        raise ValueError(
            "The split needs at least two train examples and nonempty validation and test sets"
        )
    weights = _pos_weights(splits["train"], config)
    paths = _model_paths(config, options.output, len(templates))

    with new_directory(options.output) as staging:
        supervised = staging / "supervised"
        for folder in (supervised, staging / "templates", staging / "targets"):
            folder.mkdir()
        for split, rows in splits.items():
            with (supervised / f"{split}.pkl").open("wb") as handle:
                pickle.dump(rows, handle, protocol=5)
        for filename, data in (
            ("seqs_embeddings.pt", bank), ("esm2_mean.pt", mean), ("fp_pos_weights.pt", weights)
        ):
            with (supervised / filename).open("wb") as handle:
                torch.save(data, handle)
        template_path = staging / "templates" / TEMPLATE_NAME
        if fixed:
            shutil.copyfile(source, template_path)
        else:
            templates.to_csv(template_path, index=False)
        for name, value in (
            ("target_index_cleaned.json", {}), ("train_target_index.json", []),
            ("docking_centers.json", {}),
        ):
            write_json(staging / "targets" / name, value)
        with (staging / "data_paths.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(paths, handle, sort_keys=False)
        write_json(staging / "split_groups.json", [
            [cluster, core, split] for (cluster, core), split in sorted(assignments.items())
        ])
        write_json(staging / "sequence_clusters.json", clusters)
        manifest = {
            **stats, "seed": options.seed, "min_steps": options.min_steps,
            "max_steps": options.max_steps, "start_max_mw": options.start_max_mw,
            "requested_template_coverage": options.template_coverage,
            "template_count": len(templates), "sequence_count": len(sequences),
            "esm_batch_size": options.esm_batch_size,
            "esm2": {name: paths[name] for name in
                     ("esm2_model", "esm2_revision", "esm2_max_length", "esm2_centering")},
            "cluster_count": len(set(clusters.values())), "building_blocks": index_count,
            "split": {split: len(rows) for split, rows in splits.items()},
            "requested_ratios": options.ratios, "sequence_cosine_cutoff": options.similarity,
            "split_group": "sequence_cluster_and_murcko_scaffold",
            "supervised_embedding_sha256": embedding_digest(bank, sequences, config.esm2_dim),
            "reuse_embeddings": options.reuse_embeddings, "frozen_templates": fixed,
            "excluded_sequences": len(held_out),
            "additional_registry_sequences": len(registry_sequences - set(sequences)),
            "input_files": {
                "routes": [describe_file(path) for path in options.routes],
                "templates": describe_file(source),
                "building_blocks": {field: describe_file(getattr(config, field))
                                    for field in ("fp_index_path", "smi_list_path")},
                "exclusions": [describe_file(path) for path in options.exclude_fasta],
                "registry_fastas": [describe_file(path) for path in options.registry_fasta],
                "raw_esm2": describe_file(options.raw_embeddings) if options.raw_embeddings else None,
                "calibration_bank": (
                    describe_file(config.target_embedding_path)
                    if options.reuse_embeddings and config.target_embedding_path
                    and Path(config.target_embedding_path).is_file() else None
                ),
                "calibration_mean": (
                    describe_file(config.esm2_mean_path)
                    if options.reuse_embeddings and config.esm2_mean_path else None
                ),
            },
            "outputs": {name: describe_file(staging / name) for name in (
                f"templates/{TEMPLATE_NAME}", "supervised/train.pkl", "supervised/valid.pkl",
                "supervised/test.pkl", "supervised/seqs_embeddings.pt", "supervised/esm2_mean.pt",
                "supervised/fp_pos_weights.pt", "sequence_clusters.json", "split_groups.json",
            )},
            "versions": _versions(),
            "python": sys.version.split()[0],
        }
        write_json(staging / "manifest.json", manifest)
    return {"data_config": str(options.output / "data_paths.yaml"), **stats,
            "split": manifest["split"], "template_count": len(templates)}
