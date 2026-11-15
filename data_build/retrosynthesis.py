import csv
import json
import logging
import random
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import yaml
from rdkit import rdBase

from .common import PAIR_COLUMNS, atomic_file, describe_file


def _portable_config(path):
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}")
    assets = set()

    def local_file(filename):
        resolved = (path.parent / filename).resolve(strict=True)
        assets.add(resolved)
        return str(resolved)

    for group, fields in (
        ("expansion", ("model", "template", "mask")),
        ("filter", ("model",)),
        ("stock", ("path",)),
    ):
        for name, entry in config.get(group, {}).items():
            if isinstance(entry, str):
                config[group][name] = local_file(entry)
            elif isinstance(entry, list):
                config[group][name] = [local_file(item) for item in entry]
            elif isinstance(entry, dict):
                for field in fields:
                    if entry.get(field):
                        entry[field] = local_file(entry[field])
    return config, [describe_file(asset) for asset in sorted(assets)]


def initialize_finder(config, options):
    from aizynthfinder.aizynthfinder import AiZynthFinder

    global finder, search_options
    finder = AiZynthFinder(configdict=config)
    finder.stock.select(options.stock)
    finder.expansion_policy.select(options.policy)
    if options.filter:
        finder.filter_policy.select(options.filter)
    finder.config.search.max_transforms = options.max_depth
    finder.config.search.time_limit = options.time_limit
    finder.config.search.iteration_limit = options.iterations
    finder.config.search.return_first = False
    finder.config.post_processing.max_routes = max(
        finder.config.post_processing.max_routes, options.max_routes
    )
    search_options = options


def search_task(task):
    row, sequence, smiles = task
    seed = (search_options.seed + row) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    finder.target_smiles = smiles
    try:
        finder.prepare_tree()
        finder.tree_search()
        finder.build_routes()
    except Exception as exc:
        raise RuntimeError(f"AiZynthFinder failed at input row {row + 2} ({smiles}): {exc}") from exc
    return {
        "target_seq": sequence, "target_smi": smiles,
        "trees": finder.routes.dicts[:search_options.max_routes],
    }


def _write_results(results, output, provenance):
    pairs = routes = 0
    with atomic_file(output) as handle:
        handle.write(json.dumps({"_trace_retrosynthesis": provenance}) + "\n")
        for result in results:
            handle.write(json.dumps(result, default=_numpy_scalar) + "\n")
            pairs += 1
            routes += len(result["trees"])
            if pairs % 1000 == 0:
                logging.info("Retrosynthesis processed %d ligand-target pairs", pairs)
        if not pairs:
            raise ValueError("This shard has no ligand-target pairs")
    return {"pairs": pairs, "raw_routes": routes, "shard": provenance}


def _numpy_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _tasks(options):
    with options.pairs.open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        if not set(PAIR_COLUMNS).issubset(rows.fieldnames or []):
            raise ValueError(f"{options.pairs} must contain {', '.join(PAIR_COLUMNS)}")
        for i, row in enumerate(rows):
            if options.limit is not None and i >= options.limit:
                break
            if i % options.parts == options.part - 1:
                yield i, row["target_seq"], row["target_smi"]


def retrosynthesize(options):
    from importlib.metadata import PackageNotFoundError, version

    try:
        finder_version = version("aizynthfinder")
    except PackageNotFoundError as exc:
        raise RuntimeError("Use AiZynthFinder 4.4.1 in the separate retrosynthesis environment") from exc
    if options.part > options.parts:
        raise ValueError("--part must be between 1 and --parts")
    if options.output.exists():
        raise FileExistsError(f"Completed shard already exists: {options.output}")
    config, assets = _portable_config(options.aizynth_config)
    description = {
        "pairs_sha256": describe_file(options.pairs)["sha256"],
        "config_sha256": describe_file(options.aizynth_config)["sha256"],
        "policy_and_stock_files": assets,
        "aizynthfinder": finder_version,
        "rdkit": rdBase.rdkitVersion,
        "part": options.part, "parts": options.parts, "limit": options.limit,
        "policy": options.policy, "stock": options.stock, "filter": options.filter,
        "max_depth": options.max_depth, "max_routes": options.max_routes,
        "iterations": options.iterations, "time_limit_seconds": options.time_limit,
        "seed": options.seed,
    }
    if options.workers == 1:
        initialize_finder(config, options)
        return _write_results(map(search_task, _tasks(options)), options.output, description)
    with ProcessPoolExecutor(
        max_workers=options.workers, mp_context=get_context("spawn"),
        initializer=initialize_finder, initargs=(config, options),
    ) as executor:
        def parallel_results():
            tasks = _tasks(options)
            # Bound queued work for million-row input tables.
            while batch := list(islice(tasks, 8 * options.workers)):
                yield from executor.map(search_task, batch)

        return _write_results(parallel_results(), options.output, description)


def _pickle_records(path):
    import pickle

    with path.open("rb") as handle:
        while handle.peek(1):
            batch = pickle.load(handle)
            if not isinstance(batch, list):
                raise ValueError(f"{path} must contain route-record lists")
            yield from batch


def _jsonl_header(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return json.loads(line).get("_trace_retrosynthesis")
    raise ValueError(f"Empty route file: {path}")


def route_records(paths):
    """Read new JSONL shards or the original (possibly appended) route pickles."""
    paths = [path.resolve(strict=True) for path in paths]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate input route files")
    shards = {}
    for path in paths:
        if path.suffix != ".jsonl":
            continue
        header = _jsonl_header(path)
        if header is None:
            continue
        part = header["part"]
        batch = json.dumps({key: value for key, value in header.items() if key != "part"},
                           sort_keys=True)
        group = shards.setdefault(batch, [])
        if part in group or not 1 <= part <= header["parts"]:
            raise ValueError(f"Repeated or invalid retrosynthesis shard in {path}")
        group.append(part)
    for batch, group in shards.items():
        if len(group) != json.loads(batch)["parts"]:
            raise ValueError(f"Incomplete retrosynthesis run, found shards {sorted(group)}")

    for path in sorted(paths):
        if path.suffix in {".pkl", ".pickle"}:
            yield from _pickle_records(path)
        elif path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        record = json.loads(line)
                        if "_trace_retrosynthesis" not in record:
                            yield record
        else:
            raise ValueError(f"Use a TRACE .jsonl or a legacy .pkl route file: {path}")
