import filecmp
import json
import shutil
from pathlib import Path

from src.esm2 import normalize_sequence
from src.runtime import save_torch
from src.targets import TargetData, read_json, validate_box

from .common import atomic_file, write_json
from .inputs import canonical_smiles, fasta_sequence


def _target_requests(options):
    if not options.manifest:
        if options.target_id is None:
            raise ValueError("Set --target-id to a unique nonnegative integer")
        yield {
            "target_id": options.target_id, "fasta": options.fasta,
            "sequence": options.sequence, "receptor": options.receptor,
            "box": options.box, "global_rl": options.global_rl,
            "start_mol": options.start_mol, "pdb_id": options.pdb_id,
            "chain": options.chain,
        }
        return

    if (options.target_id is not None or options.receptor or options.box
            or options.global_rl or options.start_mol or options.pdb_id or options.chain):
        raise ValueError("Define IDs, receptors, boxes and global_rl inside --manifest")
    with options.manifest.open(encoding="utf-8-sig") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError("The target manifest must contain a nonempty JSON array")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each target manifest entry must be a JSON object")
        yield {
            **record,
            **{name: options.manifest.parent / record[name]
               for name in ("fasta", "receptor") if record.get(name)},
        }


def _registry_paths(config):
    paths = {}
    for field in (
        "target_index", "target_box_center_path", "target_choose_list", "target_embedding_path"
    ):
        path = getattr(config, field, None)
        if not isinstance(path, (str, Path)) or not path:
            raise ValueError(f"Set {field} to a file path in --config")
        paths[field] = Path(path)
    return paths


def _register(config, options, paths):
    targets = TargetData(config)
    records, centers = targets.records, targets.centers
    index_dirty = centers_dirty = allowlist_dirty = False
    raw_ids = read_json(paths["target_choose_list"], [])
    if not isinstance(raw_ids, list):
        raise ValueError("target_choose_list must contain an array of target IDs")
    allowlist = list(dict.fromkeys(map(int, raw_ids)))
    allowed_ids = set(allowlist)
    prepared = []
    seen = set()
    pending_receptors = {}

    for request in _target_requests(options):
        key = request.get("target_id")
        if type(key) is not int or key < 0 or key in seen:
            raise ValueError(f"Duplicate or invalid target ID: {key}")
        seen.add(key)
        fasta, sequence = request.get("fasta"), request.get("sequence")
        if (fasta is None) == (sequence is None):
            raise ValueError(f"Target {key} requires exactly one FASTA or sequence")
        seq = fasta_sequence(fasta) if fasta is not None else normalize_sequence(sequence)
        record = records.get(key)
        if record is not None and normalize_sequence(record.get("seq")) != seq:
            raise ValueError(f"Target ID {key} is already assigned to another sequence")
        if record is None:
            record = {"idx": key, "seq": seq}
            records[key] = record
            index_dirty = True
        for field in ("pdb_id", "chain"):
            value = request.get(field)
            if value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Provide {field} as text for target {key}")
            if field in record and record[field] != value:
                raise ValueError(f"Target {key} already has a different {field}")
            if field not in record:
                record[field] = value
                index_dirty = True

        anchors = request.get("start_mol") or []
        if isinstance(anchors, str):
            anchors = [anchors]
        if anchors:
            if not isinstance(anchors, list) or not all(
                isinstance(smi, str) and canonical_smiles(smi) for smi in anchors
            ):
                raise ValueError(f"Invalid starting-molecule SMILES for target {key}")
            previous = record.get("start_mol")
            if previous and ([previous] if isinstance(previous, str) else previous) != anchors:
                raise ValueError(f"Target {key} already has different starting molecules")
            if not previous:
                record["start_mol"] = anchors
                index_dirty = True

        receptor, box = request.get("receptor"), request.get("box")
        if (receptor is None) != (box is None):
            raise ValueError(f"Provide the PDBQT receptor and box center together for target {key}")
        destination = targets._registered_receptor(key)
        if receptor is not None:
            receptor = receptor.resolve(strict=True)
            if (
                not receptor.is_file() or receptor.suffix.lower() != ".pdbqt"
                or receptor.stat().st_size == 0
            ):
                raise ValueError(f"Expected a nonempty receptor .pdbqt for target {key}")
            if destination.exists() and not filecmp.cmp(receptor, destination, shallow=False):
                raise ValueError(f"Target {key} already has a different receptor at {destination}")
            if not destination.exists():
                pending_receptors[destination] = receptor
            center = validate_box(box)
            if key in centers and validate_box(centers[key]) != center:
                raise ValueError(f"Target {key} already has a different docking center")
            if key not in centers:
                centers[key] = list(center)
                centers_dirty = True

        has_docking = (destination in pending_receptors or destination.is_file()) and key in centers
        in_global = request.get("global_rl", False)
        if type(in_global) is not bool:
            raise ValueError(f"global_rl for target {key} must be true or false")
        if in_global:
            if not has_docking:
                raise ValueError(f"Global RL target {key} requires a receptor and docking center")
            if key not in allowed_ids:
                allowlist.append(key)
                allowed_ids.add(key)
                allowlist_dirty = True
        prepared.append((key, seq, has_docking))

    # Finish feature calculation before publishing any changes. Keep the reference
    # bank fixed during the run so all additions share the same centering.
    features = targets.features
    vectors = {}
    for _, seq, _ in prepared:
        vector = features.get(seq)
        if seq not in features.bank:
            vectors[seq] = vector
    for destination, source in pending_receptors.items():
        with source.open("rb") as reader, atomic_file(destination, binary=True) as writer:
            shutil.copyfileobj(reader, writer)
    if centers_dirty:
        write_json(paths["target_box_center_path"], centers, replace=True)
    if vectors:
        bank_path = paths["target_embedding_path"]
        bank_path.parent.mkdir(parents=True, exist_ok=True)
        save_torch({**features.bank, **vectors}, bank_path)
    if index_dirty:
        write_json(paths["target_index"], records, replace=True)
    if allowlist_dirty or not paths["target_choose_list"].is_file():
        write_json(paths["target_choose_list"], allowlist, replace=True)
    return {
        "targets": [{"target_id": key, "residues": len(seq), "docking_ready": has_docking,
                     "global_rl": key in allowed_ids} for key, seq, has_docking in prepared],
        "new_embeddings": len(vectors),
    }


def register_targets(config, options):
    import fcntl

    paths = _registry_paths(config)
    registry = paths["target_index"]
    registry.parent.mkdir(parents=True, exist_ok=True)
    with (registry.parent / ".register-target.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _register(config, options, paths)
