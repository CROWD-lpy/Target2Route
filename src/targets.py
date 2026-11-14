import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from .esm2 import TargetFeatures, normalize_sequence


@dataclass
class Target:
    id: int | str
    sequence: str
    receptor: Path | None = None
    box: tuple[float, float, float] | None = None
    start_mols: tuple[str, ...] = ()


def read_json(path, default):
    if path and Path(path).is_file():
        with Path(path).open(encoding="utf-8") as handle:
            return json.load(handle)
    return default


def validate_box(coordinates):
    try:
        box = tuple(float(value) for value in coordinates)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Provide a docking box center as three coordinates in Å"
        ) from exc
    if len(box) != 3 or not all(math.isfinite(value) for value in box):
        raise ValueError(
            "Provide a docking box center as three finite coordinates in Å"
        )
    return box


class TargetData:
    def __init__(self, config):
        self.config = config
        index = read_json(config.target_index, [])
        if isinstance(index, dict):
            entries = ((key, record) for key, record in index.items())
        else:
            entries = enumerate(index)
        self.records = {}
        for index, record in entries:
            key = int(record.get("idx", index))
            if key in self.records:
                raise ValueError(f"Duplicate target ID {key} in {config.target_index}")
            self.records[key] = record
        centers = read_json(config.target_box_center_path, {})
        self.centers = {int(key): value for key, value in centers.items()}
        self.features = TargetFeatures(config)

    def _registered_receptor(self, target_id):
        return Path(self.config.target_base_dir) / f"{target_id:04d}" / "receptor.pdbqt"

    def resolve(
        self, *, target_id=None, sequence=None, receptor=None, box=None, docking=True
    ):
        if sequence is not None:
            sequence = normalize_sequence(sequence)
        if target_id is None and sequence is not None:
            candidates = []
            for key, record in self.records.items():
                stored_seq = record.get("seq")
                if not stored_seq:
                    continue
                try:
                    indexed_sequence = normalize_sequence(stored_seq)
                except ValueError:
                    continue
                if indexed_sequence == sequence:
                    candidates.append(key)
            if len(candidates) == 1:
                target_id = candidates[0]
            elif docking and len(candidates) > 1 and (receptor is None or box is None):
                raise ValueError(
                    "This sequence matches several target IDs. Specify --target-id or --receptor and --box."
                )

        record = self.records.get(target_id, {})
        stored_seq = record.get("seq")
        if stored_seq:
            try:
                stored_seq = normalize_sequence(stored_seq)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid protein sequence for target ID {target_id} "
                    f"in {self.config.target_index}: {exc}"
                ) from exc
        if stored_seq and sequence is not None and stored_seq != sequence:
            raise ValueError(
                f"The supplied sequence differs from the sequence for target ID {target_id}"
            )
        if sequence is None:
            if not stored_seq:
                raise ValueError(
                    "Supply --sequence, --fasta, or a --target-id in the target index"
                )
            sequence = stored_seq

        if docking:
            if receptor is None and record:
                receptor = self._registered_receptor(target_id)
            if box is None and record:
                box = self.centers.get(target_id)
            if receptor is None or box is None:
                raise ValueError(
                    "Docking requires --receptor and --box or a registered target with both files"
                )
            receptor = Path(receptor).resolve(strict=True)
            if not receptor.is_file():
                raise ValueError(f"Expected a receptor PDBQT file at {receptor}")
            box = validate_box(box)

        if target_id is None:
            target_id = "seq-" + hashlib.sha256(sequence.encode()).hexdigest()[:12]
        start_mols = record.get("start_mol") or ()
        if isinstance(start_mols, str):
            start_mols = (start_mols,)
        return Target(target_id, sequence, receptor, box, tuple(start_mols))

    def global_pool(self):
        selection = self.config.target_choose_list
        if isinstance(selection, str):
            with Path(selection).open(encoding="utf-8") as handle:
                selection = json.load(handle)
        allowed = (
            set(map(int, selection)) if selection is not None else set(self.records)
        )
        pool = [
            self.resolve(target_id=key)
            for key in sorted(allowed & self.records.keys() & self.centers.keys())
            if self._registered_receptor(key).is_file()
        ]
        if not pool:
            raise ValueError(
                "No global RL targets match the target index, receptor files, docking centers and allowlist"
            )
        return pool

    def batch(self, pool):
        targets = random.choices(pool, k=self.config.batch_size)
        # Resolve each unique sequence once per batch.
        features = {
            sequence: self.features.get(sequence)
            for sequence in {target.sequence for target in targets}
        }
        batch = torch.stack([features[target.sequence] for target in targets]).to(
            self.config.device
        )
        return targets, batch
