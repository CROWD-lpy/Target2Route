from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .chemistry.fingerprints import Fingerprints


@dataclass
class RouteBatch:
    target_features: torch.Tensor
    start_features: torch.Tensor
    seq_types: torch.Tensor
    seq_templates: torch.Tensor
    seq_fps: torch.Tensor


def load_embedding_bank(path, *, required=False):
    if not path or not Path(path).is_file():
        if required:
            raise FileNotFoundError(f"Target embedding file not found: {path}")
        return {}
    # Older banks can contain NumPy vectors instead of tensors.
    bank = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(bank, dict):
        raise ValueError(f"{path} must contain a sequence-to-embedding dictionary")
    return bank


class RouteDataset(Dataset):
    def __init__(self, records, embeddings, config):
        self.records = records
        self.embeddings = embeddings
        self.fingerprints = Fingerprints(config)
        self.empty_target = torch.zeros(
            config.esm2_dim + config.pocket_dim, dtype=torch.float32
        )
        self.empty_fingerprint = np.zeros(config.fp_dim, dtype=np.float32)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        item = self.records[index]
        target = self.embeddings.get(item["target_seq"], self.empty_target)
        start = self.fingerprints.from_smiles(item["start_mol"])
        types, templates, fingerprints = [], [], []
        for kind, value, *_ in item["route"]:
            if kind == "T":
                types.append(0)
                templates.append(value)
                fingerprints.append(self.empty_fingerprint)
            elif kind == "R":
                types.append(1)
                templates.append(0)
                fingerprints.append(self.fingerprints.from_smiles(value))

        types.append(2)
        templates.append(0)
        fingerprints.append(self.empty_fingerprint)
        return RouteBatch(
            target_features=torch.as_tensor(target, dtype=torch.float32),
            start_features=torch.from_numpy(start),
            seq_types=torch.tensor(types, dtype=torch.long),
            seq_templates=torch.tensor(templates, dtype=torch.long),
            seq_fps=torch.from_numpy(np.stack(fingerprints)),
        )


def collate_routes(items):
    return RouteBatch(
        target_features=torch.stack([item.target_features for item in items]),
        start_features=torch.stack([item.start_features for item in items]),
        seq_types=pad_sequence(
            [item.seq_types for item in items], batch_first=True, padding_value=-100
        ),
        seq_templates=pad_sequence(
            [item.seq_templates for item in items], batch_first=True
        ),
        seq_fps=pad_sequence([item.seq_fps for item in items], batch_first=True),
    )
