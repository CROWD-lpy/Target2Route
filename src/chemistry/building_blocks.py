from pathlib import Path

import faiss
import numpy as np

from ..runtime import load_pickle


class BuildingBlockIndex:
    def __init__(self, config):
        self.index = faiss.read_index(
            str(Path(config.fp_index_path).resolve(strict=True))
        )
        self.mapping = load_pickle(config.smi_list_path)
        if self.index.d != config.fp_dim or self.index.ntotal != len(self.mapping):
            raise ValueError(
                "FAISS dimension or number of entries does not match the configuration and SMILES map"
            )

    def search_continuous(self, fingerprints, top_k=5, nprobe=32):
        if hasattr(fingerprints, "detach"):
            fingerprints = fingerprints.detach().cpu().numpy()
        query = np.ascontiguousarray(fingerprints, dtype=np.float32)
        single = query.ndim == 1
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = nprobe
        distances, indices = self.index.search(np.atleast_2d(query), top_k)
        rows = [
            [
                (self.mapping[int(idx)], distance)
                for idx, distance in zip(ids, scores)
                if idx >= 0
            ]
            for ids, scores in zip(indices, distances)
        ]
        return rows[0] if single else rows
