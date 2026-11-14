import hashlib
import logging
import re
from functools import cached_property
from pathlib import Path

import torch

from .data import load_embedding_bank
from .runtime import save_torch


def normalize_sequence(sequence):
    if not isinstance(sequence, str):
        raise ValueError("Provide a protein sequence as text")
    lines = sequence.splitlines()
    headers = [i for i, line in enumerate(lines) if line.lstrip().startswith(">")]
    if headers:
        if len(headers) != 1 or any(line.strip() for line in lines[: headers[0]]):
            raise ValueError("Provide a single protein sequence or single-record FASTA")
        lines = lines[headers[0] + 1 :]
    sequence = "".join("".join(lines).split()).upper()
    if not re.fullmatch(r"[A-Z.-]+", sequence):
        raise ValueError(
            "Provide a nonempty protein sequence using one-letter amino acid codes"
        )
    return sequence


class TargetFeatures:
    def __init__(self, config):
        self.config = config
        self.bank = load_embedding_bank(config.target_embedding_path)
        self.resolved = {}

    @staticmethod
    def _vector(value, dimension, label):
        vector = torch.as_tensor(value, dtype=torch.float32, device="cpu").detach()
        if vector.shape != (dimension,) or not torch.isfinite(vector).all():
            raise ValueError(
                f"{label} must be a finite float vector of length {dimension}"
            )
        return vector

    @cached_property
    def _centering_reference(self):
        cfg = self.config
        digest = hashlib.sha256(
            repr(
                (
                    cfg.esm2_model,
                    cfg.esm2_revision,
                    cfg.esm2_max_length,
                    cfg.esm2_dim,
                    cfg.esm2_centering,
                )
            ).encode()
        )
        mean, references = None, {}
        if cfg.esm2_centering == "none":
            mean = torch.zeros(cfg.esm2_dim)
        elif cfg.esm2_centering == "auto":
            if cfg.esm2_mean_path:
                data = torch.load(
                    cfg.esm2_mean_path, map_location="cpu", weights_only=True
                )
                mean = self._vector(data, cfg.esm2_dim, "ESM-2 training mean")
                digest.update(mean.numpy().tobytes())
            elif self.bank:
                # Short, deterministic references keep calibration cheap.
                for seq in sorted(self.bank, key=lambda item: (len(item), item))[:3]:
                    vector = self._vector(
                        self.bank[seq], cfg.esm2_dim, "Reference embedding"
                    )
                    references[seq] = vector
                    digest.update(seq.encode())
                    digest.update(vector.numpy().tobytes())
            else:
                raise FileNotFoundError(
                    "ESM-2 centering requires the matching target_embedding_path bank or esm2_mean_path. "
                    "Set esm2_centering: none for a model trained on raw CLS embeddings."
                )
        else:
            raise ValueError("esm2_centering must be auto or none")
        return mean, references, digest.hexdigest()[:20]

    @cached_property
    def _encoder(self):
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Install the ESM-2 dependencies with pip install -e '.[esm2]'"
            ) from exc

        cfg = self.config
        device = cfg.esm2_device or cfg.device
        logging.getLogger("target_to_route").info("Loading %s on %s", cfg.esm2_model, device)
        with torch.random.fork_rng(devices=[]):
            tokenizer = AutoTokenizer.from_pretrained(
                cfg.esm2_model, revision=cfg.esm2_revision
            )
            model = (
                AutoModel.from_pretrained(
                    cfg.esm2_model,
                    revision=cfg.esm2_revision,
                    add_pooling_layer=False,
                    torch_dtype=torch.float32,
                    attn_implementation="eager",
                )
                .to(device)
                .eval()
            )
        return tokenizer, model

    @torch.inference_mode()
    def _raw_features(self, sequences):
        tokenizer, model = self._encoder
        tokens = tokenizer(
            list(sequences),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.esm2_max_length,
        )
        outputs = model(**tokens.to(model.device))
        return outputs.last_hidden_state[:, 0, :].to(device="cpu", dtype=torch.float32)

    @cached_property
    def _mean(self):
        mean, references, _ = self._centering_reference
        if mean is not None:
            return mean
        offsets = self._raw_features(references) - torch.stack(
            list(references.values())
        )
        mean = offsets.mean(dim=0)
        if not torch.allclose(offsets, mean.expand_as(offsets), atol=0.002, rtol=0.002):
            raise ValueError(
                "Existing target vectors do not match a shared ESM-2 CLS centering offset. "
                "Use the training embedding bank and encoder settings, or provide esm2_mean_path."
            )
        return mean

    def get(self, sequence):
        cfg = self.config
        dim = cfg.esm2_dim + cfg.pocket_dim
        if sequence in self.resolved:
            return self.resolved[sequence]
        if sequence in self.bank:
            vector = self._vector(self.bank[sequence], dim, "Target embedding")
            self.resolved[sequence] = vector
            return vector
        if cfg.pocket_dim:
            raise ValueError(
                "This checkpoint requires pocket features. Provide a precomputed target vector."
            )

        _, _, cache_version = self._centering_reference
        key = hashlib.sha256(sequence.encode()).hexdigest()
        cache_path = Path(cfg.esm2_cache_dir) / cache_version / f"{key}.pt"
        if cache_path.is_file():
            vector = torch.load(cache_path, map_location="cpu", weights_only=True)
        else:
            mean = self._mean
            raw = self._vector(
                self._raw_features([sequence])[0], dim, "Raw ESM-2 embedding"
            )
            vector = raw - mean
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            save_torch(vector, cache_path)
        vector = self._vector(vector, dim, "Cached ESM-2 embedding")
        self.resolved[sequence] = vector
        return vector
