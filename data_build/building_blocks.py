import gzip
import pickle
import random
from pathlib import Path

import faiss
import numpy as np
from rdkit import Chem, DataStructs, rdBase

from src.chemistry.fingerprints import Fingerprints

from .common import atomic_file, describe_file, new_directory, write_json
from .inputs import molecule


def block_names(config):
    prefix = f"zinc_{config.fp_type.upper()}_{config.fp_dim}"
    return f"{prefix}_faiss_ivfpq.index", f"{prefix}_smiles_dict.pkl"


def block_files(paths):
    allowed = (".smi", ".smi.gz", ".txt", ".txt.gz")
    files = set()
    for path in paths:
        if path.is_dir():
            files.update(item.resolve() for item in path.rglob("*")
                         if item.is_file() and item.name.lower().endswith(allowed))
        elif path.is_file():
            files.add(path.resolve())
        else:
            raise FileNotFoundError(path)
    if not files:
        raise ValueError("No ZINC20 SMILES files found")
    return sorted(files)


def _smiles_rows(files):
    for path in files:
        opener = gzip.open if path.suffix.lower() == ".gz" else open
        with opener(path, "rt", encoding="utf-8-sig") as handle:
            for row in handle:
                fields = row.split(maxsplit=1)
                if fields and fields[0].lower() not in {"smiles", "smi"}:
                    yield fields[0]


def building_blocks(smiles_rows):
    for smiles in smiles_rows:
        mol = molecule(smiles)
        if mol is not None and mol.GetNumAtoms():
            yield smiles, mol


def build_stock(inputs, output):
    count = 0
    if output.suffix != ".txt":
        raise ValueError("Use a .txt destination for the AiZynthFinder InChIKey stock")
    files = block_files(inputs)
    with atomic_file(output) as handle:
        for _, mol in building_blocks(_smiles_rows(files)):
            key = Chem.MolToInchiKey(mol)
            if key:
                handle.write(key + "\n")
                count += 1
        if not count:
            raise ValueError("The SMILES inputs did not produce any stock InChIKeys")
    return {
        "stock": describe_file(output),
        "inchi_key_rows": count,
        "rdkit": rdBase.rdkitVersion,
        "smiles_files": [describe_file(path) for path in files],
    }


def _training_smiles(files, cap, seed):
    samples = []
    rng = random.Random(seed)
    for i, smiles in enumerate(_smiles_rows(files)):
        if i < cap:
            samples.append(smiles)
        else:
            slot = rng.randrange(i + 1)
            if slot < cap:
                samples[slot] = smiles
    return samples


def build_faiss(config, options):
    if options.output.exists():
        raise FileExistsError(f"Output already exists: {options.output}")
    files = block_files(options.smiles)
    dim = config.fp_dim
    fingerprints = Fingerprints(config)
    faiss.omp_set_num_threads(options.threads)
    flat = faiss.IndexFlatL2(dim)
    if options.flat:
        index = flat
    else:
        if dim % options.pq_m:
            raise ValueError("--pq-m must divide fp_dim")
        if options.pq_bits > 8:
            raise ValueError("--pq-bits must be at most 8")
        index = faiss.IndexIVFPQ(flat, dim, options.nlist, options.pq_m, options.pq_bits)
        index.cp.seed = options.seed
        index.pq.cp.seed = options.seed
        sample = _training_smiles(files, options.train_size, options.seed)
        matrix = np.zeros((len(sample), dim), dtype=np.float32)
        training_rows = 0
        for _, mol in building_blocks(sample):
            DataStructs.ConvertToNumpyArray(fingerprints.bit_vector(mol), matrix[training_rows])
            training_rows += 1
        minimum = max(options.nlist, 2 ** options.pq_bits)
        if training_rows < minimum:
            raise ValueError(
                f"IVFPQ needs at least {minimum} valid training rows, got {training_rows}. "
                "Increase --train-size or use --flat for small inputs."
            )
        index.train(matrix[:training_rows])
        del matrix, sample

    mapping = []
    buffer = np.zeros((options.batch_size, dim), dtype=np.float32)
    for smiles, mol in building_blocks(_smiles_rows(files)):
        offset = len(mapping) % len(buffer)
        row = buffer[offset]
        row.fill(0)
        DataStructs.ConvertToNumpyArray(fingerprints.bit_vector(mol), row)
        mapping.append(smiles)
        if offset + 1 == len(buffer):
            index.add(buffer)
    if not mapping:
        raise ValueError("No valid SMILES in the building-block inputs")
    remainder = len(mapping) % len(buffer)
    if remainder:
        index.add(buffer[:remainder])

    index_name, mapping_name = block_names(config)
    result = {
        "building_blocks": len(mapping),
        "fingerprint": f"{config.fp_type}_{dim}",
        "seed": options.seed,
        "method": "flat_l2" if options.flat else "ivfpq",
        "index": str(options.output / index_name),
        "smiles_map": str(options.output / mapping_name),
    }
    with new_directory(options.output) as staging:
        faiss.write_index(index, str(staging / index_name))
        with (staging / mapping_name).open("wb") as handle:
            pickle.dump(mapping, handle, protocol=5)
        write_json(staging / "manifest.json", {
            **result,
            "faiss": faiss.__version__,
            "rdkit": rdBase.rdkitVersion,
            "threads": options.threads,
            "ivfpq": None if options.flat else {
                "nlist": options.nlist, "pq_m": options.pq_m,
                "pq_bits": options.pq_bits, "max_training_rows": options.train_size,
                "training_rows": training_rows,
            },
            "smiles_files": [describe_file(path) for path in files],
            "outputs": [describe_file(staging / name) for name in (index_name, mapping_name)],
        })
    return result
