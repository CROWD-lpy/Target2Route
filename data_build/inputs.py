import csv
from collections import Counter
from functools import lru_cache

import pandas as pd
from rdkit import Chem, rdBase

from src.esm2 import normalize_sequence

from .common import PAIR_COLUMNS, atomic_file, describe_file


CHAIN_PREFIX = "BindingDB Target Chain Sequence"
POTENCY_COLUMNS = ("Ki (nM)", "Kd (nM)", "IC50 (nM)")


@lru_cache(maxsize=50000)
def molecule(smiles):
    with rdBase.BlockLogs():
        return Chem.MolFromSmiles(smiles)


@lru_cache(maxsize=50000)
def canonical_smiles(smiles):
    mol = molecule(smiles)
    if mol is None or not mol.GetNumAtoms():
        return None
    if any(atom.GetAtomMapNum() for atom in mol.GetAtoms()):
        mol = Chem.Mol(mol)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def fasta_sequence(path):
    return normalize_sequence(path.read_text(encoding="utf-8-sig"))


def filter_bindingdb(source, output, *, threshold, exclude_fasta, chunk_size):
    header = pd.read_csv(source, sep="\t", encoding="utf-8-sig", nrows=0)
    chains = [field for field in header if field.startswith(CHAIN_PREFIX)]
    fields = ["Ligand SMILES", *POTENCY_COLUMNS, *chains]
    if not chains or not set(fields).issubset(header):
        raise ValueError(f"{source} must contain Ligand SMILES, Ki/Kd/IC50 and target chain fields")
    held_out = {fasta_sequence(path) for path in exclude_fasta}
    normalized = lru_cache(maxsize=25000)(normalize_sequence)
    seen = set()
    counts = Counter()
    batches = pd.read_csv(
        source, sep="\t", encoding="utf-8-sig", usecols=fields,
        dtype=str, keep_default_na=False, chunksize=chunk_size,
    )
    with batches, atomic_file(output) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(PAIR_COLUMNS)
        for frame in batches:
            counts["input_rows"] += len(frame)
            potency = frame[list(POTENCY_COLUMNS)].apply(
                lambda col: pd.to_numeric(col.str.replace(r"[<>~\s]", "", regex=True), errors="coerce")
            )
            active = frame.loc[
                ((potency >= 0) & (potency < threshold)).any(axis=1),
                ["Ligand SMILES", *chains],
            ]
            counts["potency_pass"] += len(active)
            for smiles, *values in active.itertuples(index=False, name=None):
                sequences = [value for value in values if value.strip()]
                if len(sequences) != 1:
                    counts["not_single_chain"] += 1
                    continue
                try:
                    sequence = normalized(sequences[0])
                except ValueError:
                    counts["invalid_sequence"] += 1
                    continue
                if sequence in held_out:
                    counts["held_out"] += 1
                    continue
                smiles = canonical_smiles(smiles.strip())
                if not smiles:
                    counts["invalid_smiles"] += 1
                    continue
                pair = (sequence, smiles)
                if pair in seen:
                    counts["duplicate_pair"] += 1
                    continue
                seen.add(pair)
                writer.writerow(pair)
        if not seen:
            raise ValueError("No eligible BindingDB ligand-target pairs")
    return {
        "bindingdb": describe_file(source),
        "threshold_nm": threshold,
        "excluded_fastas": [describe_file(path) for path in exclude_fasta],
        "unique_pairs": len(seen),
        "pairs": describe_file(output),
        **counts,
    }
