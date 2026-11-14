import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from functools import partial
from importlib import import_module

from rdkit import Chem
from rdkit.Chem import AllChem


def _dock(job, seed):
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from vina import Vina

    receptor, box, smiles = job
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(molecule, randomSeed=42) != 0:
        return None

    try:
        setups = MoleculePreparation().prepare(molecule)
        if not setups:
            return None
        pdbqt, success, _ = PDBQTWriterLegacy.write_string(setups[0])
        if not success:
            return None

        vina = Vina(sf_name="vina", verbosity=0, cpu=2, seed=seed)
        vina.set_receptor(receptor)
        vina.compute_vina_maps(center=box, box_size=[20.0] * 3)
        vina.set_ligand_from_string(pdbqt)
        vina.dock(exhaustiveness=2, n_poses=1)
        return float(vina.energies(n_poses=1)[0][0])
    except (ValueError, RuntimeError, OSError, IndexError):
        return None


@contextmanager
def docking_pool(config):
    try:
        import_module("meeko")
        import_module("vina")
    except ImportError as exc:
        raise RuntimeError(
            "Install the docking dependencies with pip install -e '.[docking]'"
        ) from exc

    # Vina interprets seed zero as a request for a random seed.
    seed = 0 if config.seed is None else config.seed % (2**31 - 1) or 1
    worker = partial(_dock, seed=seed)
    if config.max_workers == 1:
        yield lambda jobs: dict(zip(jobs, map(worker, jobs)))
    else:
        with ProcessPoolExecutor(
            max_workers=config.max_workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            yield lambda jobs: dict(zip(jobs, executor.map(worker, jobs)))
