import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors


class Fingerprints:
    def __init__(self, config):
        kind = config.fp_type.lower()
        self.dim = config.fp_dim
        if kind == "maccs":
            if self.dim != 167:
                raise ValueError("MACCS fingerprints require fp_dim=167")
            self.bit_vector = rdMolDescriptors.GetMACCSKeysFingerprint
        elif kind in {"ecfp4", "ecfp6", "fcfp4", "fcfp6"}:
            features = (
                rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
                if kind.startswith("f")
                else None
            )
            radius = 2 if kind.endswith("4") else 3
            self.bit_vector = rdFingerprintGenerator.GetMorganGenerator(
                radius=radius, fpSize=self.dim, atomInvariantsGenerator=features
            ).GetFingerprint
        elif kind == "rdkit":
            self.bit_vector = rdFingerprintGenerator.GetRDKitFPGenerator(
                fpSize=self.dim
            ).GetFingerprint
        elif kind == "atompair":
            self.bit_vector = rdFingerprintGenerator.GetAtomPairGenerator(
                fpSize=self.dim
            ).GetFingerprint
        else:
            raise ValueError(f"Unsupported fingerprint type: {config.fp_type}")

    def from_smiles(self, smiles):
        bits = np.zeros(self.dim, dtype=np.float32)
        if smiles:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise ValueError(
                    f"Invalid SMILES in a route or building-block library: {smiles}"
                )
            DataStructs.ConvertToNumpyArray(self.bit_vector(molecule), bits)
        return bits
