from rdkit import Chem
from rdkit.Chem import AllChem


gen = AllChem.GetMorganGenerator(radius=2)


def smiles_to_fp(smiles):
    fingerprint = gen.GetFingerprint(smiles)
    return fingerprint
