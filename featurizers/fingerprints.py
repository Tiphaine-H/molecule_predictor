from rdkit import Chem
from rdkit.Chem import AllChem
import logging

gen = AllChem.GetMorganGenerator(radius=2)


def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        logging.error("Molecule File contains an error in SMILES")
        return None
    fp = gen.GetFingerprint(mol)
    return fp
