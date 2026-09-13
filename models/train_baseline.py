from featurizers.fingerprints import *
import pandas as pd

data = pd.read_csv("../data/esol.csv")

data["smiles"] = [smiles_to_fp(k) for k in data["smiles"]]