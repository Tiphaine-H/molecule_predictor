import deepchem as dc

tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='Raw', splitter=None)
df = datasets[0].to_dataframe()
df = df.rename(columns={"ids": "smiles", "y": "solubility"})
df[["smiles", "solubility"]].to_csv("esol.csv", index=False)
print(df.head())