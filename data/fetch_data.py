import deepchem as dc

tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='Raw', splitter="scaffold")
df_train = datasets[0].to_dataframe()
df_test = datasets[1].to_dataframe()


def clean_df(df, category):
    df = df.rename(columns={"ids": "smiles", "y": "solubility"})
    df[["smiles", "solubility"]].to_csv(f"data/esol_{category}.csv", index=False)


clean_df(df_train, "train")
clean_df(df_test, "test")
