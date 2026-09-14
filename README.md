# Molecule Property Predictor


Predicting properties of a given molecules, starting with solubility, then toxicity. 

## Datasets

ESOL, ~1128 molecules, log-solubility target, from MoleculeNet.

## Methods

Each molecule gets turned into a Morgan fingerprint (2048-bit vector describing which small substructures (rings, chains) are present around each atom,). 

That fingerprint goes into different models to predict solubility. 

I tested two ways of splitting the data: randomly, and by molecular scaffold (grouping similar-structure molecules together so they don't leak between train and test). Scaffold split is harder and more realistic. It forces the model to generalize to genuinely new structures instead of near-copies it's already seen.


## Literature

https://arxiv.org/pdf/2606.08825


## Results

| Split    | Model         | RMSE  | R²    |
|----------|---------------|-------|-------|
| Random   | RandomForest  | 1.040 | 0.722 |
| Random   | XGBoost       | 1.066 | 0.708 |
| Scaffold | RandomForest  | 1.802 | 0.169 |
| Scaffold | XGBoost       | 1.665 | 0.291 |

As expected, XGBoost is slightly better on the scaffold-splitted dataset : a more difficult task as very similar types of molecules cannot be both in the test and in the training dataset. The tested molecules are structurally novel in that case. 


## How to run

uv sync  
uv run python data/fetch_data.py  
uv run python -m models.train_baseline  

## Next Steps

- Try to use a GNN (pytorch)  
- Try to use ChemBERTa
- Use other (bigger) solubility datasets  
- Learn to determine toxicity of molecules   

