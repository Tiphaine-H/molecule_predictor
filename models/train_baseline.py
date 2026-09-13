import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt

from featurizers.fingerprints import *

data = pd.read_csv("data/esol.csv")

data["fp"] = [smiles_to_fp(k) for k in data["smiles"]]

X, y = data['fp'], data['solubility']

# move to numpy structures to be exploitable with sklearn
X = np.stack(data['fp'].values)
y = y.values

# fixing random_state to get reproducible results to know if we're improving
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# REGRESSION RANDOM FOREST:
regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)

# EVALUATION
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.3f}  |  R²: {r2:.3f}")

plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual solubility")
plt.ylabel("Predicted solubility")
plt.title("Random Forest: Predicted vs Actual")
plt.savefig("models/randomforest_results_plot.png", dpi=150)
print("Saved plot to models/results_plot.png")