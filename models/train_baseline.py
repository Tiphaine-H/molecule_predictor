import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt

from featurizers.fingerprints import *


def prepare_features(csv_path):
    df = pd.read_csv(csv_path)
    df["fp"] = [smiles_to_fp(s) for s in df["smiles"]]
    X = np.stack(df["fp"].values)
    y = df["solubility"].values
    return X, y


# REGRESSION RANDOM FOREST:
def random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=200, random_state=0)
    model.fit(X_train, y_train)
    return model


def main():

    X_train, y_train = prepare_features("data/esol_train.csv")
    X_test, y_test = prepare_features("data/esol_test.csv")

    model = random_forest(X_train, y_train)
    y_pred = model.predict(X_test)

    # EVALUATION
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"RMSE: {rmse:.3f}  |  R²: {r2:.3f}")

    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")

    plt.title("Predicted vs Actual")
    plt.savefig("models/results_plot.png", dpi=150)
    print("Saved plot to models/results_plot.png")


main()