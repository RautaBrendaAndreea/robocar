# calcule les metriques pour evaluer si le modele est bon ou pas
# mse = erreur moyenne au carre, mae = erreur moyenne absolue
# rmse = racine de mse (plus lisible), r2 = score de 0 a 1 (1 = parfait)

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(y_true, y_pred):
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def print_metrics(y_true, y_pred, label="Model"):
    metrics = evaluate(y_true, y_pred)
    print(f"\n--- {label} Evaluation ---")
    for name, val in metrics.items():
        print(f"  {name.upper():>5}: {val:.6f}")
    return metrics
