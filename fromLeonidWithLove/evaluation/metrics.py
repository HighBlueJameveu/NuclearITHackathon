import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_predictions(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if not np.isfinite(y_pred).all():
        raise ValueError("y_pred содержит NaN/inf")
    if not np.isfinite(y_true).all():
        raise ValueError("y_true содержит NaN/inf")

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
        "spearman": float(spearmanr(y_true, y_pred).correlation),
    }