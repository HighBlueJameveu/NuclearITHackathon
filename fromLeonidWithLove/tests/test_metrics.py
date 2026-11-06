import numpy as np
from src.evaluation.metrics import evaluate_predictions

def test_metrics_perfect():
    y = np.array([1.0, 2.0, 3.0])
    m = evaluate_predictions(y, y)
    assert abs(m["rmse"]) < 1e-9
    assert abs(m["r2"] - 1.0) < 1e-9
    assert abs(m["spearman"] - 1.0) < 1e-9

def test_metrics_nan_raises():
    import pytest
    with pytest.raises(ValueError):
        evaluate_predictions([1, 2, 3], [1, np.nan, 3])