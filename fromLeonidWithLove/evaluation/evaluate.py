import json
from pathlib import Path
from .metrics import evaluate_predictions


def evaluate_to_json(y_true, y_pred, out_path: str) -> dict:
    metrics = evaluate_predictions(y_true, y_pred)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics