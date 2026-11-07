"""Предсказание ансамбля -> results/submission.csv"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.features import build
from src import features as F

ROOT = Path(__file__).resolve().parents[1]


def main():
    obj = joblib.load(ROOT / "models/si_model.joblib")
    models = obj["models"]
    clip = float(obj.get("si_pred_clip", 15000.0))

    F._strain_le = obj.get("strain_le")
    F._culture_le = obj.get("culture_le")

    te = pd.read_csv(ROOT / "data/raw/test.csv")
    X = build(te, fit=False)

    # Усредняем предсказания всех моделей
    preds = []
    for model in models:
        preds.append(model.predict(X))
    pred_log = np.mean(preds, axis=0)
    pred = np.clip(10.0 ** pred_log - 1.0, 0.0, clip)

    sub = pd.DataFrame({"index": te["index"], "SI VACV": pred})
    sample = pd.read_csv(ROOT / "data/raw/sample_submit.csv")
    assert len(sub) == len(sample)
    assert sub["index"].equals(sample["index"])
    assert np.isfinite(sub["SI VACV"]).all()
    assert (sub["SI VACV"] >= 0).all()

    sub.to_csv(ROOT / "results/submission.csv", index=False)
    print("Saved results/submission.csv")
    print(sub["SI VACV"].describe())
    print(f"Ensemble size: {len(models)}")


if __name__ == "__main__":
    main()