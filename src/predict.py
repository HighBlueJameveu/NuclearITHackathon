from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.features import build

ROOT = Path(__file__).resolve().parents[1]


def main():
    obj = joblib.load(ROOT / "models/si_model.joblib")
    if isinstance(obj, dict):
        model = obj["model"]
        clip = float(obj.get("si_pred_clip", 15000.0))
    else:
        model = obj
        clip = 15000.0

    te = pd.read_csv(ROOT / "data/raw/test.csv")
    pred = np.clip(10.0 ** model.predict(build(te["SMILES"])) - 1.0, 0.0, clip)

    sub = pd.DataFrame({"index": te["index"], "SI VACV": pred})
    sample = pd.read_csv(ROOT / "data/raw/sample_submit.csv")
    assert len(sub) == len(sample)
    assert sub["index"].equals(sample["index"])
    assert np.isfinite(sub["SI VACV"]).all()
    assert (sub["SI VACV"] >= 0).all()

    sub.to_csv(ROOT / "results/submission.csv", index=False)
    print("Saved results/submission.csv")
    print(sub["SI VACV"].describe())


if __name__ == "__main__":
    main()