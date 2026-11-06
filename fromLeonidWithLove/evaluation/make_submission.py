import numpy as np
import pandas as pd
from pathlib import Path
from .predictor import SiPredictor


def make_submission(
    predictor: SiPredictor,
    test_csv: str,
    out_csv: str,
    id_col: str = "id",
    smiles_col: str = "smiles",
    fallback: float = 5.0,
) -> pd.DataFrame:
    df = pd.read_csv(test_csv)
    preds = np.asarray(predictor.predict(df[smiles_col].tolist()), dtype=float)

    preds = np.where(np.isfinite(preds), preds, fallback)

    sub = pd.DataFrame({id_col: df[id_col].values, "si_pred": preds})

    if len(sub) != len(df):
        raise RuntimeError("Число строк submission != test")
    if sub["si_pred"].isna().any():
        raise RuntimeError("В submission попал NaN")

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_csv, index=False)
    return sub