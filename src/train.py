"""Обучение модели SI VACV.
Запуск: python -m src.train [--no-external]

- SI=10000 в train = «неактивно»
- Внешние данные фильтруются SI ∈ [0.05, 500]
- Таргет: log10(1+SI), предсказания клипаются до 15000
- Вес competition-данных = 3.0
"""
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from src.features import build, scaffold
from src.model import make_model

ROOT = Path(__file__).resolve().parents[1]

SI_CLIP_EXTERNAL = 500.0
SI_PRED_CLIP = 15000.0
DROP_TEST_OVERLAP = True


def canon(smiles):
    from rdkit import Chem
    m = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(m) if m else None


def load_training_data(use_external=True):
    tr = pd.read_csv(ROOT / "data/raw/train.csv")
    tr = tr.dropna(subset=["SMILES", "SI VACV"]).reset_index(drop=True)
    tr["is_competition"] = True
    parts = [tr]

    if use_external:
        known = set(tr.SMILES.map(canon))
        if DROP_TEST_OVERLAP:
            known |= set(pd.read_csv(ROOT / "data/raw/test.csv").SMILES.map(canon))

        for f in sorted((ROOT / "data/external").glob("*.csv")):
            try:
                # Более устойчивое чтение
                e = pd.read_csv(f, on_bad_lines="skip", engine="python")
            except Exception as ex:
                print(f"Пропускаю файл {f.name}: {ex}")
                continue

            # Проверяем наличие нужных колонок
            needed = {"SMILES", "IC50_uM", "CC50_uM"}
            if not needed.issubset(set(e.columns)):
                print(f"Пропускаю {f.name}: нет нужных колонок")
                continue

            n0 = len(e)
            e = e.dropna(subset=["SMILES", "IC50_uM", "CC50_uM"])
            e = e[(e.IC50_uM > 0) & (e.CC50_uM > 0)]
            e = e[~e.SMILES.map(canon).isin(known)]
            e["SI VACV"] = (e["CC50_uM"] / e["IC50_uM"]).clip(upper=SI_CLIP_EXTERNAL)
            e = e[(e["SI VACV"] >= 0.05) & (e["SI VACV"] <= SI_CLIP_EXTERNAL)]
            e["is_competition"] = False
            e = e[["SMILES", "SI VACV", "is_competition"]]
            parts.append(e)
            print(f"External: {f.name}: {n0} -> {len(e)} rows used")

    return pd.concat(parts, ignore_index=True)


def main():
    use_ext = "--no-external" not in sys.argv
    tr = load_training_data(use_external=use_ext)
    comp = tr["is_competition"].to_numpy()
    X = build(tr.SMILES)
    y = np.log10(1.0 + tr["SI VACV"].to_numpy(dtype=float))
    groups = tr.SMILES.map(scaffold).to_numpy()
    sample_weight = np.where(comp, 3.0, 1.0)

    oof = np.zeros(len(y))
    for a, b in GroupKFold(5).split(X, y, groups):
        model = make_model()
        model.fit(X[a], y[a], sample_weight=sample_weight[a])
        oof[b] = model.predict(X[b])

    pred_si = np.clip(10.0 ** oof - 1.0, 0.0, SI_PRED_CLIP)
    si_true = tr["SI VACV"].to_numpy(dtype=float)

    metrics = {
        "mode": "SMILES + filtered external + SI=10000 inactive + pred_clip=15000",
        "train_rows_total": int(len(tr)),
        "eval_rows_competition": int(comp.sum()),
        "RMSE_log10(1+SI)": float(np.sqrt(np.mean((oof[comp] - y[comp]) ** 2))),
        "MAE_log10(1+SI)": float(np.mean(np.abs(oof[comp] - y[comp]))),
        "MAE_SI": float(np.mean(np.abs(pred_si[comp] - si_true[comp]))),
        "Spearman": float(pd.Series(oof[comp]).corr(pd.Series(y[comp]), method="spearman")),
    }
    print("Scaffold-CV:", metrics)
    pd.Series(metrics).to_json(ROOT / "results/cv_metrics.json", indent=2)

    model = make_model()
    model.fit(X, y, sample_weight=sample_weight)
    joblib.dump({"model": model, "si_pred_clip": SI_PRED_CLIP}, ROOT / "models/si_model.joblib")
    print("Saved models/si_model.joblib")


if __name__ == "__main__":
    main()