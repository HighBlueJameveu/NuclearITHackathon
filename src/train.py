"""Обучение ансамбля моделей SI VACV.
Запуск: python -m src.train [--no-external]
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
SEEDS = [42, 7, 123, 2024, 999, 313, 777, 555, 888, 1337, 2025, 84]  # 12 моделей
COMP_WEIGHT = 2.5


def canon(smiles):
    from rdkit import Chem
    m = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(m) if m else None


def load_one_external(path, known):
    """Пытается прочитать внешний файл в двух форматах."""
    try:
        e = pd.read_csv(path, on_bad_lines="skip", engine="python")
    except Exception as ex:
        print(f"  не удалось прочитать {path.name}: {ex}")
        return None

    n0 = len(e)
    cols = set(e.columns)

    # Формат 1: SMILES + IC50_uM + CC50_uM
    if {"SMILES", "IC50_uM", "CC50_uM"} <= cols:
        e = e.dropna(subset=["SMILES", "IC50_uM", "CC50_uM"])
        e["IC50_uM"] = pd.to_numeric(e["IC50_uM"], errors="coerce")
        e["CC50_uM"] = pd.to_numeric(e["CC50_uM"], errors="coerce")
        e = e.dropna(subset=["IC50_uM", "CC50_uM"])
        e = e[(e.IC50_uM > 0) & (e.CC50_uM > 0)]
        e = e[~e.SMILES.map(canon).isin(known)]
        e["SI VACV"] = (e["CC50_uM"] / e["IC50_uM"]).clip(upper=SI_CLIP_EXTERNAL)
        e = e[(e["SI VACV"] >= 0.05) & (e["SI VACV"] <= SI_CLIP_EXTERNAL)]
        e["is_competition"] = False
        e["strain"] = "unknown"
        e["Culture"] = "unknown"
        e = e[["SMILES", "SI VACV", "is_competition", "strain", "Culture"]]
        print(f"External (IC50/CC50): {path.name}: {n0} -> {len(e)} rows")
        return e if len(e) > 0 else None

    # Формат 2: SMILES + SI VACV
    if {"SMILES", "SI VACV"} <= cols or {"SMILES", "SI_VACV"} <= cols:
        si_col = "SI VACV" if "SI VACV" in cols else "SI_VACV"
        e = e.dropna(subset=["SMILES", si_col])
        e = e.rename(columns={si_col: "SI VACV"})
        e["SI VACV"] = pd.to_numeric(e["SI VACV"], errors="coerce")
        e = e.dropna(subset=["SI VACV"])
        e = e[~e.SMILES.map(canon).isin(known)]
        e["SI VACV"] = e["SI VACV"].clip(upper=SI_CLIP_EXTERNAL)
        e = e[(e["SI VACV"] >= 0.05) & (e["SI VACV"] <= SI_CLIP_EXTERNAL)]
        e["is_competition"] = False
        if "strain" not in e.columns:
            e["strain"] = "unknown"
        if "Culture" not in e.columns:
            e["Culture"] = "unknown"
        e["strain"] = e["strain"].fillna("unknown").astype(str)
        e["Culture"] = e["Culture"].fillna("unknown").astype(str)
        e = e[["SMILES", "SI VACV", "is_competition", "strain", "Culture"]]
        print(f"External (SI ready): {path.name}: {n0} -> {len(e)} rows")
        return e if len(e) > 0 else None

    print(f"Пропускаю {path.name}: нет нужных колонок ({list(cols)[:6]}...)")
    return None


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
            e = load_one_external(f, known)
            if e is not None:
                parts.append(e)
                known |= set(e.SMILES.map(canon))

    return pd.concat(parts, ignore_index=True)


def main():
    use_ext = "--no-external" not in sys.argv
    tr = load_training_data(use_external=use_ext)
    comp = tr["is_competition"].to_numpy()

    X = build(tr, fit=True)
    y = np.log10(1.0 + tr["SI VACV"].to_numpy(dtype=float))
    groups = tr.SMILES.map(scaffold).to_numpy()
    sample_weight = np.where(comp, COMP_WEIGHT, 1.0)

    oof = np.zeros(len(y))
    for a, b in GroupKFold(5).split(X, y, groups):
        model = make_model(seed=42)
        model.fit(X[a], y[a], sample_weight=sample_weight[a])
        oof[b] = model.predict(X[b])

    pred_si = np.clip(10.0 ** oof - 1.0, 0.0, SI_PRED_CLIP)
    si_true = tr["SI VACV"].to_numpy(dtype=float)

    metrics = {
        "mode": f"ensemble {len(SEEDS)} seeds + weight={COMP_WEIGHT} + robust external",
        "train_rows_total": int(len(tr)),
        "eval_rows_competition": int(comp.sum()),
        "RMSE_log10(1+SI)": float(np.sqrt(np.mean((oof[comp] - y[comp]) ** 2))),
        "MAE_log10(1+SI)": float(np.mean(np.abs(oof[comp] - y[comp]))),
        "MAE_SI": float(np.mean(np.abs(pred_si[comp] - si_true[comp]))),
        "Spearman": float(pd.Series(oof[comp]).corr(pd.Series(y[comp]), method="spearman")),
    }
    print("Scaffold-CV (seed=42):", metrics)
    pd.Series(metrics).to_json(ROOT / "results/cv_metrics.json", indent=2)

    models = []
    for seed in SEEDS:
        print(f"Training model seed={seed} ...")
        model = make_model(seed=seed)
        model.fit(X, y, sample_weight=sample_weight)
        models.append(model)

    from src import features as F
    joblib.dump({
        "models": models,
        "seeds": SEEDS,
        "si_pred_clip": SI_PRED_CLIP,
        "strain_le": F._strain_le,
        "culture_le": F._culture_le,
    }, ROOT / "models/si_model.joblib")
    print(f"Saved ensemble of {len(models)} models -> models/si_model.joblib")


if __name__ == "__main__":
    main()