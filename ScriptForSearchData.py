# download_chembl_data.py
"""
Загрузка данных об активности малых молекул против VACV из ChEMBL.
Требование хакатона: SMILES, IC50, CC50, SI, doi.
"""

import pandas as pd
import numpy as np
from chembl_webresource_client.new_client import new_client

# ---------- 1. Настройки ----------
# Целевые ID ChEMBL, связанные с VACV (можно расширить)
TARGET_IDS = [
    "CHEMBL613493",   # Vaccinia virus (organism)
    # при необходимости добавьте другие ID штаммов
]

# Стандартные типы активностей, которые нас интересуют
ACTIVITY_TYPES = ["IC50", "CC50", "EC50"]  # EC50 иногда используется как IC50

# ---------- 2. Загрузка активностей ----------
def fetch_activities(target_ids, std_types):
    all_rows = []
    for tid in target_ids:
        print(f"Загрузка активностей для target {tid}...")
        activities = new_client.activity.filter(
            target_chembl_id=tid,
            standard_type__in=std_types,
            standard_units__in=["nM", "uM", "µM"],
        ).only([
            "molecule_chembl_id",
            "standard_type",
            "standard_value",
            "standard_units",
            "assay_chembl_id",
            "document_chembl_id",
            "canonical_smiles",
        ])
        for act in activities:
            all_rows.append(act)
    return pd.DataFrame(all_rows)

# ---------- 3. Получение DOI по document_chembl_id ----------
def get_doi(document_id):
    if not document_id:
        return None
    try:
        doc = new_client.document.get(document_id)
        return doc.get("doi")
    except Exception:
        return None

# ---------- 4. Приведение единиц к µM ----------
def to_micromolar(value, unit):
    if pd.isna(value) or not unit:
        return np.nan
    if unit in ("nM",):
        return value / 1000.0
    if unit in ("uM", "µM"):
        return value
    return np.nan

# ---------- 5. Основной конвейер ----------
def build_dataset():
    raw = fetch_activities(TARGET_IDS, ACTIVITY_TYPES)

    if raw.empty:
        raise ValueError("Не удалось загрузить активности. Проверьте target ID.")

    # Разделяем по типам
    ic50 = raw[raw["standard_type"] == "IC50"].copy()
    cc50 = raw[raw["standard_type"] == "CC50"].copy()

    # Приводим значения к µM
    ic50["IC50_uM"] = ic50.apply(
        lambda r: to_micromolar(r["standard_value"], r["standard_units"]), axis=1
    )
    cc50["CC50_uM"] = cc50.apply(
        lambda r: to_micromolar(r["standard_value"], r["standard_units"]), axis=1
    )

    # Оставляем нужные колонки
    ic50 = ic50[["molecule_chembl_id", "canonical_smiles", "IC50_uM", "document_chembl_id"]]
    cc50 = cc50[["molecule_chembl_id", "CC50_uM", "document_chembl_id"]]

    # Объединяем по молекуле (берём первую запись, если несколько)
    merged = pd.merge(
        ic50.drop_duplicates("molecule_chembl_id"),
        cc50.drop_duplicates("molecule_chembl_id"),
        on="molecule_chembl_id",
        how="inner",   # нужны обе величины для расчёта SI
        suffixes=("_ic50", "_cc50"),
    )

    # Если есть готовый SI в исходных данных — можно добавить, но здесь считаем сами
    merged["SI_VACV"] = merged["CC50_uM"] / merged["IC50_uM"]

    # Убираем некорректные значения
    merged = merged[
        (merged["IC50_uM"] > 0)
        & (merged["CC50_uM"] > 0)
        & (merged["SI_VACV"] > 0)
        & (merged["SI_VACV"].notna())
    ].copy()

    # Добавляем DOI (по одному из документов, например, по IC50)
    merged["doi"] = merged["document_chembl_id_ic50"].apply(get_doi)

    # Переименовываем SMILES
    merged = merged.rename(columns={"canonical_smiles": "SMILES"})

    # Финальный набор колонок
    final = merged[["SMILES", "IC50_uM", "CC50_uM", "SI_VACV", "doi"]].copy()
    final["source"] = "ChEMBL"

    # Удаляем дубликаты по SMILES
    final = final.drop_duplicates(subset=["SMILES"])

    return final

if __name__ == "__main__":
    df = build_dataset()
    out_path = "train_chembl.csv"
    df.to_csv(out_path, index=False)
    print(f"Готово. Сохранено {len(df)} строк в {out_path}")
    print(df.head())