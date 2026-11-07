import io
import re
import time

import numpy as np
import pandas as pd
import requests
from rdkit import Chem

from chembl_webresource_client.new_client import new_client


TARGET_IDS = ["CHEMBL613493"]          # Vaccinia virus (organism)
IC50_TYPES = ["IC50", "EC50"]

PUBCHEM_AIDS = [1959881, 217819]

OUT_PATH = "train_vacv_clean.csv"

ONLY_FIELDS = [
    "molecule_chembl_id",
    "standard_type",
    "standard_value",
    "standard_units",
    "standard_relation",
    "document_chembl_id",
    "canonical_smiles",
]

def parse_value(value):
    """Превращает значение в float. Квалификаторы (>, <, ~) отбрасываются"""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    s = re.sub(r"^[><~=≈≤≥\s]+", "", s)
    try:
        return float(s)
    except ValueError:
        return None


def to_micromolar(value, unit):
    if not unit:
        return np.nan
    v = parse_value(value)
    if v is None:
        return np.nan
    unit = str(unit).strip()
    if unit == "nM":
        return v / 1000.0
    if unit in ("uM", "µM", "μM"):
        return v
    return np.nan


def normalize_smiles(smiles):
    """Возвращает канонический SMILES или None, если структура невалидна"""
    if not smiles or pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def fetch_chembl_ic50():
    rows = []
    for tid in TARGET_IDS:
        print(f"ChEMBL: IC50 для target {tid}...")
        try:
            acts = new_client.activity.filter(
                target_chembl_id=tid,
                standard_type__in=IC50_TYPES,
                standard_units__in=["nM", "uM", "µM", "μM"],
            ).only(ONLY_FIELDS)
            for a in acts:
                rows.append(a)
        except Exception as e:
            print(f"  Ошибка: {e}")
        time.sleep(1)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Отбрасываем цензурированные значения
    if "standard_relation" in df.columns:
        df = df[
            df["standard_relation"].isin(["=", None])
            | df["standard_relation"].isna()
        ].copy()

    df = df[df["canonical_smiles"].notna()].copy()
    df["IC50_uM"] = df.apply(
        lambda r: to_micromolar(r["standard_value"], r["standard_units"]),
        axis=1,
    )
    df = df[df["IC50_uM"].notna() & (df["IC50_uM"] > 0)].copy()

    # Нормализуем SMILES
    df["SMILES"] = df["canonical_smiles"].apply(normalize_smiles)
    df = df[df["SMILES"].notna()].copy()

    # Агрегируем по молекуле
    agg = (
        df.groupby("molecule_chembl_id", as_index=False)
        .agg(
            SMILES=("SMILES", "first"),
            IC50_uM=("IC50_uM", "median"),
            document_chembl_id=("document_chembl_id", "first"),
        )
    )
    return agg


def fetch_chembl_cc50(molecule_ids):
    """
    Запрашивает CC50 для заданных molecule_chembl_id,
    не ограничиваясь target вируса
    """
    if not molecule_ids:
        return pd.DataFrame(columns=["SMILES", "CC50_uM"])

    rows = []
    # Разбиваем на чанки, чтобы не превысить лимит API
    for i in range(0, len(molecule_ids), 50):
        chunk = molecule_ids[i:i + 50]
        print(f"ChEMBL: CC50 для {len(chunk)} молекул (чанк {i // 50 + 1})...")
        try:
            acts = new_client.activity.filter(
                molecule_chembl_id__in=chunk,
                standard_type="CC50",
                standard_units__in=["nM", "uM", "µM", "μM"],
            ).only(ONLY_FIELDS)
            for a in acts:
                rows.append(a)
        except Exception as e:
            print(f"  Ошибка: {e}")
        time.sleep(1)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["SMILES", "CC50_uM"])

    if "standard_relation" in df.columns:
        df = df[
            df["standard_relation"].isin(["=", None])
            | df["standard_relation"].isna()
        ].copy()

    df = df[df["canonical_smiles"].notna()].copy()
    df["CC50_uM"] = df.apply(
        lambda r: to_micromolar(r["standard_value"], r["standard_units"]),
        axis=1,
    )
    df = df[df["CC50_uM"].notna() & (df["CC50_uM"] > 0)].copy()

    df["SMILES"] = df["canonical_smiles"].apply(normalize_smiles)
    df = df[df["SMILES"].notna()].copy()

    # Агрегируем: медиана по SMILES (на случай нескольких assay)
    agg = (
        df.groupby("SMILES", as_index=False)
        .agg(CC50_uM=("CC50_uM", "median"))
    )
    return agg


# ---------- 3. CC50 из PubChem ----------
def fetch_pubchem_cc50(aids):
    """
    Скачивает CSV для каждого AID и ищет колонку с CC50.
    Возвращает DataFrame с колонками SMILES, CC50_uM.
    """
    all_rows = []

    for aid in aids:
        print(f"PubChem: проверка AID {aid}...")
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/csv"
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
        except Exception as e:
            print(f"  Не удалось скачать: {e}")
            continue

        try:
            assay_df = pd.read_csv(io.StringIO(resp.text))
        except Exception as e:
            print(f"  Не удалось разобрать CSV: {e}")
            continue

        # Ищем колонку CC50
        cc50_col = None
        for col in assay_df.columns:
            if "CC50" in col.upper() or "CYTOTOX" in col.upper():
                cc50_col = col
                break

        if cc50_col is None:
            print(f"  Колонка CC50 не найдена. Колонки: {assay_df.columns.tolist()[:5]}...")
            continue

        # Ищем колонку SMILES
        smiles_col = None
        for col in assay_df.columns:
            if "SMILES" in col.upper():
                smiles_col = col
                break

        if smiles_col is None:
            print(f"  Колонка SMILES не найдена.")
            continue

        sub = assay_df[[smiles_col, cc50_col]].copy()
        sub.columns = ["SMILES_raw", "CC50_raw"]
        sub["CC50_uM"] = sub["CC50_raw"].apply(parse_value)
        sub = sub[sub["CC50_uM"].notna() & (sub["CC50_uM"] > 0)].copy()

        if sub.empty:
            continue

        # Нормализуем SMILES
        sub["SMILES"] = sub["SMILES_raw"].apply(normalize_smiles)
        sub = sub[sub["SMILES"].notna()].copy()

        if not sub.empty:
            print(f"  Найдено {len(sub)} строк с CC50.")
            all_rows.append(sub[["SMILES", "CC50_uM"]])

    if not all_rows:
        return pd.DataFrame(columns=["SMILES", "CC50_uM"])

    result = pd.concat(all_rows, ignore_index=True)
    # Медиана по SMILES, если соединение встречается в нескольких AID
    result = (
        result.groupby("SMILES", as_index=False)
        .agg(CC50_uM=("CC50_uM", "median"))
    )
    return result


def build_dataset():
    # 4.1. IC50 из ChEMBL
    ic50_df = fetch_chembl_ic50()
    if ic50_df.empty:
        raise ValueError("Не удалось загрузить IC50 из ChEMBL.")
    print(f"ChEMBL: молекул с IC50 = {len(ic50_df)}")

    # 4.2. CC50: сначала ChEMBL
    molecule_ids = ic50_df["molecule_chembl_id"].tolist()
    cc50_chembl = fetch_chembl_cc50(molecule_ids)
    print(f"ChEMBL: молекул с CC50 = {len(cc50_chembl)}")

    # 4.3. CC50: если ChEMBL дал мало — пробуем PubChem
    cc50_pubchem = pd.DataFrame(columns=["SMILES", "CC50_uM"])
    if len(cc50_chembl) < 10:  # порог можно менять
        print("ChEMBL дал мало CC50, пробуем PubChem...")
        cc50_pubchem = fetch_pubchem_cc50(PUBCHEM_AIDS)
        print(f"PubChem: молекул с CC50 = {len(cc50_pubchem)}")

    # 4.4. Объединяем CC50 из двух источников (если молекула есть в обоих — берём среднее)
    cc50_frames = [df for df in [cc50_chembl, cc50_pubchem] if not df.empty]
    if not cc50_frames:
        raise ValueError("Не удалось получить CC50 ни из ChEMBL, ни из PubChem.")

    cc50_all = pd.concat(cc50_frames, ignore_index=True)
    cc50_all = (
        cc50_all.groupby("SMILES", as_index=False)
        .agg(CC50_uM=("CC50_uM", "median"))
    )

    # 4.5. Объединяем IC50 и CC50 по нормализованному SMILES
    merged = pd.merge(ic50_df, cc50_all, on="SMILES", how="inner")

    if merged.empty:
        raise ValueError(
            "После объединения IC50 и CC50 не осталось общих молекул"
            "Проверьте нормализацию SMILES"
        )

    # 4.6. Считаем SI
    merged["SI_VACV"] = merged["CC50_uM"] / merged["IC50_uM"]

    # 4.7. Фильтрация
    merged = merged[
        merged["IC50_uM"].notna()
        & merged["CC50_uM"].notna()
        & merged["SI_VACV"].notna()
        & np.isfinite(merged["SI_VACV"])
        & (merged["SI_VACV"] > 0)
    ].copy()

    # 4.8. DOI
    doc_ids = merged["document_chembl_id"].dropna().unique().tolist()
    doi_map = {}
    if doc_ids:
        try:
            docs = new_client.document.filter(
                document_chembl_id__in=doc_ids
            ).only(["document_chembl_id", "doi"])
            for doc in docs:
                doi_map[doc["document_chembl_id"]] = doc.get("doi")
        except Exception as e:
            print(f"Ошибка загрузки DOI: {e}")

    merged["doi"] = merged["document_chembl_id"].map(doi_map)

    # 4.9. Финальный набор колонок
    final = merged[["SMILES", "IC50_uM", "CC50_uM", "SI_VACV", "doi"]].copy()
    final["source_ic50"] = "ChEMBL"
    final["source_cc50"] = "ChEMBL/PubChem"

    # Округление
    final["IC50_uM"] = final["IC50_uM"].round(6)
    final["CC50_uM"] = final["CC50_uM"].round(6)
    final["SI_VACV"] = final["SI_VACV"].round(4)

    final = final.drop_duplicates(subset=["SMILES"])
    return final


if __name__ == "__main__":
    df = build_dataset()
    df.to_csv(OUT_PATH, index=False)

    print(f"\nГотово. Сохранено {len(df)} строк в {OUT_PATH}")
    print("\nПервые 5 строк:")
    print(df.head())
    print("\nДиапазоны значений:")
    print(df[["IC50_uM", "CC50_uM", "SI_VACV"]].describe())
