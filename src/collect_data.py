"""Сбор данных об активности малых молекул против вируса осповакцины (VACV) из ChEMBL.

Источник : ChEMBL REST API  https://www.ebi.ac.uk/chembl/api/data
Результат: data/external/chembl_vacv.csv  (+ data_report.json)
Колонки  : SMILES, IC50_uM, CC50_uM, SI, doi, source, document_chembl_id, molecule_chembl_id

Логика:
  1. Находим VACV-мишени и assay с «vaccinia» в описании -> множество VACV-assay и статей (documents).
  2. Из этих статей берём IC50/EC50 (только VACV-assay) и CC50/CD50/TC50 (цитотоксичность).
  3. Переводим всё в мкМ (нМ, мкМ, мМ, мкг/мл через молярную массу), оставляем только relation '='.
  4. Для пары (молекула, статья) берём геометрическое среднее, SI = CC50 / IC50.
  5. Валидация: корректные SMILES, положительные значения, SI пересчитан, дубли убраны.

Запуск: python -m src.collect_data [--max-docs N]
"""
import hashlib, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "external"
CACHE_DIR = OUT_DIR / "cache"
HOST = "https://www.ebi.ac.uk"
BASE = HOST + "/chembl/api/data"

IC_TYPES = {"IC50", "EC50"}
CC_TYPES = {"CC50", "CD50", "TC50"}
TO_UM = {"nM": 1e-3, "uM": 1.0, "mM": 1e3}
MASS_UNITS = {"ug.mL-1", "ug/ml", "ug ml-1", "ug/mL"}


# ----------------------------------------------------------------- HTTP / ChEMBL
def api_get(url, params=None, retries=4):
    """GET с повторами и файловым кэшем (повторный запуск не ходит в сеть)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5((url + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60, headers={"Accept": "application/json"})
            r.raise_for_status()
            data = r.json()
            path.write_text(json.dumps(data), encoding="utf-8")
            time.sleep(0.2)
            return data
        except Exception as e:
            if attempt == retries - 1:
                raise RuntimeError(f"ChEMBL request failed: {url} {params}: {e}")
            time.sleep(2 ** attempt)


def iterate(endpoint, key, params=None, page=1000):
    """Проходит все страницы ChEMBL-ответа и отдаёт записи."""
    url, p = f"{BASE}/{endpoint}.json", {**(params or {}), "limit": page}
    while url:
        data = api_get(url, p)
        yield from data.get(key, [])
        nxt = (data.get("page_meta") or {}).get("next")
        url, p = (HOST + nxt if nxt else None), None


def chunks(seq, n):
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# ----------------------------------------------------------------- шаги сбора
def find_vacv_assays():
    """Возвращает (множество assay_id, множество document_id) для VACV."""
    targets = {t["target_chembl_id"] for t in iterate("target/search", "targets", {"q": "vaccinia"})
               if "vaccinia" in f"{t.get('organism')} {t.get('pref_name')}".lower()}
    assays = {}
    for chunk in chunks(targets, 20):
        for a in iterate("assay", "assays", {"target_chembl_id__in": ",".join(chunk)}):
            assays[a["assay_chembl_id"]] = a.get("document_chembl_id")
    for a in iterate("assay", "assays", {"description__icontains": "vaccinia"}):
        assays[a["assay_chembl_id"]] = a.get("document_chembl_id")
    docs = {d for d in assays.values() if d}
    print(f"VACV targets: {len(targets)}, assays: {len(assays)}, documents: {len(docs)}")
    return set(assays), docs


def fetch_activities(docs):
    types = ",".join(sorted(IC_TYPES | CC_TYPES))
    rows = []
    for chunk in chunks(docs, 20):
        rows += list(iterate("activity", "activities",
                             {"document_chembl_id__in": ",".join(chunk), "standard_type__in": types}))
    print("Activity records:", len(rows))
    return rows


def fetch_dois(docs):
    out = {}
    for chunk in chunks(docs, 50):
        for d in iterate("document", "documents", {"document_chembl_id__in": ",".join(chunk)}):
            out[d["document_chembl_id"]] = d.get("doi")
    return out


# ----------------------------------------------------------------- обработка
def to_uM(value, units, smiles):
    """Перевод в мкМ. Для мкг/мл нужна молярная масса. Иначе None."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if units in TO_UM:
        return v * TO_UM[units]
    if units in MASS_UNITS:
        mol = Chem.MolFromSmiles(str(smiles))
        return v / Descriptors.MolWt(mol) * 1e3 if mol else None
    return None


def build_table(activities, vacv_assays, dois):
    recs = []
    for a in activities:
        if a.get("standard_relation") not in ("=", None) or not a.get("canonical_smiles"):
            continue
        st = a.get("standard_type")
        if st in IC_TYPES and a.get("assay_chembl_id") in vacv_assays:
            kind = "IC50_uM"
        elif st in CC_TYPES:
            kind = "CC50_uM"
        else:
            continue
        v = to_uM(a.get("standard_value"), a.get("standard_units"), a["canonical_smiles"])
        if v is None or v <= 0:
            continue
        recs.append({"SMILES": a["canonical_smiles"], "molecule_chembl_id": a.get("molecule_chembl_id"),
                     "document_chembl_id": a.get("document_chembl_id"), "kind": kind, "value": v})
    if not recs:
        return pd.DataFrame(columns=["SMILES", "IC50_uM", "CC50_uM", "SI", "doi", "source",
                                     "document_chembl_id", "molecule_chembl_id"])
    df = pd.DataFrame(recs)
    gm = lambda s: float(np.exp(np.log(s).mean()))            # геометрическое среднее
    wide = (df.groupby(["molecule_chembl_id", "document_chembl_id", "SMILES", "kind"])["value"]
              .apply(gm).unstack("kind").reset_index())
    for c in ("IC50_uM", "CC50_uM"):
        if c not in wide:
            wide[c] = np.nan
    wide["SI"] = wide["CC50_uM"] / wide["IC50_uM"]
    wide["doi"] = wide["document_chembl_id"].map(dois)
    wide["source"] = "ChEMBL"
    cols = ["SMILES", "IC50_uM", "CC50_uM", "SI", "doi", "source", "document_chembl_id", "molecule_chembl_id"]
    return wide[cols]


def canonical(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True)               # убираем соли/противоионы
    return Chem.MolToSmiles(max(frags, key=lambda m: m.GetNumHeavyAtoms()))


def validate(df, test_path=None):
    """Валидация: SMILES, положительные значения, SI пересчитан, дубли. Возвращает (df, report)."""
    rep = {"rows_raw": int(len(df))}
    df = df.copy()
    df["SMILES"] = df["SMILES"].map(canonical)
    rep["invalid_smiles"] = int(df["SMILES"].isna().sum())
    df = df.dropna(subset=["SMILES"])
    for c in ("IC50_uM", "CC50_uM"):
        df.loc[~(df[c] > 0), c] = np.nan
    df["SI"] = df["CC50_uM"] / df["IC50_uM"]
    rep["rows_without_both_ic50_cc50"] = int(df["SI"].isna().sum())
    df = df.dropna(subset=["SI"])
    before = len(df)
    df = df.drop_duplicates(subset=["SMILES", "doi", "IC50_uM", "CC50_uM"])
    rep["duplicates_removed"] = int(before - len(df))
    assert np.allclose(df["SI"], df["CC50_uM"] / df["IC50_uM"]), "SI != CC50/IC50"
    rep["rows_final"] = int(len(df))
    rep["unique_molecules"] = int(df["SMILES"].nunique())
    rep["unique_dois"] = int(df["doi"].nunique())
    if test_path and Path(test_path).exists():
        test_smiles = set(pd.read_csv(test_path)["SMILES"].map(canonical).dropna())
        rep["overlap_with_test_molecules"] = int(df["SMILES"].isin(test_smiles).sum())
    return df.reset_index(drop=True), rep


def main():
    max_docs = int(sys.argv[sys.argv.index("--max-docs") + 1]) if "--max-docs" in sys.argv else None
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assays, docs = find_vacv_assays()
    docs = sorted(docs)[:max_docs] if max_docs else sorted(docs)
    table = build_table(fetch_activities(docs), assays, fetch_dois(docs))
    table, report = validate(table, ROOT / "data" / "raw" / "test.csv")
    table.to_csv(OUT_DIR / "chembl_vacv.csv", index=False)
    (OUT_DIR / "data_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Saved", OUT_DIR / "chembl_vacv.csv")


if __name__ == "__main__":
    main()
