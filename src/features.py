"""Признаки: SMILES (descriptors + MACCS + Morgan + AtomPair) + strain + Culture."""
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator as G
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.preprocessing import LabelEncoder

RDLogger.DisableLog("rdApp.*")

_desc = [f for _, f in Descriptors.descList]
_morgan = G.GetMorganGenerator(radius=2, fpSize=1024)
_ap = G.GetAtomPairGenerator(fpSize=512)
N_MOL = len(_desc) + 167 + 1024 + 512

# Глобальные энкодеры
_strain_le = None
_culture_le = None


def mol_features(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.full(N_MOL, np.nan)
    d = []
    for f in _desc:
        try:
            d.append(f(mol))
        except Exception:
            d.append(np.nan)
    d = np.clip(np.nan_to_num(d, nan=np.nan, posinf=1e6, neginf=-1e6), -1e6, 1e6)
    maccs = np.array(list(MACCSkeys.GenMACCSKeys(mol)), dtype=float)
    morgan = _morgan.GetCountFingerprintAsNumPy(mol).astype(float)
    ap = _ap.GetCountFingerprintAsNumPy(mol).astype(float)
    return np.concatenate([d, maccs, morgan, ap])


def scaffold(smiles):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(str(smiles)) or str(smiles)
    except Exception:
        return str(smiles)


def fit_encoders(df):
    global _strain_le, _culture_le
    _strain_le = LabelEncoder()
    _culture_le = LabelEncoder()
    strains = df["strain"].fillna("unknown").astype(str).tolist() + ["unknown"]
    cultures = df["Culture"].fillna("unknown").astype(str).tolist() + ["unknown"]
    _strain_le.fit(strains)
    _culture_le.fit(cultures)


def encode_cat(series, le):
    s = series.fillna("unknown").astype(str)
    known = set(le.classes_)
    s = s.map(lambda x: x if x in known else "unknown")
    return le.transform(s)


def build(df, fit=False):
    if fit:
        fit_encoders(df)

    X_mol = np.vstack([mol_features(s) for s in df["SMILES"]])

    if _strain_le is not None and "strain" in df.columns:
        strain_enc = encode_cat(df["strain"], _strain_le).reshape(-1, 1)
    else:
        strain_enc = np.zeros((len(df), 1))

    if _culture_le is not None and "Culture" in df.columns:
        culture_enc = encode_cat(df["Culture"], _culture_le).reshape(-1, 1)
    else:
        culture_enc = np.zeros((len(df), 1))

    return np.hstack([X_mol, strain_enc, culture_enc])