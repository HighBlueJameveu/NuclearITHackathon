"""Признаки только из SMILES."""
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator as G
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

_desc = [f for _, f in Descriptors.descList]
_morgan = G.GetMorganGenerator(radius=2, fpSize=1024)
_ap = G.GetAtomPairGenerator(fpSize=512)
N_FEATURES = len(_desc) + 167 + 1024 + 512


def mol_features(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.full(N_FEATURES, np.nan)
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


def build(smiles_series):
    return np.vstack([mol_features(s) for s in smiles_series])