from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np


class SiPredictor(ABC):

    @abstractmethod
    def predict(self, smiles: list[str]) -> np.ndarray:
        ...


class DummyPredictor(SiPredictor):

    def __init__(self, constant: float = 5.0):
        self.constant = float(constant)

    def predict(self, smiles: list[str]) -> np.ndarray:
        return np.full(len(smiles), self.constant, dtype=float)


class ModelPredictor(SiPredictor):

    def __init__(self, featurizer, model, log1p: bool = False):
        self.featurizer = featurizer
        self.model = model
        self.log1p = log1p

    def predict(self, smiles: list[str]) -> np.ndarray:
        X = self.featurizer.transform(smiles)
        y = self.model.predict(X)
        if self.log1p:
            y = np.expm1(y)
        return np.asarray(y, dtype=float)


def build_predictor(cfg: dict) -> SiPredictor:
    if cfg["type"] == "dummy":
        return DummyPredictor(constant=cfg.get("constant", 5.0))

    if cfg["type"] == "model":
        import joblib
        from pathlib import Path
        bundle = joblib.load(cfg["path"])
        return ModelPredictor(
            featurizer=bundle["featurizer"],
            model=bundle["model"],
            log1p=bundle.get("log1p", False),
        )

    raise ValueError(f"Неизвестный тип модели {cfg['type']}")