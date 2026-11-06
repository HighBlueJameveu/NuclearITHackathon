import pandas as pd
from src.evaluation.make_submission import make_submission
from src.evaluation.predictor import DummyPredictor

def test_submission_shape(tmp_path):
    test_csv = tmp_path / "test.csv"
    pd.DataFrame({"id": [1, 2, 3], "smiles": ["CCO", "c1ccccc1", "bad"]}).to_csv(test_csv, index=False)
    out = tmp_path / "sub.csv"
    sub = make_submission(DummyPredictor(5.0), str(test_csv), str(out))
    assert list(sub.columns) == ["id", "si_pred"]
    assert len(sub) == 3
    assert sub["si_pred"].notna().all()