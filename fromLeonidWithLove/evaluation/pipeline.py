import yaml
import pandas as pd
from .predictor import build_predictor
from .evaluate import evaluate_to_json
from .make_submission import make_submission


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    predictor = build_predictor(cfg["model"])

    train = pd.read_csv(cfg["data"]["train_csv"])
    test  = pd.read_csv(cfg["data"]["test_csv"])

    y_true = train[cfg["data"]["target_col"]].values
    y_pred = predictor.predict(train[cfg["data"]["smiles_col"]].tolist())

    metrics = evaluate_to_json(y_true, y_pred, cfg["output"]["metrics_json"])
    print("Метрики ", metrics)

    make_submission(
        predictor=predictor,
        test_csv=cfg["data"]["test_csv"],
        out_csv=cfg["output"]["submission_csv"],
        id_col=cfg["data"]["id_col"],
        smiles_col=cfg["data"]["smiles_col"],
    )
    print(f"{cfg['output']['submission_csv']}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/evaluation.yaml")