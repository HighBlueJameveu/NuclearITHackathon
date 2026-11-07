from lightgbm import LGBMRegressor


def make_model(seed=42):
    return LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.02,
        num_leaves=12,
        min_child_samples=6,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.3,
        reg_lambda=6.0,
        reg_alpha=0.8,
        max_depth=6,
        random_state=seed,
        verbose=-1,
    )