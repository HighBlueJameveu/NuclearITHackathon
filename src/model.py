from lightgbm import LGBMRegressor


def make_model():
    """Сильная регуляризация под маленький и шумный датасет."""
    return LGBMRegressor(
        n_estimators=800,
        learning_rate=0.02,
        num_leaves=8,
        min_child_samples=8,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.25,
        reg_lambda=8.0,
        reg_alpha=1.0,
        max_depth=5,
        random_state=42,
        verbose=-1,
    )