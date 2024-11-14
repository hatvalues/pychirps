from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
import numpy as np
from app.config import DEFAULT_RANDOM_SEED


def fit_random_forest(
    X: np.ndarray, y: np.ndarray, random_state=DEFAULT_RANDOM_SEED, **kwargs
) -> RandomForestClassifier:
    hyper_parameter_defaults = {
        "n_estimators": 100,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "oob_score": True,
    } | kwargs
    model = RandomForestClassifier(random_state=DEFAULT_RANDOM_SEED, **hyper_parameter_defaults)
    model.fit(X, y)
    return model


def fit_adaboost(
    X: np.ndarray, y: np.ndarray, algorithm="SAMME", random_state=DEFAULT_RANDOM_SEED
) -> AdaBoostClassifier:
    model = AdaBoostClassifier(
        n_estimators=100, algorithm=algorithm, random_state=random_state
    )
    model.fit(X, y)
    return model


def fit_gradient_boosting(
    X: np.ndarray, y: np.ndarray, loss="log_loss", random_state=DEFAULT_RANDOM_SEED
) -> GradientBoostingClassifier:
    model = GradientBoostingClassifier(
        n_estimators=100, loss=loss, random_state=random_state
    )
    model.fit(X, y)
    return model
