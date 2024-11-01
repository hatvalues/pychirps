from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
import numpy as np
from src.config import DEFAULT_RANDOM_SEED


def fit_random_forest(
    X: np.ndarray, y: np.ndarray, random_state=DEFAULT_RANDOM_SEED
) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
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
