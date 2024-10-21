from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
import numpy as np


def fit_random_forest(
    X: np.ndarray, y: np.ndarray, random_state=12321
) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X, y)
    return model


def fit_adaboost(
    X: np.ndarray, y: np.ndarray, algorithm="SAMME", random_state=12321
) -> AdaBoostClassifier:
    model = AdaBoostClassifier(
        n_estimators=100, algorithm=algorithm, random_state=random_state
    )
    model.fit(X, y)
    return model


def fit_gradient_boosting(
    X: np.ndarray, y: np.ndarray, loss="log_loss", random_state=12321
) -> GradientBoostingClassifier:
    model = GradientBoostingClassifier(
        n_estimators=100, loss=loss, random_state=random_state
    )
    model.fit(X, y)
    return model
