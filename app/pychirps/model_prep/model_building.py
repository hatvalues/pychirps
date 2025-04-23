from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from app.config import DEFAULT_RANDOM_SEED


def fit_random_forest(
    X: np.ndarray, y: np.ndarray, random_state=DEFAULT_RANDOM_SEED, **kwargs
) -> RandomForestClassifier:
    hyper_parameters = {
        "n_estimators": 100,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "oob_score": True,
    } | kwargs
    model = RandomForestClassifier(random_state=random_state, **hyper_parameters)
    model.fit(X, y)
    return model


def fit_adaboost(
    X: np.ndarray, y: np.ndarray, random_state=DEFAULT_RANDOM_SEED, **kwargs
) -> AdaBoostClassifier:
    hyper_parameters = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 1,
    } | kwargs
    max_depth = hyper_parameters.pop("max_depth")
    max_features = hyper_parameters.pop("max_features", None)
    model = AdaBoostClassifier(
        algorithm="SAMME",
        random_state=random_state,
        estimator=DecisionTreeClassifier(
            max_depth=max_depth, max_features=max_features
        ),
        **hyper_parameters,
    )
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    model.fit(X, y)
    model.cv_scores = cv_scores
    model.mean_cv_score = np.mean(cv_scores)
    model.std_cv_score = np.std(cv_scores)
    return model


def fit_gradient_boosting(
    X: np.ndarray, y: np.ndarray, loss="log_loss", random_state=DEFAULT_RANDOM_SEED
) -> GradientBoostingClassifier:
    model = GradientBoostingClassifier(
        n_estimators=100, loss=loss, random_state=random_state
    )
    model.fit(X, y)
    return model
