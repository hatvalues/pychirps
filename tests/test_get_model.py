from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from app.pychirps.model_prep.model_building import (
    fit_random_forest,
    fit_adaboost,
    fit_gradient_boosting,
)
from sklearn.datasets import make_classification
from typing import Union


X, y = make_classification(
    n_samples=1000,
    n_features=4,
    n_informative=2,
    n_redundant=0,
    random_state=0,
    shuffle=False,
)


def assert_model_fitted(
    model: Union[
        RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
    ],
):
    assert model.n_classes_ == 2
    assert model.n_features_in_ == 4


def test_fit_random_forest():
    model = fit_random_forest(X, y)
    assert isinstance(model, RandomForestClassifier)
    assert_model_fitted(model)


def test_fit_adaboost():
    model = fit_adaboost(X, y)
    assert isinstance(model, AdaBoostClassifier)
    assert_model_fitted(model)


def test_fit_gradient_boosting():
    model = fit_gradient_boosting(X, y)
    assert isinstance(model, GradientBoostingClassifier)
    assert_model_fitted(model)
