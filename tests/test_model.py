from data_preprocs.data_providers.cervical import cervicalb_pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from app.config import DEFAULT_RANDOM_SEED
import pytest


@pytest.mark.parametrize(
    "model_class,kwargs",
    [
        (
            RandomForestClassifier,
            dict(n_estimators=100, random_state=DEFAULT_RANDOM_SEED),
        ),
        (
            AdaBoostClassifier,
            dict(
                random_state=DEFAULT_RANDOM_SEED,
                estimator=DecisionTreeClassifier(max_depth=4),
                n_estimators=100,
                learning_rate=1.0,
            ),
        ),
    ],
)
def test_model_harness(model_class, kwargs):
    X_train, X_test, y_train, y_test = train_test_split(
        cervicalb_pd.features,
        cervicalb_pd.target,
        test_size=0.2,
        random_state=DEFAULT_RANDOM_SEED,
    )
    model = model_class(**kwargs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.5
    assert type(model) == model_class
