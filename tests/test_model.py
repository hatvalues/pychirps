from data_preprocs.data_providers.cervical import cervicalb_pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from app.config import DEFAULT_RANDOM_SEED


def test_model_harness():
    X_train, X_test, y_train, y_test = train_test_split(
        cervicalb_pd.features,
        cervicalb_pd.target,
        test_size=0.2,
        random_state=DEFAULT_RANDOM_SEED,
    )
    model = RandomForestClassifier(n_estimators=100, random_state=DEFAULT_RANDOM_SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.5
